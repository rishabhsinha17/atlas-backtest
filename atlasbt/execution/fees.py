"""Broker commission + regulatory fee schedules (implements core.protocols.CostModel).

All schedules return a :class:`~atlasbt.core.types.CostBreakdown` per fill.
``commission`` is the broker's charge, ``exchange_fees`` holds exchange,
clearing (NSCC/DTC, OCC), ORF and pass-through fees, ``regulatory_fees`` holds
SEC Section 31 + FINRA TAF + CAT.

Schedules are *stateful* (month-to-date volume for tiered pricing and
per-order minimum/maximum accounting across partial fills), so one instance
must be used per backtest and called in chronological fill order (the engine
does this). Pass ``order_id`` so per-order minimums are applied once per order
rather than once per partial fill (IBKR applies minimums per order).

Sources and dates: see :mod:`atlasbt.execution.rates` and docs/ENGINE.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..core.types import AssetClass, CostBreakdown, Instrument, SettlementType
from . import rates

NON_TRADE_LIQUIDITY = {"EXERCISE", "ASSIGNMENT", "SETTLEMENT", "CORP_ACTION"}


def _date(ts) -> pd.Timestamp:
    return rates._as_naive_date(ts)


def _ceil_cent(x: float) -> float:
    if x <= 0:
        return 0.0
    return math.ceil(round(x * 100.0, 6)) / 100.0


# --------------------------------------------------------------------------
# Regulatory fees (shared by all broker schedules)
# --------------------------------------------------------------------------
def sec_fee(notional_sold: float, date, round_up_to_cent: bool = True) -> float:
    """SEC Section 31 fee on the dollar value of a covered SALE.

    ``notional_sold`` is the absolute dollar value sold (shares*price or
    contracts*premium*multiplier). Brokers pass the fee through rounded *up*
    to the next cent per trade (common industry practice)."""
    if notional_sold <= 0:
        return 0.0
    rate = rates.SEC31.at(date, "per_million")
    fee = abs(notional_sold) * rate / 1e6
    return _ceil_cent(fee) if round_up_to_cent else fee


def finra_taf_equity(shares_sold: float, date, price: Optional[float] = None,
                     round_up_to_cent: bool = True) -> float:
    """FINRA TAF on a covered equity SALE: per share, capped per trade.
    No fee if the execution price is below the per-share rate (rule text)."""
    if shares_sold <= 0:
        return 0.0
    row = rates.FINRA_TAF.row_at(date).values
    if price is not None and price < row["per_share"]:
        return 0.0
    fee = min(abs(shares_sold) * row["per_share"], row["max_per_trade"])
    return _ceil_cent(fee) if round_up_to_cent else fee


def finra_taf_options(contracts_sold: float, date, premium: Optional[float] = None,
                      round_up_to_cent: bool = True) -> float:
    if contracts_sold <= 0:
        return 0.0
    rate = rates.FINRA_TAF.at(date, "per_contract")
    if premium is not None and premium < rate:
        return 0.0
    fee = abs(contracts_sold) * rate
    return _ceil_cent(fee) if round_up_to_cent else fee


def is_cboe_spx(instrument: Instrument) -> Optional[str]:
    """Return 'SPX' or 'SPXW' for Cboe proprietary S&P 500 index options."""
    if not instrument.is_option:
        return None
    root = instrument.symbol[:6].strip().upper()
    und = (instrument.underlying or "").upper().lstrip("^$")
    if root == "SPXW" or (und == "SPX" and instrument.settlement == SettlementType.CASH_PM):
        return "SPXW"
    if root == "SPX" or und == "SPX":
        return "SPX"
    return None


def cboe_spx_fee(instrument: Instrument, contracts: float, premium: float, date) -> float:
    """Approximate Cboe customer transaction fee for SPX/SPXW (per contract:
    base fee by premium + execution surcharge)."""
    kind = is_cboe_spx(instrument)
    if kind is None:
        return 0.0
    row = rates.CBOE_SPX.row_at(date).values
    base = row["ge1"] if premium >= 1.0 else row["lt1"]
    sur = row["spx_surcharge"] if kind == "SPX" else row["spxw_surcharge"]
    return abs(contracts) * (base + sur)


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
@dataclass
class _OrderAcc:
    raw: float = 0.0
    value: float = 0.0
    charged: float = 0.0


class FeeSchedule:
    """Base class. Subclasses implement ``_stock_commission`` and
    ``_option_commission``; regulatory and clearing fees are shared."""

    name = "base"
    #: whether SEC/TAF fees are charged on sales (always true in the US)
    pass_through_regulatory = True
    #: extra option per-contract exchange fee for non-proprietary equity/ETF
    #: options (customer). Many venues charge customers $0 to add / small
    #: fees to remove; 0.10 is a conservative blended approximation.
    option_exchange_fee_per_contract = 0.10

    def __init__(self):
        self._month_key: Optional[tuple[int, int]] = None
        self._mtd_shares = 0.0
        self._mtd_contracts = 0.0
        self._orders: dict[int, _OrderAcc] = {}

    # -- protocol (core.protocols.CostModel)
    def explicit_costs(self, instrument: Instrument, quantity: float, price: float,
                       ts: pd.Timestamp, liquidity: str = "TAKER",
                       order_id: Optional[int] = None) -> CostBreakdown:
        date = _date(ts)
        self._roll_month(date)
        qty = abs(float(quantity))
        if qty == 0:
            return CostBreakdown()
        is_sale = quantity < 0
        if instrument.is_option:
            return self._option_costs(instrument, qty, abs(price), date, liquidity, is_sale, order_id)
        if instrument.asset_class in (AssetClass.STOCK, AssetClass.ETF):
            return self._stock_costs(instrument, qty, abs(price), date, liquidity, is_sale, order_id)
        return CostBreakdown()

    # -- helpers
    def _roll_month(self, date: pd.Timestamp) -> None:
        key = (date.year, date.month)
        if key != self._month_key:
            self._month_key = key
            self._mtd_shares = 0.0
            self._mtd_contracts = 0.0

    def _apply_min_max(self, order_id: Optional[int], raw: float, value: float,
                       minimum: float, max_frac: Optional[float]) -> float:
        """Per-order minimum / maximum accounting across partial fills.
        Returns the commission to charge for *this* fill."""
        acc = self._orders.setdefault(order_id, _OrderAcc()) if order_id is not None else _OrderAcc()
        acc.raw += raw
        acc.value += value
        total = max(acc.raw, minimum)
        if max_frac is not None:
            cap = max_frac * acc.value
            total = min(total, cap)
        charge = max(total - acc.charged, 0.0)
        acc.charged += charge
        return charge

    def _regulatory_stock(self, qty: float, price: float, date, is_sale: bool) -> float:
        if not (is_sale and self.pass_through_regulatory):
            return 0.0
        return sec_fee(qty * price, date) + finra_taf_equity(qty, date, price)

    def _regulatory_option(self, contracts: float, premium: float, mult: float, date, is_sale: bool) -> float:
        cat = contracts * rates.CAT_FEE.at(date, "per_contract")
        if not (is_sale and self.pass_through_regulatory):
            return cat
        return sec_fee(contracts * premium * mult, date) + finra_taf_options(contracts, date, premium) + cat

    def _option_exchange_fees(self, instrument: Instrument, contracts: float, premium: float, date) -> float:
        occ = contracts * rates.OCC_CLEARING.at(date, "per_contract")
        orf = contracts * rates.ORF_AGGREGATE.at(date, "per_contract")
        if is_cboe_spx(instrument):
            exch = cboe_spx_fee(instrument, contracts, premium, date)
        else:
            exch = contracts * self.option_exchange_fee_per_contract
        return occ + orf + exch

    def _option_costs(self, inst, contracts, premium, date, liquidity, is_sale, order_id) -> CostBreakdown:
        if liquidity in NON_TRADE_LIQUIDITY:
            return CostBreakdown()
        comm = self._option_commission(inst, contracts, premium, date, order_id)
        self._mtd_contracts += contracts
        exch = self._option_exchange_fees(inst, contracts, premium, date)
        reg = self._regulatory_option(contracts, premium, inst.multiplier, date, is_sale)
        return CostBreakdown(commission=comm, exchange_fees=exch, regulatory_fees=reg)

    def _stock_costs(self, inst, qty, price, date, liquidity, is_sale, order_id) -> CostBreakdown:
        if liquidity in NON_TRADE_LIQUIDITY:
            # exercise/assignment stock deliveries: no commission; SEC fee on sales
            reg = sec_fee(qty * price, date) if is_sale else 0.0
            return CostBreakdown(regulatory_fees=reg)
        comm, exch = self._stock_commission(inst, qty, price, date, liquidity, order_id)
        self._mtd_shares += qty
        reg = self._regulatory_stock(qty, price, date, is_sale)
        return CostBreakdown(commission=comm, exchange_fees=exch, regulatory_fees=reg)

    # -- to implement
    def _stock_commission(self, inst, qty, price, date, liquidity, order_id) -> tuple[float, float]:
        raise NotImplementedError

    def _option_commission(self, inst, contracts, premium, date, order_id) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------
# IBKR Pro option commission (same table for Fixed and Tiered accounts)
# --------------------------------------------------------------------------
def ibkr_option_rate(premium: float, mtd_contracts: float) -> float:
    """IBKR Pro US option commission per contract (2026 schedule).

    <=10,000 contracts/month: premium>=0.10 -> 0.65; 0.05<=p<0.10 -> 0.50; p<0.05 -> 0.25
    10,001-50,000:            p>=0.05 -> 0.50; p<0.05 -> 0.25
    50,001-100,000:           0.25
    >=100,001:                0.15
    """
    if mtd_contracts < 10_000:
        if premium >= 0.10:
            return 0.65
        return 0.50 if premium >= 0.05 else 0.25
    if mtd_contracts < 50_000:
        return 0.50 if premium >= 0.05 else 0.25
    if mtd_contracts < 100_000:
        return 0.25
    return 0.15


class _IBKROptionsMixin:
    option_min_per_order = 1.00

    def _option_commission(self, inst, contracts, premium, date, order_id) -> float:  # type: ignore[override]
        raw = 0.0
        remaining = contracts
        mtd = self._mtd_contracts  # type: ignore[attr-defined]
        # marginal tiering within the fill
        for lo, hi in ((0, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, math.inf)):
            if remaining <= 0:
                break
            if mtd >= hi:
                continue
            take = min(remaining, hi - max(mtd, lo))
            raw += take * ibkr_option_rate(premium, max(mtd, lo))
            mtd += take
            remaining -= take
        return self._apply_min_max(order_id, raw, contracts * premium * inst.multiplier,  # type: ignore[attr-defined]
                                   self.option_min_per_order, None)


class IBKRProFixed(_IBKROptionsMixin, FeeSchedule):
    """IBKR Pro *Fixed*: USD 0.005/share, min USD 1.00, max 1% of trade value.
    Exchange and clearing fees are included; SEC + FINRA TAF passed through on
    sales (IBKR: "Transaction fees ... passed through on all stock sales").
    Published examples: 100 sh @ $25 = $1.00; 1,000 sh @ $25 = $5.00;
    1,000 sh @ $0.25 = $2.50; 10 sh @ $0.20 = $0.02 (max below min)."""

    name = "IBKR_PRO_FIXED"
    per_share = 0.005
    min_per_order = 1.00
    max_frac = 0.01

    def _stock_commission(self, inst, qty, price, date, liquidity, order_id):
        raw = qty * self.per_share
        comm = self._apply_min_max(order_id, raw, qty * price, self.min_per_order, self.max_frac)
        return comm, 0.0


TIERS_STOCK = ((300_000, 0.0035), (3_000_000, 0.0020), (20_000_000, 0.0015),
               (100_000_000, 0.0010), (math.inf, 0.0005))


class IBKRProTiered(_IBKROptionsMixin, FeeSchedule):
    """IBKR Pro *Tiered* (default schedule).

    Commission by month-to-date shares (marginal): <=300k 0.0035; 300k-3M
    0.0020; 3M-20M 0.0015; 20M-100M 0.0010; >100M 0.0005. Min USD 0.35, max 1%
    of trade value per order. Plus: exchange fee (taker: Reg NMS access-fee cap
    by date; maker: 0, rebates ignored = conservative; auction: 0.0010/sh
    approx), clearing NSCC/DTC 0.00020/sh, NYSE pass-through 0.000175 x
    commission, FINRA pass-through 0.00056 x commission, regulatory SEC/TAF on
    sales, CAT fee."""

    name = "IBKR_PRO_TIERED"
    min_per_order = 0.35
    max_frac = 0.01
    clearing_per_share = 0.00020
    nyse_pass_through = 0.000175
    finra_pass_through = 0.00056
    auction_fee_per_share = 0.0010
    maker_fee_per_share = 0.0

    def tiered_raw(self, qty: float) -> float:
        raw = 0.0
        remaining = qty
        mtd = self._mtd_shares
        lo = 0.0
        for hi, rate in TIERS_STOCK:
            if remaining <= 0:
                break
            if mtd < hi:
                take = min(remaining, hi - max(mtd, lo))
                raw += take * rate
                mtd += take
                remaining -= take
            lo = hi
        return raw

    def _stock_commission(self, inst, qty, price, date, liquidity, order_id):
        raw = self.tiered_raw(qty)
        comm = self._apply_min_max(order_id, raw, qty * price, self.min_per_order, self.max_frac)
        if liquidity == "MAKER":
            exch = qty * self.maker_fee_per_share
        elif liquidity == "AUCTION":
            exch = qty * self.auction_fee_per_share
        else:
            access = rates.EQUITY_TAKER_FEE.at(date, "per_share")
            if price < 1.0:
                access = min(access, 0.003 * price)   # Rule 610: 0.3% of price below $1
            exch = qty * access
        exch += qty * self.clearing_per_share
        exch += comm * (self.nyse_pass_through + self.finra_pass_through)
        cat = qty * rates.CAT_FEE.at(date, "per_share")
        # CAT recorded with exchange/clearing pass-throughs
        return comm, exch + cat


class ZeroCommissionRetail(FeeSchedule):
    """Zero-commission retail broker (Robinhood/Schwab/Fidelity style).

    From ``zero_since`` (default 2019-10-07, Schwab's $0 date) stock/ETF
    commission is $0 and option commission ``option_per_contract`` (default
    $0.00, Robinhood style; use 0.65 for Schwab/Fidelity). Before that date,
    if ``historical=True``, a flat ``legacy_per_trade`` ($4.95) per order plus
    $0.65/contract applies. SEC fee and FINRA TAF are always passed through;
    OCC/ORF are passed through on options. Payment-for-order-flow execution
    quality is NOT modelled as price improvement (conservative)."""

    name = "ZERO_COMMISSION_RETAIL"
    option_exchange_fee_per_contract = 0.0

    def __init__(self, historical: bool = True, zero_since=rates.ZERO_COMMISSION_ERA,
                 legacy_per_trade: float = 4.95, legacy_per_contract: float = 0.65,
                 option_per_contract: float = 0.0):
        super().__init__()
        self.historical = historical
        self.zero_since = pd.Timestamp(zero_since)
        self.legacy_per_trade = legacy_per_trade
        self.legacy_per_contract = legacy_per_contract
        self.option_per_contract = option_per_contract

    def _legacy(self, date) -> bool:
        return self.historical and date < self.zero_since

    def _stock_commission(self, inst, qty, price, date, liquidity, order_id):
        if self._legacy(date):
            return self._apply_min_max(order_id, 0.0, qty * price, self.legacy_per_trade, None), 0.0
        return 0.0, 0.0

    def _option_commission(self, inst, contracts, premium, date, order_id):
        if self._legacy(date):
            return self._apply_min_max(order_id, contracts * self.legacy_per_contract,
                                       contracts * premium * inst.multiplier, self.legacy_per_trade, None)
        return contracts * self.option_per_contract


class NoFees(FeeSchedule):
    """For mechanics tests / gross-of-cost studies only."""

    name = "NO_FEES"
    pass_through_regulatory = False
    option_exchange_fee_per_contract = 0.0

    def explicit_costs(self, instrument, quantity, price, ts, liquidity="TAKER", order_id=None):
        return CostBreakdown()

    def _stock_commission(self, *a, **k):
        return 0.0, 0.0

    def _option_commission(self, *a, **k):
        return 0.0


def make_fee_schedule(name: str = "IBKR_PRO_TIERED", **kw) -> FeeSchedule:
    table = {
        "IBKR_PRO_TIERED": IBKRProTiered,
        "IBKR_PRO_FIXED": IBKRProFixed,
        "ZERO_COMMISSION_RETAIL": ZeroCommissionRetail,
        "NO_FEES": NoFees,
    }
    try:
        cls = table[name.upper()]
    except KeyError:
        raise ValueError(f"unknown fee schedule {name!r}; choose from {sorted(table)}") from None
    return cls(**kw)

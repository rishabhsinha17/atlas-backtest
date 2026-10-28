"""The 13 sizing profiles of the Atlas sheet ("Sizing Profiles"), verbatim
defaults, as plain functions usable from any strategy.

Every function is pure given its inputs (NAV, prices, current positions), so
it can be unit-tested without an engine; the ``ctx``-taking variants read
those inputs PIT from the strategy Context. Quantities returned are SIGNED
(+ long / - short) shares or contracts; 0 means "do not trade".

Interpretations that the sheet leaves open are documented on each function
and in docs/ENGINE.md (sizing section).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from ..core.clock import et
from ..core.types import Instrument, Right

# ------------------------------------------------------------------ sheet text
PROFILE_TEXT: dict[str, str] = {
    "DAY_RISK": "Risk at initial planned stop = 0.25% of prior-day NAV; gross single-name cap = 10% NAV; no new entries after 15:15 ET, flat by 15:55 ET.",
    "DAY_NOTIONAL": "Notional = 10% of prior-day NAV per instrument; one open position per instrument; exit before RTH end.",
    "SWING_RISK": "Risk at initial stop = 0.50% current NAV; notional <=10% NAV; aggregate equity gross <=100% unless stated.",
    "SWING_NOTIONAL": "Long notional 10% of NAV per instrument (shorts +/-5% NAV gross); no implicit leverage.",
    "CROSS_SECTION": "Long basket +50% NAV, short basket -50% NAV, equal weights among selected names, sector caps +/-10% and single-name <=2% NAV; cash/borrow and margin modeled.",
    "PORTFOLIO": "Weights sum to <=100% unless explicit leverage; residual in historical short Treasury/cash earning financing rate; next-session rebalancing.",
    "PAIRS": "Dollar-neutral pair gross 10% NAV (5% long+5% short); hedge ratio from prior formation only; daily borrow costs.",
    "OPT_DEBIT": "Each package max initial debit <=0.50% NAV; max portfolio option debit exposure 5% NAV; integer contracts, multiplier/deliverable verified.",
    "OPT_CREDIT": "Each package MAX theoretical defined loss <=0.50% NAV and collateral reserved; integer contracts; no naked leg exposure.",
    "OPT_NAKED": "Margin as broker/exchange schedule at historic time; cap worst-case stress loss to 1% NAV on predefined scenario but unbounded risks remain; never assume zero margin.",
    "OPT_COVERED": "100 shares per call contract; shares + options combined P&L and dividends; cap total underlying gross exposure 10% NAV.",
    "OPT_CASHSEC": "100*put strike USD reserved per standard contract or equivalent deliverable cash; other equity gross caps apply; margin cannot be double counted.",
    "OPT_ARBITRAGE": "Position only if executable quoted market after fees, borrow, funding and margin shows positive modeled edge; short-leg locates required.",
}


@dataclass(frozen=True)
class ProfileDefaults:
    risk_frac: float = 0.0
    notional_frac: float = 0.0
    short_notional_frac: float = 0.0
    single_name_cap: float = 0.0
    gross_cap: float = 1.0
    no_new_entries_after: Optional[str] = None
    flat_by: Optional[str] = None


PROFILES: dict[str, ProfileDefaults] = {
    "DAY_RISK": ProfileDefaults(risk_frac=0.0025, single_name_cap=0.10, no_new_entries_after="15:15", flat_by="15:55"),
    "DAY_NOTIONAL": ProfileDefaults(notional_frac=0.10, flat_by="15:55"),
    "SWING_RISK": ProfileDefaults(risk_frac=0.005, notional_frac=0.10, gross_cap=1.0),
    "SWING_NOTIONAL": ProfileDefaults(notional_frac=0.10, short_notional_frac=0.05, gross_cap=1.0),
    "CROSS_SECTION": ProfileDefaults(notional_frac=0.50, short_notional_frac=0.50, single_name_cap=0.02),
    "PORTFOLIO": ProfileDefaults(gross_cap=1.0),
    "PAIRS": ProfileDefaults(notional_frac=0.05, short_notional_frac=0.05),
    "OPT_DEBIT": ProfileDefaults(risk_frac=0.005, gross_cap=0.05),
    "OPT_CREDIT": ProfileDefaults(risk_frac=0.005),
    "OPT_NAKED": ProfileDefaults(risk_frac=0.01),
    "OPT_COVERED": ProfileDefaults(single_name_cap=0.10),
    "OPT_CASHSEC": ProfileDefaults(single_name_cap=0.10),
    "OPT_ARBITRAGE": ProfileDefaults(),
}


def _whole(x: float, allow_fractional: bool = False) -> float:
    """Truncate toward zero to whole units (shares / contracts)."""
    if not math.isfinite(x):
        return 0.0
    if allow_fractional:
        return x
    return float(math.trunc(x + (1e-9 if x > 0 else -1e-9)))


def _gross_equity(positions: Mapping[str, float], prices: Mapping[str, float],
                  exclude: Iterable[str] = ()) -> float:
    ex = set(exclude)
    return sum(abs(q) * prices.get(s, float("nan")) for s, q in positions.items()
               if s not in ex and prices.get(s, float("nan")) == prices.get(s, float("nan")))


# ===================================================================== DAY_RISK
def size_day_risk(prior_nav: float, entry: float, stop: float, *, existing_qty: float = 0.0,
                  risk_frac: float = 0.0025, single_name_cap: float = 0.10,
                  now_et: Optional[pd.Timestamp] = None, cutoff: str = "15:15",
                  allow_fractional: bool = False) -> float:
    """DAY_RISK: shares = 0.25% * prior-day NAV / |entry - stop|, signed by
    direction (entry > stop -> long, entry < stop -> short), capped so that
    the single-name gross (existing + new) <= 10% of prior-day NAV. Returns 0
    after 15:15 ET (no new entries) or for a degenerate stop.
    Interpretation: the 10% cap is measured on prior-day NAV (the same base as
    the risk budget) at the planned entry price."""
    if now_et is not None:
        hh, mm = (int(x) for x in cutoff.split(":"))
        t = pd.Timestamp(now_et)
        if (t.hour, t.minute, t.second) > (hh, mm, 0):
            return 0.0
    risk_ps = abs(entry - stop)
    if not (prior_nav > 0 and entry > 0 and risk_ps > 0):
        return 0.0
    q = risk_frac * prior_nav / risk_ps
    cap = max(single_name_cap * prior_nav / entry - abs(existing_qty), 0.0)
    q = min(q, cap)
    sign = 1.0 if entry > stop else -1.0
    return sign * _whole(q, allow_fractional)


# ================================================================= DAY_NOTIONAL
def size_day_notional(prior_nav: float, price: float, side: int = 1, *, existing_qty: float = 0.0,
                      frac: float = 0.10, allow_fractional: bool = False) -> float:
    """DAY_NOTIONAL: 10% of prior-day NAV per instrument; one open position
    per instrument -> 0 if a position already exists. Exit before RTH end is
    enforced by the engine (flat_by 15:55 ET)."""
    if existing_qty != 0 or not (prior_nav > 0 and price > 0):
        return 0.0
    return (1 if side >= 0 else -1) * _whole(frac * prior_nav / price, allow_fractional)


# =================================================================== SWING_RISK
def size_swing_risk(nav: float, entry: float, stop: float, *, current_gross: float = 0.0,
                    existing_notional: float = 0.0, risk_frac: float = 0.005, notional_cap: float = 0.10,
                    gross_cap: float = 1.0, allow_fractional: bool = False) -> float:
    """SWING_RISK: shares = 0.50% * current NAV / |entry - stop|; position
    notional (existing + new) <= 10% NAV; aggregate equity gross (all names,
    after the trade) <= 100% NAV."""
    risk_ps = abs(entry - stop)
    if not (nav > 0 and entry > 0 and risk_ps > 0):
        return 0.0
    q = risk_frac * nav / risk_ps
    q = min(q, max(notional_cap * nav - abs(existing_notional), 0.0) / entry)
    q = min(q, max(gross_cap * nav - current_gross, 0.0) / entry)
    sign = 1.0 if entry > stop else -1.0
    return sign * _whole(q, allow_fractional)


# =============================================================== SWING_NOTIONAL
def size_swing_notional(nav: float, price: float, side: int = 1, *, current_gross: float = 0.0,
                        long_frac: float = 0.10, short_frac: float = 0.05, gross_cap: float = 1.0,
                        allow_fractional: bool = False) -> float:
    """SWING_NOTIONAL: long 10% NAV per instrument, short 5% NAV; no implicit
    leverage (aggregate gross after the trade <= 100% NAV)."""
    if not (nav > 0 and price > 0):
        return 0.0
    frac = long_frac if side >= 0 else short_frac
    notional = min(frac * nav, max(gross_cap * nav - current_gross, 0.0))
    return (1 if side >= 0 else -1) * _whole(notional / price, allow_fractional)


# ================================================================ CROSS_SECTION
def cross_section_weights(longs: Sequence[str], shorts: Sequence[str],
                          sectors: Optional[Mapping[str, str]] = None, *, long_gross: float = 0.50,
                          short_gross: float = 0.50, name_cap: float = 0.02, sector_cap: float = 0.10,
                          sector_cap_mode: str = "per_side") -> dict[str, float]:
    """CROSS_SECTION target weights (fraction of NAV, signed).

    Equal weight within each basket (+50% / -50%), single name capped at 2%,
    sector caps +/-10%. ``sector_cap_mode='per_side'`` (default): each side's
    weight within a sector is capped at 10% NAV; ``'net'``: the sector's net
    weight is capped at +/-10% by scaling down the heavier side. Weight removed
    by caps is NOT redistributed (it stays in cash) so names keep equal
    weights; this is the conservative reading of "equal weights"."""
    w: dict[str, float] = {}
    if longs:
        wl = min(long_gross / len(longs), name_cap)
        for s in longs:
            w[s] = w.get(s, 0.0) + wl
    if shorts:
        ws = min(short_gross / len(shorts), name_cap)
        for s in shorts:
            w[s] = w.get(s, 0.0) - ws
    if sectors:
        by: dict[str, list[str]] = {}
        for s in w:
            by.setdefault(sectors.get(s, "UNKNOWN"), []).append(s)
        for sec, names in by.items():
            if sector_cap_mode == "net":
                net = sum(w[s] for s in names)
                if abs(net) > sector_cap + 1e-12:
                    heavy = [s for s in names if (w[s] > 0) == (net > 0)]
                    tot = sum(abs(w[s]) for s in heavy)
                    excess = abs(net) - sector_cap
                    f = max(tot - excess, 0.0) / tot if tot else 0.0
                    for s in heavy:
                        w[s] *= f
            else:
                for sgn in (1, -1):
                    side = [s for s in names if w[s] * sgn > 0]
                    tot = sum(abs(w[s]) for s in side)
                    if tot > sector_cap + 1e-12:
                        f = sector_cap / tot
                        for s in side:
                            w[s] *= f
    return w


def weights_to_shares(weights: Mapping[str, float], nav: float, prices: Mapping[str, float],
                      allow_fractional: bool = False) -> dict[str, float]:
    out = {}
    for s, wt in weights.items():
        p = prices.get(s, float("nan"))
        if not (p == p and p > 0):
            continue
        out[s] = _whole(wt * nav / p, allow_fractional)
    return out


# ==================================================================== PORTFOLIO
def portfolio_weights(weights: Mapping[str, float], *, max_gross: float = 1.0,
                      allow_short: bool = False) -> tuple[dict[str, float], float]:
    """PORTFOLIO: scale weights so that sum(|w|) <= 100% (unless explicit
    leverage via ``max_gross``); returns (weights, residual cash weight). The
    residual earns the financing (T-bill minus spread) rate through the
    account's cash interest. Rebalance orders submitted at a session close
    fill at the NEXT session (engine semantics)."""
    w = {s: (v if allow_short else max(v, 0.0)) for s, v in weights.items() if v == v}
    g = sum(abs(v) for v in w.values())
    if g > max_gross and g > 0:
        f = max_gross / g
        w = {s: v * f for s, v in w.items()}
    resid = 1.0 - sum(w.values())
    return w, resid


def rebalance_deltas(target_shares: Mapping[str, float], current: Mapping[str, float],
                     min_trade_shares: float = 1.0) -> dict[str, float]:
    syms = sorted(set(target_shares) | {s for s, q in current.items() if q})
    out = {}
    for s in syms:
        d = target_shares.get(s, 0.0) - current.get(s, 0.0)
        if abs(d) >= min_trade_shares:
            out[s] = d
    return out


# ======================================================================== PAIRS
def size_pairs(nav: float, price_long: float, price_short: float, *, gross: float = 0.10,
               allow_fractional: bool = False) -> tuple[float, float]:
    """PAIRS: dollar-neutral, gross 10% NAV = 5% long + 5% short. Returns
    (long shares > 0, short shares < 0). The hedge ratio from the prior
    formation window is used for SIGNALS; sizing stays dollar-neutral as
    the sheet states."""
    if not (nav > 0 and price_long > 0 and price_short > 0):
        return 0.0, 0.0
    leg = 0.5 * gross * nav
    return _whole(leg / price_long, allow_fractional), -_whole(leg / price_short, allow_fractional)


# ==================================================================== OPT_DEBIT
def verify_contract(inst: Instrument, expected_multiplier: float = 100.0,
                    expected_deliverable: float = 100.0) -> tuple[bool, str]:
    if not inst.is_option:
        return False, "not an option"
    if inst.multiplier != expected_multiplier:
        return False, f"multiplier {inst.multiplier} != {expected_multiplier}"
    if inst.deliverable_shares != expected_deliverable:
        return False, f"non-standard deliverable {inst.deliverable_shares}"
    return True, ""


def size_opt_debit(nav: float, debit_per_package: float, *, existing_option_debit: float = 0.0,
                   pkg_frac: float = 0.005, total_frac: float = 0.05) -> int:
    """OPT_DEBIT: ``debit_per_package`` in USD per ONE package (premium x
    multiplier, all legs, incl. fees if desired). Contracts n (integer) with
    n * debit <= 0.50% NAV and existing + new option debit <= 5% NAV."""
    if not (nav > 0 and debit_per_package > 0):
        return 0
    budget = min(pkg_frac * nav, max(total_frac * nav - existing_option_debit, 0.0))
    return int(math.floor(budget / debit_per_package + 1e-9))


# =================================================================== OPT_CREDIT
@dataclass(frozen=True)
class OptLeg:
    instrument: Instrument
    ratio: int                    # + long / - short contracts per package
    price: float = float("nan")   # per-share premium (for credit computation)


def is_defined_risk(legs: Sequence[OptLeg]) -> tuple[bool, str]:
    """No naked leg exposure: for each (underlying, right), the number of
    short contracts must be covered by long contracts of the same right
    expiring no earlier (verticals / calendars / condors / flies)."""
    groups: dict[tuple, list[OptLeg]] = {}
    for l in legs:
        i = l.instrument
        if not i.is_option:
            continue
        groups.setdefault((i.underlying, i.right), []).append(l)
    for key, ls in groups.items():
        shorts = sorted([l for l in ls if l.ratio < 0], key=lambda l: l.instrument.expiry)
        longs = [l for l in ls if l.ratio > 0]
        for s in shorts:
            cover = sum(l.ratio for l in longs if l.instrument.expiry >= s.instrument.expiry)
            need = sum(-x.ratio for x in shorts if x.instrument.expiry <= s.instrument.expiry)
            if cover < need:
                return False, f"naked short {key[1].value if key[1] else ''} on {key[0]}"
    return True, ""


def max_loss_vertical(legs: Sequence[OptLeg]) -> float:
    """Max theoretical loss (USD per package) of a same-expiry defined-risk
    package (verticals, iron condors, butterflies) at expiry, computed by
    evaluating the payoff on all strikes (piecewise-linear payoff) minus the
    net credit received (prices per share)."""
    strikes = sorted({l.instrument.strike for l in legs})
    if not strikes:
        return 0.0
    grid = [0.0] + strikes + [strikes[-1] * 2 + 1.0]
    net_premium = -sum(l.ratio * l.price for l in legs)        # credit > 0
    mult = legs[0].instrument.multiplier
    worst = math.inf
    for s in grid:
        pay = 0.0
        for l in legs:
            k = l.instrument.strike
            iv = max(s - k, 0.0) if l.instrument.right == Right.CALL else max(k - s, 0.0)
            pay += l.ratio * iv
        worst = min(worst, pay + net_premium)
    # beyond the top strike calls are linear; check slope for unbounded loss
    slope = sum(l.ratio for l in legs if l.instrument.right == Right.CALL)
    if slope < 0:
        return math.inf
    return max(-worst, 0.0) * mult


def size_opt_credit(nav: float, max_loss_per_package: float, *, frac: float = 0.005,
                    legs: Optional[Sequence[OptLeg]] = None) -> tuple[int, float]:
    """OPT_CREDIT: n = floor(0.50% NAV / max defined loss per package).
    Returns (n, collateral to reserve = n * max_loss). Rejects (0) packages
    with naked leg exposure when ``legs`` are given."""
    if legs is not None:
        ok, _ = is_defined_risk(legs)
        if not ok:
            return 0, 0.0
    if not (nav > 0 and max_loss_per_package > 0 and math.isfinite(max_loss_per_package)):
        return 0, 0.0
    n = int(math.floor(frac * nav / max_loss_per_package + 1e-9))
    return n, n * max_loss_per_package


# ==================================================================== OPT_NAKED
@dataclass(frozen=True)
class StressScenario:
    """Predefined stress: underlying moves by each of ``moves`` (relative) and
    option extrinsic value scales by ``vol_up_extrinsic_mult``."""

    moves: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)
    vol_up_extrinsic_mult: float = 2.0


DEFAULT_STRESS = StressScenario()


def stress_loss_short_option(inst: Instrument, spot: float, premium: float,
                             scenario: StressScenario = DEFAULT_STRESS) -> float:
    """Worst-case loss (USD per short contract) under the scenario: option
    value after shock ~ intrinsic(shocked spot) + extrinsic * vol mult, minus
    the premium received. (A pricing-model based stress from the OPTIONS agent
    can replace this function.)"""
    k = inst.strike or 0.0
    iv0 = max(spot - k, 0.0) if inst.right == Right.CALL else max(k - spot, 0.0)
    extr = max(premium - iv0, 0.0)
    worst = 0.0
    for m in scenario.moves:
        s = spot * (1 + m)
        iv = max(s - k, 0.0) if inst.right == Right.CALL else max(k - s, 0.0)
        val = iv + extr * scenario.vol_up_extrinsic_mult
        worst = max(worst, val - premium)
    return worst * inst.multiplier


def size_opt_naked(nav: float, stress_loss_per_contract: float, margin_per_contract: float, *,
                   frac: float = 0.01) -> int:
    """OPT_NAKED: contracts so that the predefined-scenario stress loss <= 1%
    NAV. Raises if the margin requirement is zero (never assume zero margin)."""
    if not (margin_per_contract > 0):
        raise ValueError("OPT_NAKED: margin requirement must be > 0 (never assume zero margin)")
    if not (nav > 0 and stress_loss_per_contract > 0):
        return 0
    return int(math.floor(frac * nav / stress_loss_per_contract + 1e-9))


# ================================================================== OPT_COVERED
def size_opt_covered(nav: float, underlying_price: float, *, existing_shares: float = 0.0,
                     gross_cap: float = 0.10, deliverable: float = 100.0) -> tuple[float, int]:
    """OPT_COVERED: 100 shares per call; total underlying gross <= 10% NAV.
    Returns (shares to buy, calls to write) so that shares held are a whole
    multiple of the deliverable."""
    if not (nav > 0 and underlying_price > 0):
        return 0.0, 0
    max_shares = gross_cap * nav / underlying_price
    lots = int(math.floor(max_shares / deliverable + 1e-9))
    target = lots * deliverable
    buy = max(target - max(existing_shares, 0.0), 0.0)
    held_lots = int(math.floor((max(existing_shares, 0.0) + buy) / deliverable + 1e-9))
    return buy, min(held_lots, lots)


# ================================================================== OPT_CASHSEC
@dataclass
class CollateralBook:
    """Reserved cash per package id, so collateral is never double counted
    across OPT_CREDIT / OPT_CASHSEC sizing calls."""

    reserved: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.reserved.values())

    def reserve(self, key: str, amount: float) -> None:
        self.reserved[key] = self.reserved.get(key, 0.0) + max(amount, 0.0)

    def release(self, key: str) -> None:
        self.reserved.pop(key, None)

    def available(self, cash: float) -> float:
        return cash - self.total


def size_opt_cashsec(cash: float, strike: float, *, nav: Optional[float] = None,
                     collateral: Optional[CollateralBook] = None, multiplier: float = 100.0,
                     deliverable: float = 100.0, name_cap: float = 0.10,
                     existing_underlying_notional: float = 0.0) -> int:
    """OPT_CASHSEC: each contract reserves strike x deliverable cash; uses only
    cash not already reserved (no double counting); the assignment notional
    (strike x deliverable x n) plus existing exposure to the name must respect
    the equity single-name cap (10% NAV) when ``nav`` is given."""
    if not (strike > 0):
        return 0
    per = strike * deliverable * (multiplier / 100.0)
    avail = collateral.available(cash) if collateral is not None else cash
    n = int(math.floor(max(avail, 0.0) / per + 1e-9))
    if nav is not None and nav > 0:
        cap_n = int(math.floor(max(name_cap * nav - existing_underlying_notional, 0.0) / per + 1e-9))
        n = min(n, cap_n)
    return max(n, 0)


# ================================================================ OPT_ARBITRAGE
@dataclass
class ArbitrageCheck:
    edge: float
    ok: bool
    breakdown: dict


def arbitrage_edge(theoretical_value: float, executable_cost: float, *, fees: float = 0.0,
                   borrow_cost: float = 0.0, funding_cost: float = 0.0, margin_cost: float = 0.0,
                   shorts_located: bool = True, min_edge: float = 0.0) -> ArbitrageCheck:
    """OPT_ARBITRAGE: trade only if the executable (quoted, natural) market
    after fees, borrow, funding and margin carry shows positive modelled edge
    and every short leg has a locate. All amounts in USD for the package:
    edge = theoretical_value - executable_cost - fees - borrow - funding - margin."""
    edge = theoretical_value - executable_cost - fees - borrow_cost - funding_cost - margin_cost
    ok = shorts_located and edge > min_edge
    return ArbitrageCheck(edge, ok, {"theoretical": theoretical_value, "executable_cost": executable_cost,
                                     "fees": fees, "borrow": borrow_cost, "funding": funding_cost,
                                     "margin": margin_cost, "shorts_located": shorts_located})


# ============================================================== ctx convenience
def _prices_for(ctx, symbols: Iterable[str]) -> dict[str, float]:
    out = {}
    for s in symbols:
        try:
            out[s] = float(ctx.mark(s))
        except Exception:
            out[s] = float("nan")
    return out


def equity_gross(ctx) -> float:
    pos = {s: q for s, q in ctx.positions().items() if not ctx.instrument(s).is_option}
    return _gross_equity(pos, _prices_for(ctx, pos))


def size_day_risk_ctx(ctx, symbol: str, entry: float, stop: float, **kw) -> float:
    return size_day_risk(ctx.prior_nav, entry, stop, existing_qty=ctx.position(symbol),
                         now_et=ctx.now_et(), **kw)


def size_day_notional_ctx(ctx, symbol: str, price: float, side: int = 1, **kw) -> float:
    return size_day_notional(ctx.prior_nav, price, side, existing_qty=ctx.position(symbol), **kw)


def size_swing_risk_ctx(ctx, symbol: str, entry: float, stop: float, **kw) -> float:
    q0 = ctx.position(symbol)
    return size_swing_risk(ctx.nav, entry, stop, current_gross=equity_gross(ctx),
                           existing_notional=abs(q0) * entry, **kw)


def size_swing_notional_ctx(ctx, symbol: str, price: float, side: int = 1, **kw) -> float:
    return size_swing_notional(ctx.nav, price, side, current_gross=equity_gross(ctx), **kw)

"""Backtest configuration (all realism knobs with documented defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from ..execution.constraints import BorrowConfig, ShortSaleRules
from ..execution.fees import FeeSchedule, make_fee_schedule
from ..execution.fills import AuctionConfig, BarFillConfig, ComboConfig
from ..execution.impact import SquareRootImpact
from ..execution.spread import SpreadConfig
from ..portfolio.account import AccountType
from ..portfolio.corporate_actions import CorporateActionConfig
from ..portfolio.financing import FinancingConfig
from ..portfolio.hooks import DefaultOptionLifecycle, OptionLifecycle
from ..portfolio.margin import OptionMarginModel


@dataclass
class BacktestConfig:
    # account
    initial_cash: float = 1_000_000.0
    account_type: AccountType = AccountType.REG_T_MARGIN
    fees: Union[str, FeeSchedule] = "IBKR_PRO_TIERED"
    # latency (ARCHITECTURE §4): 0 for bar-driven, 1s for quote-driven
    latency_bars: pd.Timedelta = pd.Timedelta(0)
    latency_quotes: pd.Timedelta = pd.Timedelta(seconds=1)
    # fills
    bar_fill: BarFillConfig = field(default_factory=BarFillConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    combo: ComboConfig = field(default_factory=ComboConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    impact: SquareRootImpact = field(default_factory=SquareRootImpact)
    daily_open_spread_mult: float = 1.0
    use_quotes: str = "auto"                   # 'auto' | 'never' | 'always'
    resolve_intrabar_freq: Optional[str] = "1m"  # finer bars used to resolve ambiguous coarse bars (None: off)
    #: 'auto' quote mode: a bar window fills from quotes only if it holds at least this many quotes per
    #: RTH hour (else bars); symbols without bars always use quotes
    quote_min_per_hour: float = 60.0
    # constraints
    short_sale_rules: ShortSaleRules = field(default_factory=ShortSaleRules)
    borrow: BorrowConfig = field(default_factory=BorrowConfig)
    enforce_locates: bool = True
    halt_events: bool = True
    # portfolio
    financing: FinancingConfig = field(default_factory=FinancingConfig)
    corporate_actions: CorporateActionConfig = field(default_factory=CorporateActionConfig)
    #: None or "cboe" -> atlasbt.options.plugins.CboeStrategyMargin (OPTIONS agent);
    #: "conservative" -> portfolio.margin.ConservativeOptionMargin; or any OptionMarginModel instance
    option_margin: Union[None, str, OptionMarginModel] = None
    #: None or "occ" -> atlasbt.options.plugins.OccOptionLifecycle bound to the portal (rate = portal
    #: risk-free, dividends = portal CASH_DIVIDEND actions known at the session);
    #: "conservative" -> portfolio.hooks.DefaultOptionLifecycle; or any OptionLifecycle instance
    option_lifecycle: Union[None, str, OptionLifecycle] = None
    check_margin_on_submit: bool = True
    margin_calls: bool = True
    rf_tenor: str = "3M"
    # engine
    stats_window: int = 20
    lookback_sessions: int = 60
    debug: bool = False
    record_intraday_equity: bool = False
    max_option_mark_staleness: pd.Timedelta = pd.Timedelta(days=5)
    stale_price_sessions: int = 5          # held symbol without a bar for N sessions -> T2 flag + warning
    #: universes (portal.universe) are filtered to members with verified data; the run is
    #: NOT_TESTABLE if coverage at the start or at any session the universe is queried is
    #: below this (1.0 = every member must have data)
    min_universe_coverage: float = 0.95

    def make_fees(self) -> FeeSchedule:
        if isinstance(self.fees, FeeSchedule):
            return self.fees
        return make_fee_schedule(self.fees)

    def margin_model(self) -> OptionMarginModel:
        m = self.option_margin
        if m is None or m == "cboe":
            from ..options.plugins import CboeStrategyMargin
            return CboeStrategyMargin()
        if m == "conservative":
            from ..portfolio.margin import ConservativeOptionMargin
            return ConservativeOptionMargin()
        if isinstance(m, str):
            raise ValueError(f"unknown option_margin {m!r}")
        return m

    def lifecycle(self, portal=None, now_fn=None) -> OptionLifecycle:
        lc = self.option_lifecycle
        if lc == "conservative":
            return DefaultOptionLifecycle()
        if lc is None or lc in ("occ", "occ_early_assignment", "occ_no_early_assignment"):
            from ..options.plugins import OccOptionLifecycle
            early = lc != "occ_no_early_assignment"
            if portal is None:
                return OccOptionLifecycle(early_assignment=early)
            return OccOptionLifecycle(rate=_portal_rate(portal, self.rf_tenor, now_fn),
                                      dividends=_portal_dividends(portal, now_fn), early_assignment=early)
        if isinstance(lc, str):
            raise ValueError(f"unknown option_lifecycle {lc!r}")
        return lc


def _portal_rate(portal, tenor, now_fn):
    import math

    def rate(session) -> float:
        as_of = now_fn() if now_fn else None
        try:
            r = float(portal.risk_free_rate(as_of, tenor))
        except Exception:
            return 0.0
        return math.log1p(r) if r == r and r > -1 else 0.0       # simple annual -> continuous
    return rate


def _portal_dividends(portal, now_fn, horizon_days: int = 120):
    from ..core.types import ActionType
    from ..portfolio.corporate_actions import as_session_date

    def dividends(underlying: str, session) -> list:
        as_of = now_fn() if now_fn else None
        s = pd.Timestamp(session)
        try:
            acts = portal.corporate_actions([underlying], s, s + pd.Timedelta(days=horizon_days), as_of=as_of)
        except Exception:
            return []
        return sorted((as_session_date(a.ex_date), float(a.amount)) for a in acts
                      if a.type in (ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND)
                      and as_session_date(a.ex_date) > as_session_date(s))
    return dividends

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
    # constraints
    short_sale_rules: ShortSaleRules = field(default_factory=ShortSaleRules)
    borrow: BorrowConfig = field(default_factory=BorrowConfig)
    enforce_locates: bool = True
    halt_events: bool = True
    # portfolio
    financing: FinancingConfig = field(default_factory=FinancingConfig)
    corporate_actions: CorporateActionConfig = field(default_factory=CorporateActionConfig)
    option_margin: Optional[OptionMarginModel] = None      # None -> ConservativeOptionMargin
    option_lifecycle: Optional[OptionLifecycle] = None     # None -> DefaultOptionLifecycle
    check_margin_on_submit: bool = True
    margin_calls: bool = True
    rf_tenor: str = "3M"
    # engine
    stats_window: int = 20
    lookback_sessions: int = 60
    debug: bool = False
    record_intraday_equity: bool = False
    max_option_mark_staleness: pd.Timedelta = pd.Timedelta(days=5)

    def make_fees(self) -> FeeSchedule:
        if isinstance(self.fees, FeeSchedule):
            return self.fees
        return make_fee_schedule(self.fees)

    def lifecycle(self) -> OptionLifecycle:
        return self.option_lifecycle or DefaultOptionLifecycle()

"""Canonical column schemas for BacktestResult frames and meta keys (contract).

The engine MUST emit exactly these columns (extra columns allowed, appended).
Reporting reads them via atlasbt.reporting.schema normalisers.
"""
FILL_COLUMNS = [
    "ts", "order_id", "symbol", "asset_class", "multiplier", "quantity", "price",
    "commission", "exchange_fees", "regulatory_fees", "spread_cost", "impact_cost",
    "fidelity",          # int (core.types.Fidelity value)
    "fidelity_reason",   # str
    "liquidity", "note", "reference_mid",
    "adv",               # 20d ADV in shares known at fill time (NaN for options)
    "underlying_price",  # options: underlying mark at fill; NaN otherwise
    "notional",          # |quantity| * price * multiplier
    "tag",               # order tag ('entry','stop','target','time_exit',...)
]
TRADE_COLUMNS = [
    "trade_id", "symbol", "side", "entry_ts", "exit_ts", "quantity", "entry_price", "exit_price",
    "pnl",               # net of ALL costs incl. fees, borrow, dividends attributable
    "pnl_gross", "costs", "r_multiple", "initial_risk", "mae", "mfe", "bars_held", "multiplier",
    "exit_reason",
]
LEDGER_COLUMNS = ["ts", "amount", "reason", "symbol", "note"]   # amount signed: + inflow
POSITION_COLUMNS = ["ts", "symbol", "quantity", "mark", "market_value"]

# Stable ledger reason codes
LEDGER_REASONS = (
    "TRADE", "COMMISSION", "EXCHANGE_FEE", "REGULATORY_FEE",
    "DIVIDEND", "DIVIDEND_IN_LIEU", "WITHHOLDING_TAX", "SPLIT_CASH_IN_LIEU", "DELIST_CASHOUT",
    "CASH_INTEREST", "MARGIN_INTEREST", "BORROW_FEE", "SHORT_REBATE",
    "OPTION_EXERCISE", "OPTION_ASSIGNMENT", "OPTION_CASH_SETTLEMENT",
    "DEPOSIT", "WITHDRAWAL",
)

# meta keys
META_KEYS = (
    "initial_capital",   # float
    "rf_daily",          # pd.Series: per-session simple risk-free return used for cash accrual
    "risk_free_rate",    # float fallback, annualised simple
    "start", "end", "bar_freq", "universe", "config", "data_sources",
    "survivorship_bias", # bool/str warning
    "n_trials",          # grid size for DSR
)

# fidelity dict shape
# {"tiers": {"T0_QUOTE": {"fills": int, "notional": float}, ...},
#  "reasons": [{"tier": "T2_PROXY", "reason": str, "count": int}, ...],
#  "grade": "A".."E" | "NA"}

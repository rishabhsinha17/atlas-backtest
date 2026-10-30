# Strategy authoring guide (S001–S172)

Read first: docs/ARCHITECTURE.md, docs/ENGINE.md (Context API, brackets, session rules, fill rules), atlasbt/engine/strategy.py,
atlasbt/risk/sizing.py, docs/ANALYTICS.md (indicator conventions), atlasbt/strategies/registry.py.

## 1. One class per strategy, registered

```python
from atlasbt.strategies.registry import register
from atlasbt.engine.strategy import Strategy, Requirement

@register
class S046(Strategy):
    id = "S046"; name = "20-day Donchian breakout"; bar_freq = "1d"; sizing = "SWING_RISK"; asset = "STK/ETF"
    universe = DEFAULT_UNIVERSES["SWING_ETF"]          # see §3
    params = {"entry_n": 20, "exit_n": 10, "stop_atr": 2.0, "short": True}
    grid = {"entry_n": [20, 40, 55], "exit_n": [10, 20], "stop_atr": [1.5, 2, 3]}
    requires = [Requirement("bars", "1d")]
    deviations = []   # every interpretation that is not literally in the sheet text, with justification
    spec = "<exact sheet Entry rule> || <exact sheet Exit / risk rule>"   # copied verbatim from docs/strategies.json
    def on_bar(self, ctx): ...
```

* `params` = the sheet's baseline values; `grid` = exactly the "Parameters to test" column (every listed value, nothing
  invented; if the sheet's grid is ambiguous, declare the interpretation in `deviations`).
* `spec` holds the verbatim sheet text, so the conformance reviewers can diff it against the code.
* Indicators: use `atlasbt.indicators` (causal). Do not re-implement. If something is missing, write it in your module
  as a private causal helper with a causality test, and list it in your report.
* Sizing: use `atlasbt.risk.sizing` for the profile named in the sheet's "Sizing" column. Never hand-roll sizes.
* Orders: `ctx.order`, `ctx.bracket` (entry + stop + target + time exit), `ctx.flatten`. Signals at bar close, fills next bar
  (the engine enforces it). Stops that the sheet says are evaluated "on close" (e.g. "exit close<EMA21") are exit orders submitted
  at the close and filled next open; stops that are price levels ("Stop 2*ATR20 from fill") are resting STOP orders.
* Shorts: just submit; the engine applies locate/borrow and SSR. "Require borrow locate" in the sheet => nothing extra.
* Time rules in ET: use `ctx.now_et()`, `ctx.minutes_since_open()`, `ctx.minutes_to_close()`; early closes matter
  (a "15:55 exit" on a 13:00 close day becomes "5 minutes before close": declare this in deviations once, module-wide).
* Data the strategy cannot get: declare a critical `Requirement`. The engine returns NOT_TESTABLE. Never fabricate.
  Proxies (e.g. daily close for a 15:50 quote) are allowed only as an explicit param `proxy=True` with
  `ctx.flag_fidelity(Fidelity.T2_PROXY, reason)` and must never be the default when the real data exists.

## 2. Interpretation rules for ambiguous sheet text (apply consistently)

1. "enter next open" / "buy next open" => market order at next session's open (MOO-like bar fill).
2. "enter at first 1m close > X" => signal at that bar's close, market order fills next bar.
3. "exit 15:55" => market order submitted at the 15:54 bar close (fills 15:55 bar open); engine flat_by enforces too.
4. "2R target" => limit at entry_fill ± 2 × (entry_fill − initial_stop); R is computed from the actual entry fill.
5. "One entry per day" => at most one entry order per symbol per session, including entries that failed to fill.
6. Mirror short variants ("short mirror", "mirror long") are part of the baseline if the sheet's entry says so;
   otherwise they are a `short=True/False` param with the sheet's default.
7. Thresholds stated in % are of price; "bp" = 1e-4.
8. Historical medians for RVOL/time-of-day: prior 20 sessions unless the sheet says otherwise (declare).
9. "SMA200 positive/rising/up" = SMA200 today > SMA200 20 sessions ago unless defined otherwise (declare).
10. Month-end = last NYSE session of the month (calendar.is_last_session_of_month).

## 3. Default universes (instrument choice is part of the test design: declare it)

```
ETF_CORE      = SPY QQQ IWM DIA
SECTOR_SPDR   = XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC  (respect inception dates; XLRE 2015-10, XLC 2018-06)
SWING_ETF     = ETF_CORE + SECTOR_SPDR + TLT GLD EFA EEM
MEGACAP_STK   = point-in-time top-N S&P 500 members by 20d dollar volume (portal universe "SP500_TOP50_DV")
SP500_PIT     = portal universe "SP500" (point-in-time; survivorship caveat before 2011 recorded in result meta)
TBILL_PROXY   = BIL (from 2007-05) else SHV (2007-01) else cash at DTB3; the engine pays cash interest at rf
```
* "STK/ETF" intraday => ETF_CORE + MEGACAP_STK; "STK" swing => SP500_PIT; "ETF" swing => SWING_ETF.
* Strategies whose sheet text names instruments (SPY, RSP, VIX) use them exactly.
* Allocation strategies with "frozen universe" use a fixed list defined in the module with listing dates; never pick
  ETFs because they did well.

## 4. Tests you must ship (tests/strategies/<module>/)

* Signal tests on small synthetic frames (mechanics only, named `synthetic_*`) that hand-construct the exact pattern
  in the sheet and assert entry/exit timing and prices, plus a "near miss" that must NOT trigger.
* Lookahead test per strategy: run on data truncated at t vs full data and assert identical orders up to t.
* A smoke run of every strategy through `run_backtest` on the engine's test portal.

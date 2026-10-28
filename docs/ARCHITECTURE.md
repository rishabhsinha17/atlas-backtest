# atlasbt — architecture & contracts

Goal: backtest the 172 strategies in `docs/strategies.json` (source: `docs/Trading_Strategy_Atlas_172_Systems.xlsx`)
as close to real-life execution as free data allows. **Honesty beats coverage**: when the data needed for a
faithful test does not exist, the engine says so (Fidelity tier / NOT_TESTABLE). It never fills gaps with invented data.

Python 3.13, pandas 3.x (copy-on-write; no chained assignment), numpy 2.x, scipy, statsmodels, exchange_calendars, numba optional.
Package root: `/home/user/atlas_bt/atlasbt`. Tests: `/home/user/atlas_bt/tests` (pytest). Run `python -m pytest -q` from repo root.

## 1. Ownership (who may edit what)

| Package | Owner | Contents |
|---|---|---|
| `atlasbt/core` | integrator only | `types.py`, `protocols.py`, `clock.py` (calendar + tz helpers). Propose changes in your report; do not edit. |
| `atlasbt/engine` | ENGINE agent | event loop, Context implementation, PIT guard, order manager, results assembly, run registry/CLI |
| `atlasbt/execution` | ENGINE agent | fill models, spread estimators, impact, fees (historical schedules), market constraints (SSR, halts, locates, ticks, hours) |
| `atlasbt/portfolio` | ENGINE agent | accounting ledger, corporate actions application, financing/borrow accrual, Reg-T margin, liquidation |
| `atlasbt/risk` | ENGINE agent | the 13 sizing profiles from the sheet, exposure caps |
| `atlasbt/data` | DATA agent | DataPortal implementation, parquet store, providers (network adapters + local loaders), adjustments, universes, data QA |
| `atlasbt/pricing`, `atlasbt/options` | OPTIONS agent | BS/Black76/American pricing, IV, greeks, surfaces, synthetic chains, chain selection by delta/DTE, combo construction, expiration/exercise/assignment logic, option margin (Cboe/Reg-T rules) |
| `atlasbt/indicators`, `atlasbt/validation`, `atlasbt/reporting` | ANALYTICS agent | causal indicators, walk-forward/bootstrap/DSR/PBO/SPA, cost-stress, metrics, report generation |
| `atlasbt/strategies` | STRATEGY agents (later phase) | one module per family; registry |
| `tests/` | everyone, own subfolder: `tests/<package>/` | |

Cross-package calls go through `core.protocols` or a documented public function. If you need something from another
owner that does not exist yet, write against the documented interface below and note it in your report.

## 2. Time & calendar (`atlasbt.core.clock`)

* Internal timestamps: tz-aware UTC `pd.Timestamp`. Session dates (incl. `CorporateAction.ex_date`/`pay_date`): naive `pd.Timestamp` at midnight representing the ET date.
* `clock.nyse()` returns the singleton that implements `TradingCalendar` (exchange_calendars XNYS: holidays, early closes 13:00 ET, historical closures like 2001-09-11, 2012-10-29/30, national days of mourning).
* Helpers: `to_et(ts)`, `et(date, "HH:MM") -> UTC ts`, `session_open_close(date)`.

## 3. Canonical frames (what the DataPortal returns)

* **bars**: index `start` (UTC, tz-aware, named `start`), columns `open, high, low, close, volume` (float64), optional `vwap, trades`, `session` ('PRE','RTH','POST' for intraday; 'DAY' for daily), `known_at` (UTC). Daily bars: `start` = session open UTC, `known_at` = session close UTC. Monthly frequency is NOT stored; strategies resample daily bars causally.
* **quotes**: index `ts` (UTC), columns `bid, ask, bid_size, ask_size`.
* **option_chain**: columns `ts, underlying, symbol, expiry (naive date), strike, right ('C'/'P'), bid, ask, mid, iv, delta, gamma, vega, theta, underlying_price, volume, open_interest, exercise, settlement, multiplier, fidelity`; optional `root` (e.g. SPX vs SPXW), `settlement_date`, `kind`. Expiry = LAST TRADING DAY (vendor SPX AM monthlies listed on Friday must be converted to the Thursday).
* **corporate actions / events**: `core.types.CorporateAction`, `core.types.MarketEvent`.
  Event payloads: `EARNINGS{timing:'BMO'|'AMC'|'DURING'|'UNKNOWN', eps_actual, eps_estimate, estimate_known_at, fiscal_period}`,
  `INDEX_ADD/INDEX_DEL{index, effective_date}`, `FOMC{type:'STATEMENT'}`, `BUYBACK{amount_usd, pct_mcap, with_earnings:bool}`, `HALT{reason, resume_at}`.
* **fundamentals**: long frame `symbol, field, value, period_end, known_at (filing acceptance time), source`.
* Every frame carries `attrs['fidelity']` (or a `fidelity` column) and `attrs['source']`.

## 4. Engine loop (ENGINE agent implements; everyone must understand)

The engine advances over a **master timeline of bar END times** at the strategy's base frequency (`'1m'`, `'5m'`, `'1d'`).
At each step `t` (the end of bar `[t-Δ, t)`):

1. **Session start** (first step of a session): apply corporate actions with `ex_date == session` (splits adjust qty/cost,
   cash dividends booked as receivable on ex-date and paid on pay_date; shorts owe the dividend), delistings, symbol changes;
   expire DAY orders from prior session; reset SSR state; handle option events that occurred overnight.
2. **Fills**: every WORKING order with `eligible_at < t` is offered the market data in `(max(eligible_at, prev_t), t]` via
   its FillModel. Orders submitted at step `t-Δ` can therefore only fill against bar `[t-Δ, t)` or later — i.e. the
   **next** bar's open at the earliest. Nothing ever fills at a price from before it was submitted.
3. **Mark-to-market** at `t` using the bar close (or mid if quotes).
4. **Strategy callback** `on_bar(ctx)` with `ctx.now = t`. Orders submitted now get `eligible_at = t + latency`
   (default latency: 1s for quote-driven, 0 for bar-driven since the next bar starts at `t`).
5. **Session end** (after last bar): accrue financing (cash interest at T-bill minus broker spread, debit interest,
   short-stock borrow fees on market value, actual/360 for overnight calendar days), option expiration/settlement/
   assignment, Reg-T margin check (margin calls -> forced liquidation orders for next open), record NAV, snapshots.

Daily strategies: base bar is the whole session; a market order placed after the close fills at the **next session open**
(opening auction price + modeled costs). If the strategy needs an intraday decision time on a daily strategy
(e.g. "buy at 15:50"), it must run on an intraday base frequency when intraday data exists; otherwise it may opt into
a declared proxy (`close` as 15:50 price) which is tagged `T2_PROXY` and reported.

Intrabar ambiguity (bar data only): within a bar, price path is assumed **open -> adverse extreme -> favorable extreme -> close**
for the order being evaluated (i.e. stop is hit before target when both are inside one bar). If finer data exists
(e.g. 1m inside a 1d bar) the engine resolves the path with the finer data and tags the fill accordingly.

## 5. Execution realism (ENGINE agent, `atlasbt.execution`)

* **Prices**: buy at ask / sell at bid when quotes exist (T0). With bars only: reference = next bar open, cost =
  half-spread (estimated: Corwin-Schultz / Abdi-Ranaldo from daily OHLC, with floors by price level & tick size,
  time-of-day multiplier for the open/close) + square-root impact `k * sigma_daily * sqrt(Q/ADV)` (T1, with the
  spread flagged as estimated). Opening/closing auction fills (MOO/MOC) at official open/close +/- impact.
* **Stops** (stop-market): trigger when traded price crosses; fill at worse of stop and the first price after trigger
  (gap through -> fill at the gap open). **Limits**: fill only if price trades *through* the limit by >= 1 tick
  (touch is not enough in bar mode; queue position unknown), at the limit price.
* **Participation cap**: max % of bar volume per fill (default 10%), remainder stays working (partial fills).
* **Fees**: broker commission schedules (IBKR Pro fixed & tiered, zero-commission retail), SEC Section 31 fee
  (**historical rate table by effective date**), FINRA TAF (historical), options: per-contract commission, ORF, OCC
  clearing, exchange fee; index options (SPX) proprietary fee. Config-selectable; default = IBKR Pro tiered.
* **Constraints**: Reg SHO Rule 201 SSR (trigger: -10% vs prior close, active rest of day + next day, effective
  2011-02-28/complied 2011-05-10): short sales cannot execute at/below national best bid -> in bar mode short entries are
  filled at ask-side estimate or rejected per config. Locate/borrow: from borrow data if present else configurable
  default (ETFs/large caps easy-to-borrow at 0.25%-0.5% p.a., flagged T2). Halts: zero-volume RTH bars or HALT events
  -> no fills. Tick size: $0.01 (>= $1), $0.0001 (< $1). Odd lots allowed for stocks; options integer contracts.
  Premarket/after-hours fills only if the order explicitly allows extended hours and PRE/POST data exist.

## 6. Portfolio & accounting (ENGINE agent, `atlasbt.portfolio`)

Double-entry style cash ledger (every movement has a reason code). Invariants checked every step in debug mode:
`NAV == cash + sum(qty*multiplier*mark)`; ledger sums reconcile. Supports: long/short stock, options (premium, exercise,
assignment, cash settlement), dividends (ex-date entitlement, pay-date cash, withholding option, short pays in lieu),
splits (incl. option deliverable adjustment), delisting cash-out, cash interest, margin interest, borrow fees,
short rebate (default 0 for retail), Reg-T margin (50% initial, 25% long / 30% short maintenance) and option margin
from `atlasbt.options.margin`. Account types: CASH, REG_T_MARGIN.

## 7. Options (OPTIONS agent)

* Pricing: Black-Scholes-Merton with continuous dividend yield and with discrete dividends; Black-76; American via
  CRR binomial / Bjerksund-Stensland 2002; robust IV solver (Brent, handles below-intrinsic quotes); greeks.
* Chains: real chain quotes when available (T0 for EOD-timestamped snapshot at its true time; T2 if the snapshot time
  differs from the decision time). Otherwise a **synthetic chain** (T3_MODELED): listed-expiry & strike-grid generator
  honoring historical listing conventions (monthly 3rd Friday, weeklies era, SPXW MWF/daily 0DTE introduction dates,
  strike increments), IV from vol-index term structure (VIX9D/VIX/VIX3M/VIX6M, VIX1D when available) plus a skew model
  calibrated to SKEW or fitted smile parameters, and a bid/ask model calibrated from whatever real quotes exist.
* Selection: by target delta and DTE window with deterministic tie-breaks; same-expiry multi-leg packages;
  combos fill per leg at natural (buy ask / sell bid) by default, with optional price improvement fraction.
* Lifecycle: expiration on last trading day; automatic exercise if ITM by >= $0.01 (OCC ex-by-ex); cash settlement AM
  (SOQ ~ opening print of expiry day) vs PM (official close); early assignment model for short American calls
  (ex-dividend when extrinsic < dividend) and deep ITM puts (extrinsic < carry); pin risk handling.
* Margin: Cboe/Reg-T strategy-based rules (naked short: 100% premium + 20% underlying - OTM amount, min 10%; broad index 15%/10%;
  spreads: max loss; covered/cash-secured; short straddles: greater side + other premium).

## 8. Fidelity reporting

Each fill and each data access has a `Fidelity`. The result's `fidelity` scorecard lists, per tier, number of fills and
notional, and the list of distinct reasons (e.g. "spread estimated (Corwin-Schultz)", "15:50 fill proxied by close",
"borrow fee default 0.30%", "options synthetic from VIX surface"). A strategy whose critical input is NA returns
`RunStatus.NOT_TESTABLE` with a reason instead of a fake result.

## 9. Strategy authoring (later phase)

```python
class S001(Strategy):
    id = "S001"; bar_freq = "1m"; sizing = "DAY_RISK"; asset = "STK/ETF"
    params = {"or_minutes": 5, "rvol": 1.5, "target_r": None, "stop": "or_low"}
    grid = {"or_minutes": [5, 15, 30], "rvol": [1, 1.5, 2], "target_r": [None, 1.5, 2], "stop": ["or_low", "atr5m"]}
    requires = [Requirement("bars", "1m", sessions=("RTH",))]
    def on_bar(self, ctx): ...
```
Rules: signals only from completed bars; orders only via ctx; sizing via `atlasbt.risk.sizing`; every assumption that
deviates from the sheet text is declared in `deviations` with justification.

## 10. Testing requirements (all agents)

Each package ships unit tests + at least one **known-answer test** against an independent reference (textbook value,
published number, or hand computation in the test). Lookahead tests: perturb data after `t` and assert decisions at
`<= t` are unchanged. No network access in tests: fixtures must be small, synthetic **for testing mechanics only**
(clearly named `synthetic_*`), or real files under `data_raw/`.

## 11. Contract changes

Only the integrator edits `atlasbt/core`. Put requested changes in your final report under "CONTRACT CHANGE REQUESTS".

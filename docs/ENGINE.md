# atlasbt engine: event loop, realism assumptions, configuration

Owner: ENGINE agent (`atlasbt/engine`, `atlasbt/execution`, `atlasbt/portfolio`, `atlasbt/risk`).
This file documents how a backtest runs, every realism assumption with its default and source, and
every configuration knob. Contracts it builds on: `docs/ARCHITECTURE.md`, `atlasbt/core/*`.

```python
from atlasbt.engine import Backtester, BacktestConfig, Strategy, Requirement
result = Backtester(MyStrategy(**params), portal, BacktestConfig()).run("2010-01-04", "2024-12-31")
```

## 1. The loop (`engine/backtester.py`)

Master timeline = bar END times at the strategy base frequency (`'1m'`, `'5m'`, ... or `'1d'`; `'1mo'`,
`'1w'` and `'event'` strategies run on the daily timeline and resample causally). Steps are built from the
NYSE calendar (`core.clock`), including early closes, not from the data, so missing bars do not shift time.
`Strategy.sessions = ("PRE","RTH","POST")` adds extended-hours steps (04:00-09:30, 16:00-20:00 ET).

For each session:

1. **Session start** (`ctx.now` = start of the first bar): dividends whose pay date has come are posted; corporate
   actions with `ex_date == session` are applied (splits, dividends, delistings, symbol changes, spin-offs; open
   orders on the symbol are cancelled and logged); AM-settled options are settled (option lifecycle hook);
   `strategy.on_session_start(ctx)`.
2. **Each step t** (end of bar `[t-Δ, t)`):
   1. *Fills*: every working order is offered the bars/quotes in `(prev_t, t]` whose bar **start** is at or after
      the order's `eligible_at` (`submitted_at + latency`). An order submitted at the close of bar k can therefore
      only fill from bar k+1's open onward; nothing ever fills at a price printed before the order existed.
   2. *Mark to market* at the last completed bar close (stocks), quote mid when only quotes exist.
   3. Queued `on_fill` callbacks, engine session rules (`flat_by`), then `strategy.on_bar(ctx)` with `ctx.now = t`.
   4. MAE/MFE and bars-held tracking for open trades; debug invariants.
3. **Session end** (`ctx.now` = last step): `strategy.on_session_end`; PM/physical option expiry
   (exercise/assignment/cash settlement); option marks from quote snapshots; financing (cash interest, margin
   interest, borrow fees) for the calendar days to the next session; DAY-order expiry; Reg-T margin check
   (margin call -> liquidation market orders that fill at the next open, logged in
   `diagnostics["margin_calls"]`); NAV and position snapshot.

Orders submitted at a session close (daily strategies) are DAY orders for the **next** session and fill at its
open (or later per their type).

### Point-in-time guarantees
* `ctx.portal()` is a `PITPortal`: every query is pinned to `as_of = ctx.now`; an explicit later `as_of` raises
  `LookaheadError`; any row the portal returns with a knowledge time after `now` raises (data-layer contract breach,
  never silently dropped); bar `end` bounds are clipped to `now`, so the in-progress bar is never returned.
* `ctx.history(symbol, n, freq)` returns completed bars only (daily bars of the current session are invisible
  until the close). Raw same-frequency history is served from the engine cache (fast path), everything else
  through the PIT portal.
* The engine itself pre-loads raw bars per symbol once (`MarketData`) for speed; every access goes through a time
  guard (`MarketData.check`) that raises `LookaheadError` if engine code asks for a bar ending after `now`.
  Fill timestamps after `now` also raise. `diagnostics["lookahead_guard_violations"]` must be 0.
* Tests (`tests/engine/test_eng_backtester.py`) perturb all bars after a cut date and assert identical
  decisions, fills and equity before it.

### Intrabar path
Within one bar the price path is `open -> adverse extreme -> favourable extreme -> close` for the position being
managed (long: low first; short: high first; flat: adverse for the side of the pending orders, nearest extreme if
mixed). Each potential fill gets a position on this path; orders on a symbol are processed in path order, OCO
reductions and bracket-child activations happen in that order, and children of an entry that filled inside the
bar are evaluated on the remainder of the bar (path restarted from the entry price, adverse first). Fills that
depended on this assumption are tagged T2 ("intrabar order assumed adverse-first"). If finer bars exist
(`resolve_intrabar_freq`, default `'1m'`) and a coarse bar is ambiguous (two or more orders could trigger, or a
bracket parent may fill), the bar is replayed on the finer bars and the fill is tagged "intrabar path resolved with
1m bars" (stays T1). Intrabar fill timestamps are interpolated from the path position (floored to the second).

### Orders (`engine/orders.py`, `ctx.order / ctx.submit / ctx.bracket`)
* Types: MARKET, LIMIT, STOP (gap-aware stop market), STOP_LIMIT, MOO, MOC, COMBO. TIF: DAY, GTC, IOC.
* Latency: 0 for bar-driven fills (the next bar starts at `t`), 1 s for quote-driven and option/combo fills.
* DAY validity = the session containing `eligible_at` if before its close (+4 h with `outside_rth`), else the next
  session. MOO joins the open of the first session whose 09:28 cutoff is after `eligible_at`; MOC the first close
  whose 15:50 cutoff is after it. Unfilled auction remainder is cancelled.
* OCO groups use reduce-on-fill semantics (partial fills reduce the siblings). Brackets: entry + protective stop +
  target + time exit; children are created in proportion to the entry's fills, share an OCO group and default to
  GTC. `initial_stop` (or the bracket stop) is recorded for R-multiples.
* Submit-time validation: quantity (whole shares unless `allow_fractional`; integer contracts), required prices,
  tick rounding (buy limits down / sell limits up; stops away from the market), session rules, short-sale
  detection + locate, cash-account restrictions, Reg-T initial margin / cash sufficiency including pending orders.
  Rejections are listed in `diagnostics["rejected_orders"]`; all orders in `diagnostics["orders"]`.
* Session rules from sizing profiles: DAY_RISK => no exposure-increasing orders after 15:15 ET, all positions
  flattened at 15:55 ET (market orders, tag `flat_by`); DAY_NOTIONAL => flat at 15:55 ET. Strategies may override
  `no_new_entries_after` / `flat_by`.

## 2. Execution realism (`atlasbt/execution`) — assumptions, defaults, sources

| Topic | Default | Source / justification |
|---|---|---|
| Market order, bars only (T1) | next bar open x (1 ± half-spread ± impact) | ARCHITECTURE §5; trade price assumed to be mid |
| Stop market | triggers when traded price crosses; fill at max(stop, first price after trigger); gap -> open | ARCHITECTURE §5 |
| Limit (resting) | fills only if price trades THROUGH by >= 1 tick, at the limit price, no spread cost (MAKER) | queue position unknown; touch is not enough |
| Marketable limit | fills like a market order capped at the limit (TAKER) | |
| Stop-limit | stop trigger, then marketable-if-possible else resting limit | |
| Participation cap | 10% of bar volume per fill; remainder keeps working | ARCHITECTURE §5 |
| Halts | zero/NaN-volume RTH/DAY bars and `HALT` events (`effective_at`..`resume_at`) => no fills | |
| Quotes (T0) | buy at ask, sell at bid, first quote strictly after `eligible_at`; displayed size caps each fill; resting limits never fill on quotes alone; locked/crossed quotes skipped | |
| MOO / MOC | official open/close print ± impact, no spread; cap 10% of estimated auction volume (2% / 8% of daily volume in daily-bar mode, open/close bar volume intraday) | NYSE cutoffs 09:28 / 15:50 ET |
| Options & combos | one quote snapshot strictly after `eligible_at`; all legs same snapshot timestamp (tolerance 0) or REJECT; natural prices (buy ask / sell bid) with optional `price_improvement` toward mid; stock legs priced from the bar close ± estimated half-spread (T2) | ARCHITECTURE §7 |
| Spread estimator | Corwin-Schultz (2012) 21-day mean of two-day estimates, negatives set to 0, overnight adjustment; fallback Abdi-Ranaldo (2017) monthly, then Roll (1984); all PIT (session i uses <= i-1), split-consistent | papers; known-answer tests |
| Spread floors | full spread >= 1 tick (Rule 612: $0.01 >= $1, $0.0001 below) and price buckets <$1: 100bp, <$5: 30bp, <$10: 10bp, <$20: 4bp, else 1bp; default 20bp if no estimate (T2) | engineering defaults (conservative), configurable |
| Time-of-day | x2.5 first 5 min, x1.75 to 15 min, x1.3 to 30 min, x1.3 last 5 min, x1.1 last 15 min, x3 extended hours; daily-bar open fills use the daily average (x1.0, `daily_open_spread_mult`) | U-shaped intraday spreads: McInish & Wood (1992), Chung, Van Ness & Van Ness (1999) |
| Impact | `k * sigma_daily * sqrt(Q$/ADV$)`, k = 0.1, cap 5%/fill; sigma and ADV = 20 prior sessions | Almgren et al. (2005) cost ≈ 0.1·σ·sqrt(Q/V) for Q/V ≤ 10%; Tóth et al. (2011) Y≈0.5-1 is peak impact (stress k=0.5); Frazzini-Israel-Moskowitz (2018) |
| Reg SHO | before 2007-07-06: tick test => every short sale restricted; 2007-07-06..2011-02-27: none; from 2011-02-28: Rule 201 (trigger: low <= 90% of prior close; rest of day + next day). Bar mode `ssr_mode='ask_limit'`: short sale posted at the estimated ask and fills only if price trades through it; `'reject'`; `'ignore'` | SEC Rel. 34-55970 (uptick repeal), Rule 201 compliance date 2011-02-28 |
| Locates | `portal.borrow(symbol)`; if no data: easy-to-borrow at 0.30% p.a. (T2), or reject with `BorrowConfig.require_borrow_data` | ARCHITECTURE §5 |
| Ticks | stocks Rule 612; options `Instrument.tick_size` | |
| Extended hours | PRE/POST bars fill only orders with `outside_rth=True` | |
| Fill fidelity | T0 quotes; T1 bars with estimated spread; T2 when the spread is a default, intrabar ordering was assumed, SSR/proxy inputs, default borrow; T3 passes through from modelled option quotes | |

### Fees (`execution/fees.py`, tables in `execution/rates.py`, retrieved 2026-10-28)
* **IBKR Pro Tiered (default)**: 0.0035/0.0020/0.0015/0.0010/0.0005 per share by month-to-date volume (marginal),
  min $0.35, max 1% of trade value per order (minimum applied once per order across partial fills); plus taker
  access fee $0.003/sh (Reg NMS cap; $0.001 from 2025-11-03, approx.), maker 0 (rebates ignored), auction
  $0.001/sh (approx.), NSCC/DTC clearing $0.0002/sh, NYSE pass-through 0.000175×commission, FINRA pass-through
  0.00056×commission, CAT fee. Source: interactivebrokers.com commissions-stocks page.
* **IBKR Pro Fixed**: $0.005/sh, min $1.00, max 1% of trade value; exchange/clearing included; SEC/TAF passed through.
  Published examples (100@$25 = $1.00, 1,000@$25 = $5.00, 1,000@$0.25 = $2.50, 10@$0.20 = $0.02) are tests.
* **Options (IBKR Pro)**: $0.65/0.50/0.25 by premium (≥0.10 / ≥0.05 / <0.05) up to 10k contracts/month, then
  0.50/0.25, 0.25, 0.15; min $1.00 per order (per leg); + OCC clearing, aggregate ORF, exchange fee
  (equity options $0.10/contract blended approx.; SPX/SPXW Cboe proprietary schedule), SEC + TAF on sales, CAT.
  IBKR published examples are tests. Exercise/assignment: no commission; SEC fee on resulting stock sales.
* **Zero-commission retail**: $0 stock/ETF commission from 2019-10-07 (industry move led by Schwab); before that
  (with `historical=True`) $4.95/order + $0.65/contract; SEC/TAF/OCC/ORF passed through; no PFOF price
  improvement credited.
* **SEC Section 31** (USD per $1M of sales; rounded up to the cent per fill): 1/300 of 1% until 2001-12-27; $15.00
  (2001-12-28, Fee Relief Act); 30.10 (2002-04-01); 25.20 (2003-03-22); 46.80 (2003-04-01); 39.00 (2004-02-22);
  23.40 (2004-04-01); 32.90 (2005-01-07); 41.80 (2005-04-01); 30.70 (2005-12-22); 15.30 (2007-03-17); 11.00
  (2008-01-25); 5.60 (2008-04-01); 25.70 (2009-04-10); 12.70 (2010-01-15); 16.90 (2010-04-01); 19.20 (2011-01-21);
  18.00 (2012-02-21); 22.40 (2012-04-01); 17.40 (2013-05-25); 22.10 (2014-03-18); 18.40 (2015-02-14); 21.80
  (2016-02-16); 23.10 (2017-07-04); 13.00 (2018-05-22); 20.70 (2019-04-16); 22.10 (2020-02-18); 5.10 (2021-02-25);
  22.90 (2022-05-14); 8.00 (2023-02-27); 27.80 (2024-05-22); 0.00 (2025-05-14); 20.60 (2026-04-04).
  Source: sec.gov Fee Rate Advisories and each linked press release.
* **FINRA TAF** (sales; per share / max per trade / per option contract): 0.0001/$10 (2003-01-01, approx.),
  0.00005/$5 (2003-05-30), 0.0001/$10 (2003-09-01), 0.000075/$3.75/0.002 (2004-11-01, NtM 04-84),
  0.000090/$4.50 (2011-07-01), 0.000095/$4.75 (2012-03-01), 0.000119/$5.95/0.002 (2012-07-01),
  0.000130/$6.49/0.00218 (2022), 0.000145/$7.27/0.00244 (2023), 0.000166/$8.30/0.00279 (2024),
  0.000195/$9.79/0.00329 (2026-01-01), 0 (fee holiday 2026-10-01..12-31), back to 2026 rates 2027-01-01.
  Source: FINRA By-Laws Schedule A §1 and every dated revision (finra.org/rev/<id>).
* **OCC clearing** per contract: $0.05 (pre-2019, approx.), $0.055 (2019-04-01), $0.045 (2020-09-01), $0.02
  (2021-06-01), $0.025 (2025-01-01). Source: SR-OCC filings.
* **ORF** (aggregate of member-exchange ORFs passed through): APPROXIMATE order-of-magnitude table — 0 before
  2008-12, $0.0045, $0.01 (2012), $0.02 (2016), $0.01 (2025-08-01). Per-exchange rates change monthly; replace with
  a data-driven table if precision matters.
* **Cboe SPX/SPXW** customer: $0.36 (premium < $1) / $0.45 (≥ $1) + execution surcharge $0.21 (SPX) / $0.14 (SPXW);
  Cboe Fees Schedule 2026-10-01, applied to earlier dates as an approximation.

## 3. Portfolio & accounting (`atlasbt/portfolio`)
* Reason-coded cash ledger (`core.schemas.LEDGER_REASONS`): TRADE, COMMISSION, EXCHANGE_FEE, REGULATORY_FEE,
  DIVIDEND, DIVIDEND_IN_LIEU, WITHHOLDING_TAX, SPLIT_CASH_IN_LIEU, DELIST_CASHOUT, CASH_INTEREST, MARGIN_INTEREST,
  BORROW_FEE, SHORT_REBATE, OPTION_EXERCISE, OPTION_ASSIGNMENT, OPTION_CASH_SETTLEMENT, DEPOSIT, WITHDRAWAL.
  Invariants (debug mode, every step): `cash == sum(ledger)`; `NAV == cash + Σ qty·multiplier·mark + accrued
  dividends`; every position marked; no order over-filled; no fill after `now`.
* Average-cost realized PnL; flips close then reopen. Multipliers honoured everywhere.
* Splits: qty×ratio, prices÷ratio, fractional shares cashed out at the post-split mark (unless fractional shares
  allowed). Cash dividends: entitlement at ex-date on the prior-close position (receivable, NAV-neutral), cash on
  `pay_date`, optional withholding (`CorporateActionConfig.withholding_rate`, default 0); shorts pay in lieu.
  Delisting: cash-out at `amount`; missing value => last mark (T2 flag). Symbol change: position moved.
  Spin-off: `ratio` new shares, `amount` = cost-basis fraction. Action dates accept naive dates, UTC-midnight
  date labels, or ET times.
* Financing (posted daily, actual/360 over calendar days to the next session, Friday→Monday = 3 days): cash
  balance net of short-sale proceeds earns `max(rf − 0.50%, 0)`; a debit balance pays `rf + 1.50%`; short borrow
  fee on short market value at the `portal.borrow` rate; short rebate 0 (retail). Defaults reflect IBKR Pro's
  benchmark ± spread schedule for small balances. `rf` = `portal.risk_free_rate(as_of, '3M')`; recorded per session
  in `meta["rf_daily"]`.
* Reg-T margin (`margin.py`): initial 50% long / 50% short (150% total); maintenance 25% long, short max($5/sh, 30%)
  at ≥ $5 and max($2.50/sh, 100%) below $5 (FINRA 4210). Pre-trade check includes pending exposure-increasing
  orders. Margin call when equity < maintenance at a session end => proportional reduction of every
  requirement-carrying position to maintenance ≤ equity/1.10, filled at the next open. CASH accounts: no shorts,
  purchases need cash, short puts must be cash-secured, naked calls rejected.
* **Plug-in interfaces for the OPTIONS agent**
  * `portfolio.margin.OptionMarginModel.requirement(account, underlying_price) -> float` (total option margin,
    never 0 for naked shorts). Default `ConservativeOptionMargin`: long premium paid in full; covered calls/puts;
    verticals/calendars at max loss; naked max(premium + 20% (15% broad index) of underlying − OTM, premium + 10%
    of underlying (calls) / strike (puts)); cash accounts: put strike×multiplier secured. Plug in via
    `BacktestConfig(option_margin=...)`.
  * `portfolio.hooks.OptionLifecycle` with `at_session_start(session, prev_session, option_positions, prices)` and
    `at_session_end(session, option_positions, prices)` returning `OptionEvent`s (EXPIRE_WORTHLESS, EXERCISE,
    ASSIGNMENT, CASH_SETTLE with option close price and optional stock delivery). The engine books them.
    Default `DefaultOptionLifecycle`: PHYSICAL/CASH_PM expire at the expiry session close (official close);
    CASH_AM settle at the next session's open (SOQ proxy, T2); automatic exercise if ITM by ≥ $0.01; no early
    exercise/assignment. Plug in via `BacktestConfig(option_lifecycle=...)`.
* Options are marked at session ends from `portal.option_quote` mids (intraday option NAV is stale).

## 4. Results (`core.types.BacktestResult`, columns per `core/schemas.py`)
* `equity` NAV at each session close (UTC); `returns` daily simple (first vs initial cash).
* `fills`: ts, order_id, symbol, asset_class, multiplier, quantity, price, commission, exchange_fees,
  regulatory_fees, spread_cost, impact_cost (both informational, already in price), fidelity (int),
  fidelity_reason, liquidity, note, reference_mid, adv (20d shares known at fill), underlying_price (options),
  notional, tag.
* `trades`: position-level round trips — trade_id, symbol, side, entry/exit ts and prices, quantity, pnl (net of
  fees + attributed borrow fees, dividends, in-lieu payments and withholding), pnl_gross, costs, r_multiple and
  initial_risk (from the initial stop), mae/mfe (USD, on max size), bars_held, multiplier, exit_reason (exit
  order tag: stop/target/time_exit/flat_by/margin_liquidation/delist/assignment/...; `open` if still open).
* `positions` (end-of-day snapshots), `ledger`, `fidelity` = `{"tiers": {...}, "reasons": [{"tier", "reason",
  "count"}], "grade"}` (grade = worst tier carrying ≥ 5% of notional, any strategy/engine T2+ flag, or required data
  input: T0→A, T1→B, T2→C, T3→D, NA→E), `diagnostics` (rejected orders, all orders, margin calls, corporate actions,
  option events, financing totals, simulation counters, invariant checks, guard violations, runtime, logs),
  `meta` (initial_capital, rf_daily, risk_free_rate, start, end, bar_freq, universe, config, data_sources,
  survivorship_bias, n_trials = grid size, plus engine details).
* `RunStatus.NOT_TESTABLE` when a critical `Requirement` is unavailable (`portal.has` False or fidelity NA/worse
  than `min_fidelity`); `FAILED` on strategy exceptions (traceback in diagnostics; `raise_errors=True` re-raises).
  `LookaheadError` always propagates.

## 5. Sizing profiles (`atlasbt/risk/sizing.py`)
Verbatim sheet text in `PROFILE_TEXT`. Functions return signed whole shares / integer contracts:
`size_day_risk` (0.25% prior-day NAV / |entry−stop|, single-name gross ≤ 10% prior-day NAV, 0 after 15:15 ET),
`size_day_notional` (10% prior-day NAV, one position per instrument), `size_swing_risk` (0.50% current NAV risk,
notional ≤ 10%, aggregate equity gross ≤ 100%), `size_swing_notional` (long 10%, short 5%, gross ≤ 100%),
`cross_section_weights` (+50%/−50% equal weight, name ≤ 2%, sector ±10% per side by default or net; capped weight
stays in cash), `portfolio_weights` / `rebalance_deltas` (Σ|w| ≤ 100%, residual cash earns the financing rate;
orders at a close fill next session), `size_pairs` (5% + 5% dollar-neutral), `size_opt_debit` (≤ 0.50% NAV per
package, ≤ 5% total option debit, `verify_contract` checks multiplier/deliverable), `size_opt_credit` +
`is_defined_risk` + `max_loss_vertical` (max defined loss ≤ 0.50% NAV, collateral returned for reservation),
`size_opt_naked` + `stress_loss_short_option` (stress loss ≤ 1% NAV on the predefined ±10/±20% move with
extrinsic×2 scenario; raises if margin is zero), `size_opt_covered` (100 shares per call, underlying ≤ 10% NAV),
`size_opt_cashsec` + `CollateralBook` (strike×100 reserved, no double counting, 10% name cap),
`arbitrage_edge` (trade only if executable edge after fees, borrow, funding and margin > 0 and shorts located).
`*_ctx` helpers read NAV/positions from the Context.

## 6. Configuration (`engine/config.py: BacktestConfig`)
`initial_cash` (1e6), `account_type` (REG_T_MARGIN | CASH), `fees` (name or `FeeSchedule`, default
IBKR_PRO_TIERED), `latency_bars` (0), `latency_quotes` (1 s), `bar_fill` (`participation_cap` 0.10,
`allow_fractional` False, `ssr_mode` 'ask_limit', `same_bar_exit` True, `limit_through_ticks` 1), `auction`
(cutoffs 2/10 min, auction volume shares), `combo` (`price_improvement` 0, `snapshot_tolerance` 0),
`spread` (`SpreadConfig`: method/window/floors/time-of-day/default/`tick_floor`), `impact` (`SquareRootImpact(k=0.1)`),
`daily_open_spread_mult` (1.0), `use_quotes` ('auto'), `resolve_intrabar_freq` ('1m'), `short_sale_rules`,
`borrow` (`BorrowConfig`: default 0.30%, `require_borrow_data`), `enforce_locates`, `halt_events`,
`financing` (`FinancingConfig`: spreads 0.50%/1.50%, rebate 0, day count 360), `corporate_actions`
(withholding, fractional shares), `option_margin`, `option_lifecycle`, `check_margin_on_submit`, `margin_calls`,
`rf_tenor` ('3M'), `stats_window` (20), `lookback_sessions` (60), `debug` (invariants every step),
`record_intraday_equity`, `max_option_mark_staleness` (5 days).

## 7. Testing support
`atlasbt/engine/testing.py`: `InMemoryPortal` (DataPortal over DataFrames incl. PIT adjustment, quotes, option
quote snapshots, actions, events, borrow, rf) and `synthetic_*` builders — synthetic data for mechanics tests only.

## 8. Performance (single core, synthetic data)
~3.7 s per year of 1-minute bars on one symbol with a bracket strategy (≈ 75 s for 20 years); ~10 s for
2 years × 1,000 symbols daily with ~500 open positions (≈ 100 s for 20 years).

## 9. Known limitations
* Bar-mode fills assume trade prints are mid prices; maker fills ignore queue position beyond the 1-tick rule and
  adverse selection; maker rebates are ignored.
* ORF aggregate and pre-2003 TAF rows are approximate; Cboe SPX fees before 2026-10 use the 2026 schedule; IBKR
  commission schedules are the 2026 ones applied to all history.
* Leveraged-ETF maintenance surcharges, portfolio margin, pattern-day-trader rules, settlement-date (T+1/T+2) cash
  restrictions and good-faith violations in cash accounts are not modelled.
* Early exercise/assignment, pin risk and non-standard deliverables are left to the OPTIONS agent's lifecycle;
  option marks are updated only at session ends.
* SSR uses bar lows (a trigger inside the current bar applies to the whole bar: conservative); the 2005-2007
  Reg SHO pilot exemptions are not modelled.
* Dividends paid after a trade closes are attributed to the trade at the ex-date (entitlement amount).

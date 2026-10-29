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
   actions with `ex_date` in `(previous session, session]` are applied (an ex-date on an exchange holiday is applied
   on the next session, with a warning): splits, dividends, delistings, symbol changes, spin-offs — including the
   OCC adjustment of option positions/orders on a split underlying and the open trade in the trades table; resting
   orders are adjusted or cancelled (see Orders); AM-settled options are settled (option lifecycle hook);
   `strategy.on_session_start(ctx)`.
2. **Each step t** (end of bar `[t-Δ, t)`):
   1. *Fills*: every working order is offered the bars/quotes in `(prev_t, t]` whose bar **start** is at or after
      the order's `eligible_at` (`submitted_at + latency`). An order submitted at the close of bar k can therefore
      only fill from bar k+1's open onward; nothing ever fills at a price printed before the order existed.
   2. `on_fill` (and `on_order_event`) callbacks for this step's fills, delivered in the SAME step right after
      the fills and before marking, with `ctx.now = t` (no data beyond `t` is visible); an order submitted there
      (e.g. a protective stop) is eligible from the next bar. Use `ctx.bracket` when the stop must protect the
      position from the entry fill itself (bracket children are evaluated inside the entry bar). Fills booked
      during session-end processing (expiry/assignment) are delivered at the end of that session.
   3. Mark to market; engine session rules (`flat_by`), then `strategy.on_bar(ctx)` with `ctx.now = t`.
   4. MAE/MFE and bars-held tracking for open trades; debug invariants.
3. **Session end** (`ctx.now` = last step): `strategy.on_session_end`; option marks FIRST, then PM/physical
   option expiry and early-assignment tests (exercise/assignment/cash settlement). Marks are same-session only:
   per position the latest observation time-stamped in this session wins, either a fill in the contract or a
   `portal.option_quotes` snapshot (real quote T0-T2; the data portal falls back to a same-day chain-surface mark
   T2, then a PIT model value T3). `Position.mark_ts` / `mark_fidelity` record the source. With no same-session
   observation the old mark is kept for NAV only, the run is flagged T2 ("no same-session option mark"), and the
   lifecycle sees the old `mark_ts` and does not use it: a stale or missing mark never counts as zero extrinsic
   value, so no early assignment is generated from it. Then financing (cash interest, margin
   interest, borrow fees) for the calendar days to the next session; DAY-order expiry; Reg-T margin check
   (margin call -> liquidation market orders that fill at the next open, logged in
   `diagnostics["margin_calls"]`); NAV and position snapshot.

Orders submitted at a session close (daily strategies) are DAY orders for the **next** session and fill at its
open (or later per their type).

### Point-in-time guarantees
* `ctx.portal()` is a `PITPortal`: every query is pinned to `as_of = ctx.now`; an explicit later `as_of` raises
  `LookaheadError`; any row the portal returns with a knowledge time after `now` raises (data-layer contract breach,
  never silently dropped); bar `end` bounds are clipped to `now`, so the in-progress bar is never returned.
* `ctx.history_arrays(symbol, n, freq)` is the fast path: a dict of READ-ONLY numpy arrays (start, end, open,
  high, low, close, volume, session) for the last `n` completed bars (raw base-frequency bars are views on the
  engine cache; ~40x faster than `history` for a 50-symbol 1-minute scan). Same PIT guarantee.
* History is never silently truncated: if the engine cache (loaded with `lookback_sessions` of pre-start history)
  may not hold `n` bars, `history` / `history_arrays` fall back to the PIT portal for the full request.
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
reductions and bracket-child activations happen in that order. After an entry fills at path position `s` inside a
bar, the remainder of the bar is `entry -> adverse extreme -> favourable extreme -> close` **only if the original
path reaches the favourable extreme at or after `s`**; otherwise (the favourable extreme already happened before
the entry, e.g. a buy limit filled on the way down from the high) the remainder is `entry -> adverse extreme ->
close`. The adverse extreme is always assumed still reachable. A market entry at the open (`s = 0`) keeps the full
adverse-first path. Fill timestamps after a same-bar entry are mapped onto `[entry time, bar end]`, so an exit is
never stamped before its entry. Fills that depended on this assumption are tagged T2 ("intrabar order assumed
adverse-first"). If finer bars exist
(`resolve_intrabar_freq`, default `'1m'`) and a coarse bar is ambiguous (two or more orders could trigger, or a
bracket parent may fill), the bar is replayed on the finer bars and the fill is tagged "intrabar path resolved with
1m bars" (stays T1). Intrabar fill timestamps are interpolated from the path position (floored to the second).

### Orders (`engine/orders.py`, `ctx.order / ctx.submit / ctx.bracket`)
* Types: MARKET, LIMIT, STOP (gap-aware stop market), STOP_LIMIT, MOO, MOC, COMBO. TIF: DAY, GTC, IOC.
* Latency: 0 for bar-driven fills (the next bar starts at `t`), 1 s for quote-driven and option/combo fills.
* DAY validity = the session containing `eligible_at` if before its close (+4 h with `outside_rth`), else the next
  session. MOO joins the open of the first session whose 09:28 cutoff is after `eligible_at`; MOC the first close
  whose 15:50 cutoff is after it. Unfilled auction remainder is cancelled.
* Fill-relative brackets: `ctx.bracket(..., stop_offset=, target_offset=)` or `stop_fn=/target_fn=` (callables
  `f(fill_price, side)`) anchor the children to the ACTUAL entry fill price (e.g. "stop 2×ATR from fill", "2R
  target"); they are active from the entry's path position onward (same rules as other bracket children) and the
  anchored stop is the trade's initial stop for R.
* Engine-internal orders (flat_by flatten, margin-call liquidation) cannot be cancelled by `ctx.cancel`,
  `ctx.cancel_all` or `ctx.flatten`; attempts are ignored and logged in `diagnostics["warnings"]`.
* Short-sale marking at submission uses the projected position at the order's execution time: current position +
  pending same-symbol sells + pending buys sure to execute first (MARKET, MOO, earlier/same-auction MOC). A
  same-close MOC exit of a position bought MOO is therefore not a short sale (no locate / margin check).
* Fill-relative brackets (`stop_offset`/`target_offset`/`stop_fn`/`target_fn`) anchor children to the actual
  entry fill; when a level evaluates to None/NaN that child is NOT created (`CHILD_SKIPPED` in
  `diagnostics["order_events"]`), the other children keep their OCO link.
* `ctx.bracket(..., time_exit_moc=SESSION_DATE | True)`: the time exit is an MOC child (tag `time_exit`, reduce-only,
  sized to the position actually open, OCO with stop/target) that joins that session's closing auction (`True` =
  the session of `time_exit`; never earlier than the first close the child can still reach before the MOC cutoff).
  Use it on daily bars, where a market time exit could only fill at a later open. Within one window the engine
  processes the opening auction, then continuous trading (stops/targets intrabar), then the closing auction, so a
  stop hit during the session cancels the MOC leg and an MOC created by a same-bar entry fill still joins the close.
* Bracket handles: `ctx.bracket(...)` returns the entry order; `ctx.bracket_group(order)` gives the public OCO group
  (children carry it as `oco_group`); `ctx.amend_bracket(order|group, stop=, target=, time_exit=)` re-prices
  working children (cancel/replace: effective from the next bar) or adds a missing leg sized to the open exit
  quantity (reduce-only). `diagnostics["orders"]` keeps `original_quantity` next to the (possibly OCO-reduced)
  `quantity`.
* OCO groups use reduce-on-fill semantics (partial fills reduce the siblings). Brackets: entry + protective stop +
  target + time exit; children are created in proportion to the entry's fills, share an OCO group and default to
  GTC. `initial_stop` (or the bracket stop) is recorded for R-multiples.
* Submit-time validation: quantity (whole shares unless `allow_fractional`; integer contracts), required prices,
  tick rounding (buy limits down / sell limits up; stops away from the market), session rules, short-sale
  detection + locate, cash-account restrictions, buying power. Rejections are listed in
  `diagnostics["rejected_orders"]`; all orders in `diagnostics["orders"]`.
* Buying power: the order is priced at the WORSE of its order price and the current reference price (a
  non-marketable buy limit keeps its limit); pending orders get NO credit unless they increase exposure (then they
  are added, priced the same conservative way); estimated explicit fees are included (non-mutating fee estimate).
  Consequence (realistic, as at a broker): a 100% -> 100% rotation submitted in one step is rejected unless cash
  covers the buys before the sells fill; the reject reason says so explicitly. Keep a cash buffer or sequence it.
* "Exposure-increasing" is measured against position + pending non-child working orders (engine flatten orders
  included), so two sells cannot open a short that neither looks like on its own.
* Short-sale marking is decided at FILL time (position after earlier fills in the bar + earlier-entered pending
  sells that are not reduce-only exits); a mark set at submission is never cleared; the locate check follows.
* Exits are reduce-only: bracket children and engine flatten/liquidation orders are shrunk to the remaining
  position and cancelled when flat (a margin-call liquidation shrinks the protective stop instead of deleting it).
* Corporate actions (FINRA Rule 5330 style): forward split -> resting sell stops / sell stop-limits / buy limits are
  adjusted (price ÷ ratio, quantity × ratio, bracket links and child templates kept); every other order on the
  symbol is cancelled; reverse splits, symbol changes, spin-offs and delistings cancel all orders on the symbol.
  Every engine-initiated change (also session-rule and margin-call cancellations, reduce-only shrinking) is
  delivered to `Strategy.on_order_event(ctx, event)` and logged in `diagnostics["order_events"]`.
* Session rules from sizing profiles: DAY_RISK => no new entries after 15:15 ET, flat by 15:55 ET; DAY_NOTIONAL =>
  flat by 15:55 ET. Times are defined on a regular session and shift with the actual close on early-close days
  (close − 45 min / close − 5 min). At the entry cutoff resting exposure-increasing non-child orders are cancelled
  (logged); non-market entries submitted exactly at the cutoff are rejected. Flattening happens at the last step
  that ends at or before `flat_by` (e.g. 15:30 with 30-minute bars) with market orders (tag `flat_by`); after
  that only orders reducing exposure net of pending orders are accepted, and every step until the close the engine
  re-checks that position + pending flatten orders net to zero. A daily-bar run with a DAY_* profile, or an
  intraday frequency with no step before `flat_by`, is `NOT_TESTABLE` (it cannot exit before the close).

## 2. Execution realism (`atlasbt/execution`) — assumptions, defaults, sources

| Topic | Default | Source / justification |
|---|---|---|
| Market order, bars only (T1) | next bar open x (1 ± half-spread ± impact) | ARCHITECTURE §5; trade price assumed to be mid |
| Stop market | triggers when traded price crosses; fill at max(stop, first price after trigger); gap -> open | ARCHITECTURE §5 |
| Limit (resting) | fills only if price trades THROUGH by >= 1 tick, at the limit price, no spread cost (MAKER) | queue position unknown; touch is not enough |
| Marketable limit | fills like a market order capped at the limit (TAKER) | |
| Stop-limit | stop trigger, then marketable-if-possible else resting limit | |
| Marketable limit | takes only the largest size whose MODELLED price base·(1+hs+impact(q)) is within the limit, at that modelled price; the rest rests passively under the trade-through rule | |
| Tick grid | taker fill prices rounded to the tick against the trader (buys up, sells down), capped at any limit | Rule 612 |
| Participation cap | 10% of bar volume per fill; remainder keeps working; NaN volume = no available liquidity (no fill, counted) in every session | ARCHITECTURE §5 |
| Halts | zero-volume RTH/DAY bars and `HALT` events (`effective_at`..`resume_at`) => no fills | |
| Quotes (T0) | buy at ask, sell at bid, first quote strictly after `eligible_at`; quotes classified PRE/RTH/POST from the calendar and only used for RTH unless `outside_rth`, never before the order's validity session; displayed size is consumed across orders in priority order; resting limits never fill on quotes alone; locked/crossed quotes skipped | |
| MOO / MOC | official open/close print ± impact, no spread; cap 10% of estimated auction volume (2% / 8% of daily volume in daily-bar mode, open/close bar volume intraday) | NYSE cutoffs 09:28 / 15:50 ET |
| Fill source per window | `use_quotes='auto'`: a stock order in a step window fills from quotes only if the symbol has ≥ `quote_min_per_hour` (60) quotes per RTH hour overlapping that window, else from bars (never left unfillable because quotes exist elsewhere); symbols with no bars always use quotes. Every fill records its source in `fidelity_reason` ("fill source: quotes" / "fill source: bars (...)") | |
| Options & combos | every snapshot in `(eligible_at, now]` is evaluated in time order via `portal.option_quotes` (the FIRST snapshot after submission first; GTC limits re-checked on each); all option legs from the same snapshot timestamp or REJECT; natural prices (buy ask / sell bid) with optional `price_improvement`; stock legs priced from the open of the first bar STARTING at/after submission ± estimated half-spread + impact (T2), and the combo WAITs if that bar has not started by the snapshot time; when the stock has synchronized quotes (`quote_min_per_hour` coverage) the stock leg instead uses the NBBO at the snapshot time (≤ `combo.stock_quote_max_age` = 5 s old, T0). Modelled (T3) option snapshots are accepted and pass their tier through. `combo.reject_locked` (default True) never fills from a locked/crossed option-leg snapshot: the order WAITs for the next valid snapshot (not rejected; it stays working until filled, cancelled or expired). Option instruments (single orders and every COMBO leg) are replaced at submission by the instrument-master record (`portal.instrument`): deliverable, tick size, exercise/settlement style come from the master (a strategy-built default deliverable of 100 is overridden with a warning); a contradiction on multiplier / strike / right / expiry / underlying, or an explicitly set wrong deliverable, REJECTs the order with the reason | ARCHITECTURE §7 |
| Spread model (default `spread_model="calibrated"`) | `CalibratedSpreadModel`: PIT, era-aware model of the quoted NBBO spread fitted to NYSE Daily TAQ (ADV$, volatility, price, ETF flag; era/tick-regime adjustment; Corwin-Schultz/Abdi-Ranaldo cap below the calibration ADV range); inputs from prior sessions only. The engine passes as-traded closes/volume, session dates, split-adjusted `prior_close`, the asset class and listing `exchange` from the instrument master (no record: asset class left to the model and flagged "asset class inferred") and the symbol; the model computes its own PIT ADV$/sigma (IPO sessions priced). Fills pass ADV$ and asset class to the time-of-day profile (liquid-ETF profile). Taker fills carry the reason "spread modelled (calibrated)" plus each model qualifier ("spread: era-extrapolated", "spread: listing assumed", "spread: stress regime", "spread: asset class inferred", ...); their tier follows the bar-fill tier mapping below. Example: SPY 2019 round trip ≈ 0.4 bp spread + impact (Corwin-Schultz on the same bars: ~45 bp). Legacy `spread_model="corwin_schultz"`: Corwin-Schultz (2012) 21-day mean, fallback Abdi-Ranaldo (2017), then Roll (1984). An explicitly customised `spread=SpreadConfig(...)` selects the legacy model | **docs/SPREADS.md** (calibration, validation, §8 integration) |
| Spread floors | calibrated: era tick (1/8, 1/16, decimal; Rule 612 at the fill price), default half-spread 10 bp if no ADV/vol (T2). Legacy: ≥ 1 tick and price buckets <$1: 100bp, <$5: 30bp, <$10: 10bp, <$20: 4bp, else 1bp; default 20bp | docs/SPREADS.md |
| Time-of-day | MEASURED NBBO profile in 5-minute buckets (docs/SPREADS.md): first 5 min ×4.3 for liquid names (ADV$ ≥ $100M) / ×2.9 for illiquid ones, decaying to ~1 by late morning, and NARROWER into the close (~0.5–0.8×); ×3 extended hours. The legacy model uses the coarsened illiquid curve (×2.9 / 2.0 / 1.7 / 1.5 / 1.15 over the first hour; ×0.87 / 0.83 / 0.8 over the last 2 h / 30 min / 5 min). Daily-bar open fills use the daily average (×1.0, `daily_open_spread_mult`). The earlier engineering-choice profile is superseded | docs/SPREADS.md |
| Impact | `k * sigma_daily * sqrt(Q$/ADV$)`, k = 0.1, cap 5%/fill; sigma and ADV = 20 prior sessions | Almgren et al. (2005) cost ≈ 0.1·σ·sqrt(Q/V) for Q/V ≤ 10%; Tóth et al. (2011) Y≈0.5-1 is peak impact (stress k=0.5); Frazzini-Israel-Moskowitz (2018) |
| Reg SHO | before 2007-07-06: tick test => every short sale restricted; 2007-07-06..2011-02-27: none; from 2011-02-28: Rule 201 (trigger: low <= 90% of prior close; rest of day + next day). Bar mode `ssr_mode='ask_limit'`: short sale posted at the estimated ask and fills only if price trades through it; `'reject'`; `'ignore'` | SEC Rel. 34-55970 (uptick repeal), Rule 201 compliance date 2011-02-28 |
| Locates | `portal.borrow(symbol)`; if no data: easy-to-borrow at 0.30% p.a. (T2), or reject with `BorrowConfig.require_borrow_data` | ARCHITECTURE §5 |
| Ticks | stocks Rule 612; options `Instrument.tick_size` | |
| Extended hours | PRE/POST bars fill only orders with `outside_rth=True` | |
| Fill fidelity | T0 quote fills; T1 bar-mode fills (real bars at the strategy resolution, next-bar fill, modelled spread, incl. the daily-bar next-open fill); auction fills T1; T2 only for true proxies per the deterministic mapping below; option / combo fills pass the quote data tier through (T0-T3) | core.types; ARCHITECTURE §5 |

### Bar-fill tier mapping (`execution.fills.bar_fill_tier`, `retier_fill`)
Deterministic, reason set -> tier, applied at fill time; the runner can re-tier finished runs from the fills frame
(`retier_fill(fidelity, liquidity, fidelity_reason, asset_class)`) without re-running. It applies to bar-mode
stock/ETF fills with liquidity TAKER/MAKER; quote fills (T0), auction fills (T1), option/combo fills (data tier)
and lifecycle fills keep their recorded tier. A bar-mode fill is **T1_BAR** unless one of its reasons (the
`fidelity_reason` split on "; ") is a true proxy, which makes it **T2_PROXY**:

| Reason (exact / prefix / substring) | Why T2 |
|---|---|
| `spread default (no estimate)` | no spread estimate at all |
| `intrabar order assumed adverse-first (bar data)` | intrabar ordering assumed without finer bars |
| prefix `spread estimated (` | legacy Corwin-Schultz / Abdi-Ranaldo / Roll spread (`spread_model="corwin_schultz"`) |
| `spread: Corwin-Schultz fallback` | calibrated model fell back to CS/AR (no volume) |
| `spread: asset class inferred`, prefix `spread: listing assumed`, prefix `spread: stress regime` | spread uncertainty high |
| substring `daily proxy` | a daily bar standing in for an intraday decision time |

Everything else stays T1 with its reasons kept: "spread modelled (calibrated)" with era-extrapolated /
pessimistic-floor / ADV-below-range / sigma-prior qualifiers, "daily bar: next-open fill" at the strategy's own
resolution, "intrabar path resolved with 1m bars", participation cap, marketable-limit sizing, "impact not
modelled". Run-level flags (SSR inputs, default borrow, rates) are recorded in the fidelity scorecard, not on fills.

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
* Marks: a corporate-action-adjusted mark (e.g. post-split) is used — by `ctx.mark` and by buying-power pricing —
  until a post-ex-date bar prints; prices of symbols without a position are divided by split ratios that went ex
  after their last bar. A held symbol with no bar for `stale_price_sessions` (5) sessions is flagged T2 with a
  warning ("stale price").
* Splits: qty×ratio, prices÷ratio, fractional shares cashed out at the post-split mark (unless fractional shares
  allowed). Cash dividends: entitlement at ex-date on the prior-close position (receivable, NAV-neutral), cash on
  `pay_date`, optional withholding (`CorporateActionConfig.withholding_rate`, default 0); shorts pay in lieu.
  Dividends without a `pay_date` (e.g. Yahoo) are paid at ex-date + `pay_date_lag_days` (7) calendar days rolled
  to the next session, flagged T2 ("pay_date unknown, assumed ex+7d"); the receivable keeps NAV neutral meanwhile,
  so only cash-interest timing is affected.
  Delisting: cash-out at `amount`; missing value => last mark (T2 flag). Symbol change: position moved.
  Spin-off: `ratio` new shares, `amount` = cost-basis fraction. Action dates accept naive dates, UTC-midnight
  date labels, or ET times.
* Financing (posted daily, actual/360 over calendar days to the next session, Friday→Monday = 3 days): cash
  balance net of short-sale proceeds earns `max(rf − 0.50%, 0)`; a debit balance pays `rf + 1.50%`; short borrow
  fee on short market value at the `portal.borrow` rate; short rebate 0 (retail). Defaults reflect IBKR Pro's
  benchmark ± spread schedule for small balances. `rf` = `portal.risk_free_rate(as_of, '3M')`; recorded per session
  in `meta["rf_daily"]`.
* Reg-T margin (`margin.py`): initial 50% long / 50% short (150% total); maintenance 25% long, short max($5/sh, 30%)
  at ≥ $5 and max($2.50/sh, 100%) below $5 (FINRA 4210). Long options have no loan value: 100% of their market value
  is added to the requirement (75% when more than 9 months to expiry, FINRA 4210(f)(2)(C)). Pre-trade check includes
  pending exposure-increasing orders only. Margin call when equity < maintenance at a session end => proportional
  reduction of every requirement-carrying position to maintenance ≤ equity/1.10, filled at the next open; only
  exposure-increasing orders are cancelled, exits are kept and shrink with the position. CASH accounts: no shorts;
  purchases need cash net of estimated fees and of cash reserved for short puts (strike × deliverable per
  contract, existing and pending); naked calls rejected.
* **Option plug-ins — defaults are the OPTIONS agent's implementations** (`atlasbt/options/plugins.py`):
  `BacktestConfig.option_margin=None|"cboe"` -> `CboeStrategyMargin()` (Cboe strategy-based requirement; it charges
  long options in full itself, so the Reg-T model does not add them again), and
  `option_lifecycle=None|"occ"` -> `OccOptionLifecycle(rate=portal risk-free (simple -> continuous),
  dividends=portal CASH_DIVIDEND/SPECIAL_DIVIDEND actions known at the session, next 120 days)` (OCC ex-by-ex,
  AM/PM settlement, early-assignment model). The conservative stand-ins below stay selectable with
  `"conservative"` (or pass any instance). Whatever plug-in is used, the engine's price adapter falls back to the
  last known underlying close when a settlement close is missing (T2 flag), and an option the plug-in would expire
  worthless WITHOUT any settlement price is instead closed at its last mark (T2).
* **Plug-in interfaces**
  * `portfolio.margin.OptionMarginModel.requirement(account, underlying_price) -> float` (total option margin,
    never 0 for naked shorts). Default `ConservativeOptionMargin`: long premium paid in full; covered calls/puts;
    verticals/calendars at max loss; naked max(premium + 20% (15% broad index) of underlying − OTM, premium + 10%
    of underlying (calls) / strike (puts)); cash accounts: put strike×multiplier secured. Plug in via
    `BacktestConfig(option_margin=...)`.
  * `portfolio.hooks.OptionLifecycle` with `at_session_start(session, prev_session, option_positions, prices)` and
    `at_session_end(session, option_positions, prices)` returning `OptionEvent`s (EXPIRE_WORTHLESS, EXERCISE,
    ASSIGNMENT, CASH_SETTLE, SETTLE_AT_MARK with option close price and optional stock delivery). The engine books
    them and delivers every resulting fill (option leg and stock delivery; `order_id = -1`, `liquidity` =
    EXERCISE / ASSIGNMENT / SETTLEMENT) to `Strategy.on_fill` in the same step, after the account is updated, so
    wheel / covered strategies can react without polling positions. `prices` also offers `last_close(symbol)`. Default `DefaultOptionLifecycle`: PHYSICAL/CASH_PM expire at the
    expiry session close (official close); CASH_AM settle at the next session's open (SOQ proxy, T2); automatic
    exercise if ITM by ≥ $0.01; non-standard deliverables honoured (intrinsic = deliverable·S/multiplier − strike;
    delivery price = strike·multiplier/deliverable); a missing settlement price falls back to the last known
    underlying close (T2), else the option is closed at its last mark (T2) — never at 0. No early
    exercise/assignment. Plug in via `BacktestConfig(option_lifecycle=...)`.
  * Splits of an underlying apply the OCC contract adjustment from `atlasbt.options.lifecycle.adjust_for_split`
    (OPTIONS agent) to every option position and working option order on it (logged as `OCC_SPLIT_ADJUSTMENT`).
* Options are marked at session ends from `portal.option_quote` mids (intraday option NAV is stale).

## 4. Results (`core.types.BacktestResult`, columns per `core/schemas.py`)
* `equity` NAV at each session close (UTC); `returns` daily simple (first vs initial cash).
* `fills`: ts, order_id, symbol, asset_class, multiplier, quantity, price, commission, exchange_fees,
  regulatory_fees, spread_cost, impact_cost (both informational, already in price), fidelity (int),
  fidelity_reason, liquidity, note, reference_mid, adv (20d shares known at fill), underlying_price (options),
  notional, tag.
* `trades`: position-level round trips — trade_id, symbol, side, entry/exit ts and prices, quantity, pnl (net of
  fees + attributed borrow fees, dividends (only the accrual created by each action), in-lieu payments and
  withholding), pnl_gross, costs, r_multiple and initial_risk (from the ACTUAL entry fill to the initial stop),
  mae/mfe (USD, on max size), bars_held, multiplier, exit_reason (exit order tag:
  stop/target/time_exit/flat_by/margin_liquidation/delist/assignment/cash_in_lieu/...; `open` if still open);
  extra `planned_risk` / `planned_r_multiple` when the strategy passes `planned_entry=` to `ctx.bracket/submit`.
  Splits rescale the open trade (quantities × ratio, prices ÷ ratio, cash-in-lieu booked as a partial exit),
  symbol changes re-key it, spin-offs move the allocated basis into a child trade.
* `positions` (end-of-day snapshots), `ledger`, `fidelity` = `{"tiers": {...}, "reasons": [{"tier", "reason",
  "count"}], "grade"}` (grade = worst tier carrying ≥ 5% of notional, any strategy/engine T2+ flag, or required data
  input: T0→A, T1→B, T2→C, T3→D, NA→E), `diagnostics` (rejected orders, all orders, margin calls, corporate actions,
  option events, financing totals, simulation counters, invariant checks, guard violations, runtime, logs),
  `meta` (initial_capital, rf_daily, risk_free_rate = Σ rf_daily × 360 / total calendar days, start, end, bar_freq,
  universe, config, data_sources,
  survivorship_bias, n_trials = grid size, plus engine details).
* `RunStatus.NOT_TESTABLE` when a critical `Requirement` is unavailable (`portal.has` False or fidelity NA/worse
  than `min_fidelity`). Symbols come from `Requirement.symbols`, the strategy `universe`, or `universe_name`
  resolved via `portal.universe(name, as_of=first open)` and filtered by the coverage rule below; `symbols=("*",)` asks portal-wide
  (`portal.has("*", "events", KIND)`, `portal.has("*", "fundamentals", FIELD)`); bars requirements with
  `sessions` including PRE/POST are probed over the first 10 sessions and fail if that session has no bars
  (quotes requirements: no quote time-stamped in that session; options: no chain snapshot in that window); `FAILED` on strategy exceptions (traceback in diagnostics; `raise_errors=True` re-raises).
  `LookaheadError` always propagates. A `NotTestableError` raised by strategy code (init / callbacks) ends the
  run as `NOT_TESTABLE` with the exception message as the reason (also with `raise_errors=True`; it is a status,
  not an error).
* Universe coverage (`BacktestConfig.min_universe_coverage`, default 0.95): `portal.universe(name)` as seen by
  the strategy (and the `universe_name` used for requirement checks) contains only members with VERIFIED data
  (`has` for base-frequency bars, using the windowed `has(sym, "bars", freq, start=, end=, as_of=)` over the run
  window when the portal supports it, and for every critical requirement without explicit `symbols`; fidelity
  only tiers: an unknown tier (NA) does not drop a member, a known tier worse than `min_fidelity` does;
  memoised per symbol; security keys such as `T~CIK...` work as symbols throughout). Members are not each required: if coverage (kept / members) at the run start or at any session
  where the universe is queried is below the minimum, the run is `NOT_TESTABLE` with the coverage figure and
  example members (1.0 = the old every-member rule). Coverage < 100% flags T2 "universe NAME coverage X%
  (members without verified data excluded)"; `meta["universe_coverage"]` holds the series (date, members,
  kept, coverage), every dropped member (first/last date, count, reason) and the minimum, and
  `meta["survivorship_bias"]` notes the exclusion. Static `universe` lists remain strict.
* `Strategy.on_run_end(ctx)` is called once after the last step (before results are built).
* `fidelity_of(..., "actions")` for a requirement is the WORST tier over all actions (including `"*"`) in the
  in-memory portal; the real data `Portal` currently returns T2_PROXY for actions irrespective of content.

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
`size_opt_cashsec` + `CollateralBook` (strike × deliverable shares reserved per contract, no double counting, 10%
name cap), `size_swing_notional` also nets the existing same-side notional in the instrument,
`arbitrage_edge` (trade only if executable edge after fees, borrow, funding and margin > 0 and shorts located).
`*_ctx` helpers read NAV/positions from the Context.

## 6. Configuration (`engine/config.py: BacktestConfig`)
`initial_cash` (1e6), `account_type` (REG_T_MARGIN | CASH), `fees` (name or `FeeSchedule`, default
IBKR_PRO_TIERED), `latency_bars` (0), `latency_quotes` (1 s), `bar_fill` (`participation_cap` 0.10,
`allow_fractional` False, `ssr_mode` 'ask_limit', `same_bar_exit` True, `limit_through_ticks` 1), `auction`
(cutoffs 2/10 min, auction volume shares), `combo` (`price_improvement` 0, `snapshot_tolerance` 0),
`spread_model` ("calibrated" | "corwin_schultz"), `spread_calibrated` (`CalibratedSpreadConfig`, docs/SPREADS.md), `spread` (legacy `SpreadConfig`: method/window/floors/time-of-day/default/`tick_floor`), `impact` (`SquareRootImpact(k=0.1)`),
`daily_open_spread_mult` (1.0), `use_quotes` ('auto'), `resolve_intrabar_freq` ('1m'), `short_sale_rules`,
`borrow` (`BorrowConfig`: default 0.30%, `require_borrow_data`), `enforce_locates`, `halt_events`,
`financing` (`FinancingConfig`: spreads 0.50%/1.50%, rebate 0, day count 360), `corporate_actions`
(withholding, fractional shares), `quote_min_per_hour` (60), `option_margin` (None/'cboe'/'conservative'/instance), `option_lifecycle` (None/'occ'/'conservative'/instance), `check_margin_on_submit`, `margin_calls`,
`rf_tenor` ('3M'), `stats_window` (20), `lookback_sessions` (60), `debug` (invariants every step),
`record_intraday_equity`, `max_option_mark_staleness` (5 days), `stale_price_sessions` (5),
`min_universe_coverage` (0.95).

Per-strategy overrides: the class attribute `Strategy.config_overrides = {...}` may set ONLY `option_lifecycle` and `option_margin`
(whitelist; any other key raises `ValueError`); applied overrides are recorded in `meta["config_overrides"]`.

## 7. Testing support
`atlasbt/engine/testing.py`: `InMemoryPortal` (DataPortal over DataFrames incl. PIT adjustment, quotes, option
quote snapshots incl. `option_quotes(instrument, start, end, as_of)`, actions, events, borrow, rf; `fidelity={(symbol, kind[, freq]): Fidelity}` configures `fidelity_of` per symbol/kind/freq, `"*"` = any symbol; without an override, events/actions report the worst tier of the matching records and the event KIND is never parsed as a bar frequency) and `synthetic_*`
builders — synthetic data for mechanics tests only.

PIT hardening: the Context and the PIT portal are slotted proxies with name-mangled fields (no `ctx._bt`,
no `ctx.portal()._p`; the PIT portal exposes a WHITELIST of data methods (bars, quotes, option chain /
option_quotes, actions, events, fundamentals / fundamentals_history, borrow, risk_free_rate(_info), universe,
universe_survivorship_bias, has, fidelity_of, ...) and raises `AttributeError` for anything else), and `tests/engine/test_eng_strategy_lint.py` statically rejects strategy code that accesses
private attributes of non-`self` objects, uses `getattr(x, "_...")`, imports `inspect`/`gc`/`ctypes`, touches
`sys.modules`, or imports engine internals (`backtester`, `context`, `market`, `pit`).

## 8. Performance (single core, synthetic data)
~4-8 s per year of 1-minute bars on one symbol with a bracket strategy (≈ 1.5-3 min for 20 years; the range
reflects machine load); ~10-20 s for 2 years × 1,000 symbols daily with ~500 open positions.

## 9. Known limitations
* Bar-mode fills assume trade prints are mid prices; maker fills ignore queue position beyond the 1-tick rule and
  adverse selection; maker rebates are ignored.
* ORF aggregate and pre-2003 TAF rows are approximate; Cboe SPX fees before 2026-10 use the 2026 schedule; IBKR
  commission schedules are the 2026 ones applied to all history.
* Leveraged-ETF maintenance surcharges, portfolio margin, pattern-day-trader rules, settlement-date (T+1/T+2) cash
  restrictions and good-faith violations in cash accounts are not modelled.
* Early exercise/assignment and pin risk come from the OPTIONS agent's `OccOptionLifecycle` (default); its
  physical-delivery price is the strike per share, which is only exact for standard 100-share deliverables (the
  conservative `DefaultOptionLifecycle` handles non-standard deliverables). Option marks are updated only at session
  ends; OCC-adjusted contracts keep their scaled mark until quotes exist under the new symbol (stale-mark T2 flag).
* Resting orders after corporate actions follow FINRA 5330 for forward splits only (dividend price adjustments of
  resting orders are not applied).
* SSR uses bar lows (a trigger inside the current bar applies to the whole bar: conservative); the 2005-2007
  Reg SHO pilot exemptions are not modelled.
* Dividends paid after a trade closes are attributed to the trade at the ex-date (entitlement amount).

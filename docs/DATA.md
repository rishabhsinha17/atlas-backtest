# atlasbt data layer (`atlasbt/data`)

Owner: DATA agent. Contracts: `docs/ARCHITECTURE.md` §3 (canonical frames), `atlasbt/core/protocols.py`
(`DataPortal`). This document lists every source, what was actually downloaded (numbers measured on
2026-10-28 from the store), the point-in-time (PIT) conventions, fidelity tiers, rate limits,
licensing and the known gaps. **Read §8 (gaps) before trusting any cross-sectional backtest.**

Coverage numbers in this file were produced by the store inventory on 2026-10-28; re-run
`python -m atlasbt.data.fetch --set daily` to extend them (it is incremental).

---------------------------------------------------------------------------------------------------

## 1. Layout

```
atlasbt/data/
  config.py      paths (cache/), User-Agent (env ATLASBT_USER_AGENT, SEC_USER_AGENT), host throttles
  http.py        polite HTTP client: per-host throttle, backoff on 429/5xx, raw-response disk cache
                 (cache/raw/<host>/..; >256 kB bodies gzip'd), http->https redirect upgrade
  store.py       ParquetStore: cache/store/<kind>/<source>/<SYMBOL>[/<YYYY>].parquet, idempotent
                 upserts, provenance metadata (b'atlasbt' key: source, fetched_at, first_fetched_at,
                 fidelity, adjusted, key, rows, extra...)
  frames.py      canonical frame builders/validators, NYSE schedule (1960->), session tagging
  adjust.py      Yahoo un-adjustment, PIT split/dividend factors (total-return back-adjustment)
  rates.py       T-bill discount -> money-market / bond-equivalent / continuous conversions
  portal.py      Portal (DataPortal implementation) + causal intraday resampler
  events.py      assembly of EARNINGS / EARNINGS_SCHEDULE / EARNINGS_RELEASE / INDEX_* / FOMC
  universes.py   fixed ETF lists from the sheet, Sector SPDRs (+inception), SIC->sector map
  survivorship.py  delisting detection, membership coverage, universe_survivorship_bias()
  qa.py          data QA (bad ticks, OHLC, gaps, zero volume, split cross-check, stale, adj-close check)
  fetch.py       CLI: python -m atlasbt.data.fetch --set core|daily|bulk|quick  /  --only <task>
  providers/     yahoo, cboe, fred, french, dolthub (SQL API), dolthub_bulk (CSV mirrors), sec, fomc,
                 sp500, ziplime, hod, iex, dukascopy, nasdaqtrader, borrow,
                 key-gated: alpaca, polygon, tiingo, databento, alphavantage (+ _keyed helpers)
```

Store kinds: `bars_1d`, `bars_1m`, `bars_5m`, `bars_60m`, `quotes`, `trades`, `actions`, `events`,
`option_chain`, `fundamentals`, `rates`, `factors`, `reference`, `borrow`. Logs and per-task JSON
summaries: `cache/logs/`. QA reports: `cache/qa/qa_<freq>_<date>.csv`.

## 2. Point-in-time conventions (all sources)

* All timestamps tz-aware UTC; session dates naive ET midnight (NYSE calendar from
  `exchange_calendars` XNYS, instantiated from 1960 in `frames.calendar()`).
* **Daily bars**: index `start` = session open (UTC), `known_at` = actual session close (16:00 ET,
  13:00 ET on early closes). Verified by tests on 2019-07-03, 2020-11-27, DST transitions.
  NB: XNYS models 09:30-16:00 for all history; pre-1985 open/close times differed (10:00 open,
  15:30 close before 1974) -- irrelevant for daily bars but noted.
* **Intraday bars**: `start` = bar start, `known_at = start + bar length`; `session` in
  PRE (04:00-open) / RTH / POST (close-20:00). Coarser bars resampled from 1m are anchored at the
  open for RTH, never straddle sessions, and a bucket is known only at its end. **RTH bars are
  capped at the actual session close** (`frames.cap_known_at`): a 60m bar starting 12:30 ET on a
  13:00 half day is known at 13:00, not 13:30 (red-team #11; applied to every stored 1m/5m/60m
  frame, migration re-run 2026-10-29).
* **Prices are RAW (as traded)** in the store. `bars(..., adjusted=True)` back-adjusts using only
  actions with `ex_date <= session(as_of)` AND `known_at <= as_of`: splits (price / r, volume x r)
  and cash dividends (price x (1 - D / raw close before ex), CRSP / Yahoo convention, i.e. dividend
  reinvested at the ex-adjusted price). An ex-date after as_of never changes history (tested by
  perturbation). Our adjusted SPY/AAPL/... returns reproduce Yahoo's Adj Close returns to
  max |diff| <= 3.3e-5 per day over full histories of 70 core symbols (`qa.adjclose_crosscheck`).
* **Corporate actions are classified** (`actions_classify.py`, red-team #9). Yahoo encodes
  spin-offs as fractional "splits" whose ratio is the price-adjustment factor (HPQ 2015-11-02
  2.202 = HPE, ABT 2013-01-02 2.0842 = ABBV, EBAY 2015-07-20 2.376 = PYPL, DRI 2015-11-10 1.1187 =
  FCPT). A table of 34 known spin-offs plus a heuristic (ratio not a common split k:1, 3:2, 5:4 ...
  and not a round stock dividend 1.01-1.25) turns them into `SPINOFF` actions: `amount` = value
  distributed per parent share = P_prev x (1 - 1/factor), `new_symbol` / `ratio` (child shares per
  parent share) when known, `adj_factor` = Yahoo's factor, `classify_note` logs why. Adjustment
  treats SPINOFF like a distribution of `amount` (so adjusted prices still match Yahoo's
  adjusted close); the ledger must NOT multiply shares. Non-standard ratios < 1 (reverse split
  combined with a distribution, e.g. MSI 2011 0.2474) stay SPLIT with fidelity T3 and a note.
  Cash dividends > 10% of the previous raw close become `SPECIAL_DIVIDEND` (excluded from
  dividend-yield estimates, still adjusted).
* **Corporate actions** `known_at`: Yahoo has no declaration dates, so known_at = close of the
  session before ex_date (the latest moment the market surely knew; real declarations are 2-6
  weeks earlier). Fidelity T2_PROXY (timing approximate; ex-dates/amounts are real).
  `pay_date` is unknown (None) for Yahoo actions -> ENGINE must assume a lag (typ. 2-4 weeks).
  `CorporateAction.ex_date` is returned as a naive ET date (ARCHITECTURE §2 convention).
* **EARNINGS actuals -- GAAP vs street**: `eps_actual_gaap` = ZipLime GAAP diluted EPS (XBRL),
  only when the figures cover the quarter (`figure_quarters` == 1; Q4 releases usually carry
  full-year figures, e.g. JPM 2024-01-12 FY 16.23, which are NOT used as a quarter EPS);
  `eps_actual_street` = DoltHub eps_history `reported` (adjusted / street basis, 2016+);
  `eps_actual` = street when available else GAAP, `eps_actual_basis` says which.
  **Surprise must be street vs street**: `eps_actual_street - eps_estimate` (all consensus values are
  street basis). GAAP vs consensus mixes bases (one-offs, amortization) and is wrong.
* **EARNINGS estimates**: preferred = DoltHub weekly consensus snapshot ('Current Quarter', same
  fiscal period end +-7 d) dated before the release, estimate_known_at = snapshot date 23:59 ET,
  payload `estimate_fidelity` 1 ("weekly consensus snapshot dated before the release (PIT)"),
  available 2017-10 ->; fallback = DoltHub eps_history final consensus, stamped release - 1 min but
  flagged `estimate_fidelity` 2 / `estimate_reason` "consensus recorded post hoc; revisions not PIT".
  Release time = ZipLime SEC acceptance timestamp (2003-2026); BMO/AMC bucket times only for
  calendar rows ZipLime lacks.
* **Events**: `known_at` is the release time (see §4); `Portal.events(kind, symbols, start, end, as_of)`
  returns events with `start <= known_at <= end` and `known_at <= as_of`; an EARNINGS estimate
  whose `estimate_known_at > as_of` is blanked.
* **Option chains**: snapshot `ts` <= as_of (see §4.6 for the snapshot-time convention).
* **Rates**: FRED daily value for date d is known at 16:30 ET of the next **Federal Reserve
  business day** (H.15 is published the next business day ~16:15 ET). Fed holidays differ from
  NYSE holidays (Columbus Day, Veterans Day are Fed holidays; Good Friday is not); a holiday
  falling on Sunday is observed Monday, on Saturday it is not observed (`fred.fed_holidays`,
  `next_fed_business_day`; red-team #8). E.g. DTB3 for Fri 2023-10-06 is known Tue 2023-10-10
  16:30 ET (Mon 10-09 Columbus Day), DTB3 for Thu 2023-04-06 is known Fri 04-07 (Good Friday, Fed
  open, NYSE closed).
* **Fundamentals**: known_at = SEC acceptance datetime of the filing that reported the fact.
  Facts whose `period_end` lies more than 7 days after their `known_at` (mis-tagged XBRL contexts,
  e.g. a forward-dated period in an old filing) are ignored by `fundamentals()` /
  `fundamentals_history()` (red-team #12).

## 3. Fidelity tiers used by the data layer

| Data | Tier | Why |
|---|---|---|
| Yahoo daily / intraday bars, DoltHub daily bars, Cboe indices | T1_BAR | real bars |
| Dukascopy 1m bars & ticks (CFD quotes) | T2_PROXY | dealer CFD quotes, not exchange prints/NBBO; opt-in only |
| IEX TOPS quotes | T2_PROXY | single venue BBO, not NBBO |
| Alpaca SIP quotes (if key) | T0_QUOTE | consolidated quotes (UNVERIFIED) |
| Corporate actions (Yahoo) | T2_PROXY | known_at approximated |
| DoltHub dividends for symbols Yahoo lacks | T2 (2014+) / T3 (pre-2014) | split-adjusted amounts un-adjusted with a split table that starts 2014 |
| Option chains (DoltHub, historicaloptiondata sample) | T2_PROXY | real quotes, capture minute unknown, sparse strikes |
| EARNINGS from ZipLime (exact 8-K acceptance) | T1 | to-the-second release time |
| EARNINGS from DoltHub calendar only | T2 | clock time approximated (BMO 08:00, AMC 16:30 ET) |
| INDEX_ADD/DEL | T2 | announcement time-of-day approximate (17:15 ET) or fallback |
| FOMC | T1 if time printed on page, else T2 | |
| Borrow | T2 | default model (no free borrow data reachable) |
| French / AQR factors | T2 | revised, not PIT -- validation only |

## 4. Sources

Rate limits are enforced in `config.HOST_MIN_INTERVAL` (+ exponential backoff, Retry-After honoured).
User-Agent: `atlasbt-research/0.1` (override `ATLASBT_USER_AGENT`); **no personal data in code**.

### 4.1 Yahoo Finance (`providers/yahoo.py`) -- daily & intraday OHLCV, splits, dividends
* Access: the `yfinance` library (1.7.0) as approved; it manages its own HTTP session (Yahoo's raw
  chart endpoint answers 429 to plain clients). We throttle >= 1 s between calls and back off on
  rate-limit errors.
* Daily: `history(period='max', auto_adjust=False, actions=True, repair=False)`. **Yahoo's
  "unadjusted" OHLC are split-adjusted to today** (AAPL 2020-08-28 Close 124.81 = 499.23/4), volume
  too, dividends too. We rebuild as-traded prices with Yahoo's own split list
  (`adjust.unadjust_yahoo`). Known answers tested: AAPL 2020-08-28 499.23, 2020-08-31 129.04,
  2014-06-06 645.57, 2014-06-09 93.70, volume 2020-08-28 46,907,479; TSLA 2020-08-28 2213.40,
  2022-08-24 891.29; AAPL dividend ex 2012-08-09 $2.65.
  Capital-gains distributions (funds) are stored as SPECIAL_DIVIDEND.
* Intraday: 1m (last ~30 days, 7-day windows), 5m (60 days), 60m (730 days), `prepost=True`.
  The store accumulates: each run appends only new bars (re-reading 1 overlapping day), so running
  `--set daily` every day builds a growing 1m archive from 2026-09-30 onward.
* Coverage obtained: see §5. License: Yahoo terms (personal/research use; do not redistribute).

### 4.2 DoltHub `post-no-preference` (SQL API `providers/dolthub.py`, CSV mirrors `providers/dolthub_bulk.py`)
* SQL API `https://www.dolthub.com/api/v1alpha1/post-no-preference/<db>/master?q=`: max 1000 rows
  per response (status `RowLimit`), ~55 s server deadline -> aggregates and any scan over many dates
  fail ("context deadline exceeded"); exact-key lookups `date=... AND act_symbol=...` take 3-8 s
  (~28 s when cold; 35-55 s under load on 2026-10-28); IN-lists force a whole-date scan (~40 s). Responses are cached only if
  `Success`. Used now only for option snapshots newer than the mirror and for small per-symbol
  tables.
* Bulk: Hugging Face CSV dumps of the same tables by `siddharthmb` (updated 2026-01-29, no license
  stated): `stocks-options/option_chain.csv` (9.13 GB, 99,575,024 rows, 2019-02-09 .. 2026-01-27),
  `stocks-ohlcv/ohlcv.csv` (1.21 GB, 26.7 M rows, 2011-01-03 ..), `stocks-earnings-eps_estimate`
  (439 MB), `stocks-symbols`. **Verified identical** to the SQL API: SPY 2024-01-03, all 104 rows,
  bid/ask/IV equal. Downloaded to the scratchpad, filtered, stored, then deletable.
* `stocks.ohlcv` appears raw (as traded) and includes since-delisted symbols -> `bars_1d/dolthub`
  for the S&P 500 PIT union (2011+), our main survivorship patch.
* `stocks.dividend` amounts are **split-adjusted to the latest basis** (AAPL 1987 $0.12 stored as
  0.00214) and `stocks.split` starts only in 2014 -> used only as a fallback for symbols Yahoo
  lacks, flagged T2/T3.
* `earnings.*`: earnings_calendar (2020-01 .. 2026-12, `when` = BMO/AMC/NULL), eps_history
  (reported vs final estimate, 2016+), eps_estimate (weekly consensus snapshots 2017-10-26 ..
  2026-01-25; snapshot dated d treated as known 23:59 ET on d).
* License: community data, no explicit license on DoltHub; mirror has none either -> research use,
  do not redistribute.

### 4.3 ZipLime earnings calendar (`providers/ziplime.py`, local, Apache-2.0)
* `data_raw/hf_ziplime_earnings_calendar` (HF `ZipLime/earnings-calendar` rev f8d9be6).
  8-K Item 2.02 releases with acceptance time read from each filing's SGML header (ET, exact),
  session, first tradeable session, as-of ticker resolution (includes delisted names), figures.
* Stored as `events/ziplime/EARNINGS/<year>.parquet` (431,241 releases with a ticker, 12,657
  symbols, 2003-04-25 .. 2026-10-23; timing AMC 219,074 / BMO 131,737 / DURING 79,877 / UNKNOWN 553;
  197,018 with EPS).
* **Timestamp check requested by the integrator** ("submissions acceptanceDateTime 'Z' is often
  ET"): across 54,909 8-K 2.02 filings of 776 companies (2004-2026) matched by accession number,
  the per-company `data.sec.gov` submissions JSON `acceptanceDateTime` equals ZipLime's SGML-header
  time exactly (0 s difference in 100% of cases, every year), i.e. it IS true UTC. The ZipLime caveat applies to EDGAR's *bulk* `submissions.zip`, which we
  do not use. We still prefer ZipLime times for earnings.

### 4.4 SEC EDGAR (`providers/sec.py`)
* `data.sec.gov` accepts the plain UA `atlasbt-research/0.1` (verified 200). `www.sec.gov`
  (company_tickers_exchange.json, Archives) answers 403 without a contact in the UA -> set
  `SEC_USER_AGENT="atlasbt-research/0.1 you@example.org"` to enable; otherwise we fall back to a
  stale cached ticker map, then to CIKs from Wikipedia (current S&P members) and ZipLime's as-of
  tickers (historical names). Rate: <= ~6.6 req/s (limit 10).
* submissions JSON (+ older `files` pages) -> security master (SIC, names, former names, exchanges,
  fiscal year end) and 8-K Item 2.02 events (`EARNINGS_RELEASE`, known_at = acceptance; an upper
  bound on the press-release time).
* companyfacts -> fundamentals long frame (`symbol, field, value, period_end, known_at, source,
  period_start, duration_days, fy, fp, form, accn, unit, concept, filed, known_at_method`).
  Fields (`FIELD_MAP`): revenue, net_income, eps_diluted/basic, operating_income, gross_profit,
  cost_of_revenue, pretax_income, income_tax, interest_expense, rnd, sga, depreciation,
  operating_cash_flow, capex, buyback, dividends_paid, dividends_per_share, weighted_shares_diluted
  (duration fields suffixed `_q` 80-100 d / `_fy` 350-380 d / `_ytd`), and instant fields
  total_assets, total_liabilities, current_assets/liabilities, stockholders_equity, cash,
  long_term_debt, inventory, receivables, ppe_net, retained_earnings, shares_outstanding.
  Book-equity / factor fields (fallback order = list order; the first concept a filing reports for
  a period wins, the concept used is kept in `concept`):
  `preferred_stock` = PreferredStockValue > PreferredStockCarryingAmount >
  PreferredStockValueOutstanding > PreferredStockIncludingAdditionalPaidInCapital;
  `deferred_taxes` = DeferredTaxLiabilitiesNoncurrent > DeferredIncomeTaxLiabilitiesNet >
  DeferredTaxLiabilities > DeferredIncomeTaxLiabilities; `short_term_debt` = DebtCurrent >
  ShortTermBorrowings > LongTermDebtCurrent > LongTermDebtAndCapitalLeaseObligationsCurrent;
  `stock_issuance_proceeds` = ProceedsFromIssuanceOfCommonStock > ProceedsFromStockOptionsExercised;
  `repurchase` = PaymentsForRepurchaseOfCommonStock > PaymentsForRepurchaseOfEquity (`buyback`
  is kept as the original name; like all flow fields these carry the `_q` / `_fy` / `_ytd` suffix,
  e.g. `repurchase_fy`, `stock_issuance_proceeds_ytd`). Fama-French book equity: stockholders_equity + deferred_taxes -
  preferred_stock; a missing preferred/deferred field means "not reported" (usually zero) -- the
  strategy decides the default. Post-ASU 2015-17 most filers report only a net deferred-tax
  figure, so deferred_taxes is sparse after 2016.
  known_at = acceptance of the reporting filing (note EDGAR dates filings accepted after 17:30 ET
  as the next day: AAPL 10-K accepted 2022-10-27 18:01 ET has filed=2022-10-28); fallback when the
  accession is not in the index: filed date 17:30 ET (never early). Restatements: all (period,
  filing) pairs kept; `fundamentals()` returns the latest period known at as_of and, for it, the
  latest filed value. Q4 quarterly values are not filed separately (only FY) -- not derived.
  XBRL facts exist from ~2009 (large filers) / 2011 (all).
* License: public domain.

### 4.5 S&P 500 PIT membership (`providers/sp500.py`)
* fja05680/sp500 "Historical Components & Changes (Updated).csv" (MIT; 1996-01-02 .. 2026-08-18),
  extended to today with Wikipedia "Historical components of the S&P 500" (raw wikitext; 412 change
  rows 1976 .. 2026-10-06).
* Membership interval [start, end): member for sessions start <= d < end.
* INDEX_ADD/INDEX_DEL events: effective_at = effective session open. **Announcement dates** are
  parsed from the S&P press-release citations in the Wikipedia table (`|date=` or a date in the
  URL), validated to lie within 75 days before the effective date: 623 of 1,544 events
  (`announce_source='sp_press_release_citation'`), known_at = announcement date 17:15 ET (S&P
  publishes after the close; time-of-day approximate). Others: `fallback_prev_session`
  = 17:15 ET of the session before the effective date (conservative; real announcements are
  usually 2-10 days earlier, so announcement-drift strategies are only testable on the 623).
  All T2_PROXY.
* Current constituents with GICS sector/sub-industry and CIK from Wikipedia's list (current only).

### 4.6 Option chains
* **DoltHub `options.option_chain`** (EOD, 2019-02-09 .. 2026-01-27 from the mirror; later dates via
  SQL): bid, ask, IV, delta, gamma, theta, vega, rho -- **no volume, no open interest, no underlying
  price** (we attach the Yahoo raw close of the snapshot session as `underlying_price`). Weekly
  (Saturday-dated) snapshots in 2019, ~3 per week 2020-2024 (SPY: 145-175 per year), ~daily in 2025.
  Coverage is SPARSE: typically 3-4 expirations (nearest ~1-2 weeks to ~2 months) and ~20-60 strikes
  per expiry. Present: SPY, DIA, the 11 Sector SPDRs, KRE, XBI, XOP and most S&P 500 stocks.
  **Absent: QQQ, IWM, SPX/SPXW, NDX, RUT, GLD, TLT, EEM, EFA, HYG, LQD, SMH.**
* **Snapshot timestamp**: rows dated d carry quotes from about the close of the last session <= d.
  Evidence (put-call parity implied spot minus carry vs SPY daily bars): 2024-01-03 468.8 vs close
  468.79; 2025-05-07 560.9 vs 561.15; 2026-10-27 769.9 vs 769.66; Saturday 2019-02-09 rows match
  Friday's close (270.5 vs 270.47); on volatile days the implied spot deviates up to ~0.5% from the
  closing print (2022-06-15 FOMC day: 377.4 vs 379.20), i.e. capture is near but not exactly at the
  close. Convention: `ts` = official close (UTC) of the last NYSE session <= d (16:00 ET, 13:00 on
  half days). The exact capture minute is unknown -> **T2_PROXY**. `option_quotes()` returns these
  snapshots; a fill model taking "the first snapshot after eligible_at" therefore fills at the next
  close's quotes.
* **historicaloptiondata.com free sample** (`providers/hod.py`): every listed option on
  2019-08-15 incl. SPX (AM, 3rd Friday) and SPXW (PM) -- SPXW is filed under underlying SPX --,
  QQQ, IWM, with volume and open interest. ts = 2019-08-15 close, T2. One day only: a reference /
  cross-check, not a backtest source.
* **Annotation** (`option_meta.annotate_chain`, applied when the Portal serves real chains):
  `expiry` = LAST TRADING DAY (holiday Fridays -> previous session; AM-settled index monthlies
  such as root SPX are listed by vendors on the settlement Friday -> expiry = Thursday,
  `settlement_date` = Friday); extra columns `root` (SPX vs SPXW), `settlement_date`, `kind`
  (MONTHLY / WEEKLY) and `underlying_price_source` (`vendor` for historicaloptiondata,
  `pit_bar_close_at_ts` for DoltHub: the raw Yahoo close of the snapshot session, known exactly at
  ts). The `symbol` column keeps the vendor OCC symbol (its date is the settlement date for SPX
  AM); `Portal.instrument(occ)` returns expiry = last trading day, and quote lookups match by
  symbol or by (root, expiry, strike, right), so `Instrument.option(...)` built from the
  annotated chain also matches.
* **Synthetic fallback**: `option_chain(u, as_of, ..., synthetic=True)` returns the real snapshot
  when one exists no older than 4 days; otherwise `Portal.synthetic_chain()` (built with
  `atlasbt.options.synthetic.SyntheticChainGenerator` from PIT inputs: last raw price, 3M T-bill
  money-market rate converted to continuous, trailing-12m dividend yield, 63-day realized-vol ratio
  vs SPY for names without a Cboe vol index). Synthetic frames have fidelity column 3
  (T3_MODELED) and `attrs['synthetic'] = True`; real frames `attrs['synthetic'] = False`, fidelity
  2. The two are never mixed in one frame. `option_quote()` returns REAL quotes only.
  `option_quotes(instrument, start, end, as_of, synthetic=True)`: real snapshots when any exist in
  the window; otherwise (and only if `synthetic`) the OPTIONS owner's
  `atlasbt.options.marking.synthetic_option_quotes(..., real_first=False)` supplies one row per
  session close -- anchored to a same-day real surface when one exists (T2_PROXY) else pure
  model (T3_MODELED), with `source` / `reason` columns and `attrs['synthetic'] = True`. A frame is
  either all real or all synthetic. `synthetic=False` = real only.
* `has(sym|'*', 'options', freq)` / `fidelity_of`: freq `None`/`'1d'` = EOD chains (kind
  `option_chain`); intraday freqs (`'1m'` ...) = kinds `option_bars_<freq>` / `option_quotes`
  (e.g. Databento OPRA when a key is configured -- none is stored today, so they return False /
  NA). `fidelity_of` is the best tier among the sources actually present
  (`portal.OPTION_SOURCE_FIDELITY`): exchange-timestamped quotes (databento / polygon / alpaca) T0,
  EOD captured at a documented time (cboe_eod) T1, EOD with approximate capture time (dolthub,
  hod) T2; with no real source, EOD falls back to the synthetic chain -> T3 when the underlying has
  daily bars; intraday -> NA. `has()` stays real-data only.
* Quote quality: SPY quotes are tight (2024-01-03 ATM 4.46/4.48), but single-stock DoltHub spreads
  look wider than the NBBO (AAPL near-ATM median relative spread 4.2% in 2019, 2.5-4.8% by year)
  and the 2019 Saturday snapshots may reflect after-hours quotes -> fills at DoltHub bid/ask are
  conservative for single names.
* No free intraday option quotes exist (Lean fixtures only). QQQ/IWM/SPX chains are therefore
  synthetic (T3) except on the single historicaloptiondata day.

### 4.7 Cboe indices (`providers/cboe.py`)
`https://cdn.cboe.com/api/global/us_indices/daily_prices/<SYM>_History.csv` -- 43 series (VIX family,
SKEW, VVIX, strategy indices PUT/BXM/BXMD/BXMW/CNDR/BFLY/CLL/CMBO/PPUT/PUTY/RXM/BXY, COR1M/3M,
DSPX, SPX/OEX/XEO/DJX/RUT, single-stock vols). Single-value files are stored as O=H=L=C.
known_at = session close + 15 min (vol indices are disseminated until 16:15 ET). Values on days
without an NYSE session are dropped (since 2022 Cboe publishes VIX on US holidays during global
trading hours: 34 such VIX dates). Cboe site terms.

### 4.8 FRED (`providers/fred.py`, `rates.py`)
22 series (DTB3/DTB4WK/DTB6/DTB1YR discount basis, DGS1MO..DGS30 CMT, DFF, SOFR, DFEDTAR(U),
T10Y2Y, T10Y3M, BAML OAS (last 3 y only), DFII10, CPIAUCSL). FRED stopped answering the plain
`atlasbt-research/0.1` UA (read timeouts) after our first pull; the provider falls back to the
stale raw cache and then to the scout's identical `data_raw/fred/<ID>.csv` -- we do not spoof UAs.
**Conversion** (`risk_free_rate`): the engine accrues ACT/360, so we return the simple
money-market yield `r = d / (1 - d*t/360)` from the discount rate d (t = 28/91/182/364 days for
1M/3M/6M/1Y); DGS* (bond-equivalent) are rebased x360/365; DFF/SOFR returned as is.
Known answer (tested): d = 5.00%, t = 91 -> price 0.987361, MMY 5.0640%, BEY 5.1343%, continuous
5.1018%. Tenor 1M falls back to 3M before 2001-07-31. FRED shows latest vintages (T-bill yields are
not revised; CPI is -- not PIT).

### 4.9 Ken French (`providers/french.py`) -- validation only
11 datasets (FF3/FF5 daily & monthly, momentum, ST/LT reversal, 10/12/49 industries daily,
momentum deciles). Stored as decimals. NOT PIT (rebuilt yearly; RF source changed to ICE BofA 1M
bill in 2024-06). AQR / q5 factors are on disk in `data_raw/` (scout) but not loaded.

### 4.10 FOMC statements (`providers/fomc.py`)
Statement links scraped from `fomccalendars.htm` + `fomchistorical<YEAR>.htm` (1994-2020), each
page fetched. 246 statements 1994-02-04 .. 2026-09-16. Release time: printed on newer pages
("For release at 2:00 p.m.", 91 statements, T1); **1994 announcements from the
Gurkaynak-Sack-Swanson (2005) intraday event-study table** (02-04 11:05, 03-22 14:20, 04-18 10:06
unscheduled, 05-17 14:26, 08-16 13:18, 11-15 14:20 ET; `time_source='gss_2005_table'`, T2: the times
are the first wire report, not an official release stamp; red-team #14); hand-verified unscheduled
actions and the 2011-2012 press-conference meetings (12:30 ET) (15, T2); otherwise era rule 14:15 ET for 1994 ..
Jan-2013 and 14:00 ET from March 2013 (T2; when unsure we use the LATER plausible time, which can
only delay a strategy, never leak). `fomchistorical` pages redirect to http://, which our client
upgrades to https.
**`FOMC_SCHEDULE`** event kind: the scheduled statements as known in advance (effective_at =
statement time, known_at = statement date - 30 days 00:00 ET, T2_PROXY: the Fed publishes its
calendar ~a year ahead but we have no publication date per calendar, so 30 days is a conservative
lower bound). Use it for pre-FOMC drift strategies; `FOMC` (known at release) for the reaction.

### 4.11 Dukascopy (`providers/dukascopy.py`) -- CFD quotes, opt-in proxy
Hourly tick files and daily BID/ASK 1m candles (`.bi5`, LZMA, month index 0-based, price/1000).
US stock/ETF CFDs (`SPYUSUSD`, ...) are quoted during RTH only, since ~2017 (SPY 2017-02-16);
index CFDs (`USA500IDXUSD`) ~23 h since 2012. Spot check 2026-10-26/27: SPY CFD mid vs Yahoo 1m close
median diff 0.00, sd 0.012 USD, but the CFD spread median is 3.6 cents vs ~1 cent NBBO -> **use
for price paths, never for spread/size modelling**. Volume is Dukascopy's own (stored as NaN).
`Portal(allow_proxy_intraday=True)` or `source='dukascopy'` is required to serve these bars
(T2_PROXY). Server throttles hard (503/timeouts): 1 req/s; backfill is slow (see §5).

### 4.12 IEX HIST TOPS (`providers/iex.py`)
pcapng (gzip) IEX-TP v1 parser for QuoteUpdate/TradeReport; streams files, filters symbols.
Loaded: the scout's 2019-12-24 file (SPY 112,316 quote updates, 3,212 trades; QQQ 60,332/657;
AAPL 661/1,467). Free daily files since 2016-12-12 (index `https://iextrading.com/api/1.0/hist`)
but 10-16 GB/day recently -> fetch targeted days only. IEX-only BBO (T2_PROXY).

### 4.13 Nasdaq Trader symbol directory (`providers/nasdaqtrader.py`)
nasdaqlisted.txt + otherlisted.txt: 13,249 non-test symbols, 5,772 ETFs (2026-10-28 snapshot).
Current-only; snapshots accumulate per run (`reference/nasdaqtrader/directory`). Used for ETF flags
and as the "still listed" set in delisting detection.

### 4.14 Borrow (`providers/borrow.py`)
iborrowdesk.com probed once (2026-10-28): connection dropped by the remote (no data). FINRA short
data is excluded by its terms. Default model (T2_PROXY, env-overridable): ETFs & S&P 500 members
0.30% p.a., other stocks 1.0%, leveraged/inverse ETFs 2.0%; always shortable.

### 4.15 Key-gated providers -- UNVERIFIED (written from official docs, never run with a key)
| Provider | Env | What | Notes |
|---|---|---|---|
| `alpaca.py` | ALPACA_API_KEY, ALPACA_API_SECRET | SIP 1m bars (`/v2/stocks/{s}/bars`, adjustment=raw, feed=sip), quotes (`/quotes`) since 2016 | quote sizes x100 (round lots) |
| `polygon.py` | POLYGON_API_KEY | aggregates `/v2/aggs/ticker/...` | **api.polygon.io is blocked (403) by this environment's egress proxy** |
| `tiingo.py` | TIINGO_API_KEY | EOD raw + divCash/splitFactor (incl. delisted tickers), IEX intraday | IEX intraday = single venue (T2) |
| `databento.py` | DATABENTO_API_KEY | `timeseries.get_range` csv (ohlcv-1m, bbo-1m), cost check first, refuses > DATABENTO_MAX_COST (default $1) | XNAS.ITCH = Nasdaq venue only |
| `alphavantage.py` | ALPHAVANTAGE_API_KEY | TIME_SERIES_INTRADAY month=YYYY-MM, extended hours, adjusted=false | 25 req/day free |
Without keys `available()` is False and fetch functions return `{'skipped': ...}` (tested).

### 4.16 Other on-disk scout data not integrated (available for cross-checks)
qlib_us_1d (8,994 symbols 1999-2020, total-adjusted, Yahoo-derived, only 178 dead names),
Lean fixtures (SPY minute quotes Oct-2013, SPX option minutes), NY Fed rates, Treasury curves, AQR,
q5, Shiller, FinanceDatabase. Not used: PCRA CRSP/S&P files (license), FINRA short data (terms).

## 5. Coverage actually in the store (2026-10-28)

Store total **1.8 GB** (`cache/store`) + raw HTTP cache (`cache/raw`, gzip'd); HF CSV mirrors were
kept outside the cache (scratch) and can be deleted after loading.

| Kind / source | Symbols | Rows | Size | Range / notes |
|---|---|---|---|---|
| bars_1d / yahoo | 797 | 6.04 M | 249 MB | 70 core ETFs + megacaps (SPY 1993-01-29.., QQQ 1999-03-10, IWM 2000-05-26, sector SPDRs 1998-12-22 / XLRE 2015-10-08 / XLC 2018-06-19, SGOV 2020-06-01, TQQQ 2010-02-11, AAPL 1980-12-12) + S&P 500 PIT union: 1,199 tickers tried, 763 served by Yahoo, **436 missing** (delisted/renamed) -- all to 2026-10-27 |
| actions / yahoo | 710 | 68,992 | 7 MB | splits, cash dividends, capital-gain distributions |
| bars_1d / dolthub | 993 | 2.93 M | 113 MB | as-traded; first year 2011 for 673 symbols, **2018 for 117**, later for the rest; ends 2026-01-27 (mirror) |
| bars_1m / yahoo | 51 | 598 k | 16 MB | SPY QQQ IWM DIA TLT GLD, 11 sector SPDRs, 34 megacaps; 2026-09-30 .. 2026-10-27 (20 sessions), PRE/RTH/POST |
| bars_5m / yahoo | 51 | 316 k | 10 MB | 2026-08-31 .. 2026-10-27 |
| bars_60m / yahoo | 51 | 417 k | 14 MB | 2024-10-30 .. 2026-10-27 |
| bars_1m / dukascopy | SPY (backfill running) | 237 k+ | 7 MB+ | CFD mid quotes 2017-02-16 -> (progressing ~1 session / 2.5 s); QQQ, IWM, DIA queued |
| bars_1d / cboe | 43 | 255 k | 8 MB | VIX 1990.., VIX9D 2011, VIX3M 2009, VIX6M 2008, VIX1Y 2007, VIX1D 2022-05-13, SKEW 1990, VVIX 2006, PUT 1991, BXM 2002, CNDR/BFLY 1986, SPX 1975 ... |
| option_chain / dolthub | 799 | 59.1 M | 1.28 GB | 2019-02-08 .. 2026-01-27 (+ SQL increment for SPY AAPL MSFT DIA XLF XLE XLK NVDA AMZN TSLA running). SPY: 1,073 snapshots (2019: 47 weekly; 2020-2024: 145-175 / yr ~ 3 per week; 2025: 248 ~ daily), 3.1 expiries and 134 contracts per snapshot, DTE range per snapshot min 11-18 / max 43-66 days |
| option_chain / hod | 28 | 75,909 | 3 MB | 2019-08-15 only, full chains incl. SPX/SPXW, QQQ, IWM, volume & OI |
| quotes / iex, trades / iex | 3 | 173 k / 4.9 k | 2 MB | 2019-12-24 TOPS (SPY, QQQ, AAPL) |
| quotes / dukascopy | 1 | 76 k | 1 MB | SPY CFD ticks 2026-10-27 (sample) |
| events / ziplime | 12,657 symbols | 430,908 | 18 MB | EARNINGS 2003-04-25 .. 2026-10-23 |
| events / sec | 1,018 | 71,757 | 11 MB | EARNINGS_RELEASE (8-K 2.02) per symbol |
| events / sp500, fomc | | 1,544 / 246 | | INDEX_ADD/DEL 1996-2026; FOMC 1994-02-04 .. 2026-09-16 |
| fundamentals / sec | 979 | 3.51 M | 35 MB | XBRL facts 2009+, 1,053 of 1,199 S&P union tickers mapped to a CIK |
| reference / sec security_master | 1,053 | | | SIC for 1,048 |
| reference / dolthub | | 2.45 M | 15 MB | earnings_calendar 122 k, eps_history 169 k, eps_estimate (Current Quarter) 1.65 M (6,823 symbols, 2017-10-26 .. 2026-01-25), dividend 502 k, split 4,168 |
| rates / fred | 22 | 236 k | 4 MB | DTB3 1954-01-04 .. 2026-10-26 etc. |
| factors / french | 11 | 228 k | 7 MB | FF3 daily 1926-07-01 .. 2026-09-30 etc. |
| reference / nasdaqtrader | 13,249 | | | 5,772 ETFs (snapshot 2026-10-28) |

## 6. DataPortal (`atlasbt.data.portal.Portal`)

* `bars(symbol, freq, start, end, as_of, adjusted=False, sessions=('RTH',), source=None)`:
  freq `1d`, `1m`, `5m`, `60m` native; other minute multiples resampled causally from 1m. Source
  priority: daily yahoo > dolthub > tiingo > cboe; intraday yahoo > alpaca > polygon > databento >
  alphavantage > tiingo (dukascopy only if opted in). `attrs`: source, fidelity, adjusted.
  **Daily requests for stocks go through the security-identity layer (§8.1)**: the ticker is
  resolved to an issuer (CIK) as of the request date, and only identity-verified segments are
  served (`attrs['security_id']` = `CIK##########`, `attrs['identity_log']`, `attrs['source']`
  e.g. `yahoo+dolthub`, fidelity T2 when another ticker's or DoltHub's history is used). If the
  identity cannot be established the call raises `DataUnavailableError` ("security identity not
  established ... (fail closed)"). ETFs, indices and Cboe series are served directly.
  `security_id(symbol, as_of)` exposes the resolution.
  Intraday: `sessions=('PRE','RTH','POST')` selects session parts (Yahoo has pre/post). Missing
  native frequencies are resampled causally from 1m: RTH buckets aligned to the 09:30 ET open,
  `known_at` = bucket end, capped at the actual close on half days (tested on 2023-11-24).
* `quotes()` (alpaca > databento > iex > dukascopy).
* `has(symbol, kind, freq=None, sessions=None)`: symbol `'*'` = any data of that kind;
  `has('*', 'events', 'EARNINGS'|'INDEX_ADD'|'INDEX_DEL'|'FOMC'|'EARNINGS_RELEASE'|'EARNINGS_SCHEDULE')`
  True when the source is loaded (BUYBACK, HALT: False -- no source integrated);
  `has('*', 'fundamentals', field)`; `has(sym, 'bars', '1m', sessions=('PRE',))` checks session
  availability. `has(sym, 'bars', '1d')` is identity-aware (False when the call would fail closed).
  `fidelity_of()` mirrors it (NA when absent).
* `corporate_actions()`: the actions of the identity-verified segments actually served (Yahoo
  actions classified as in §2, DoltHub actions for DoltHub segments, split-validated -- see
  §8.1), plus a DELIST action at the explicit identity end; `events()`.
* `option_chain(underlying, as_of, min_dte, max_dte)` (latest snapshot <= as_of; `attrs`
  snapshot_ts, stale_days), `option_quote(instrument, as_of)` (None if older than 4 days),
  `option_quotes(instrument, start, end, as_of)` (all snapshots in (start, end], ts <= as_of).
* `risk_free_rate(as_of, tenor)` tenors 1M 3M 6M 1Y ON SOFR 2Y 5Y 10Y 30Y, and
  `risk_free_rate_info(as_of, tenor) -> (rate, series_used, fidelity, observation_date)`:
  T1 when the tenor's native series was used (1M = DTB4WK from 2001-07-31, 6M = DTB6, 1Y = DTB1YR
  until 2001 then DGS1 CMT), T2_PROXY when a fallback tenor was substituted (1M -> DTB3 before
  2001-07-31) or when every candidate's last print is older than 10 days (freshest value returned).
  `risk_free_rate()` returns the same rate; strategies that care should call the `_info` variant
  and flag T2.
* `fundamentals(symbols, fields, as_of)`: latest period known at as_of (latest filed value).
  `fundamentals_history(symbols, fields, as_of, start=None)`: every filed value (all periods,
  restatements and prior-year comparatives as separate rows) with known_at <= as_of (and
  period_end >= start): symbol, field, value, period_start, period_end, fiscal_period, fy, form,
  known_at, accn, concept, duration_days, source -- for PIT TTM / YoY rebuilding.
* `borrow()`, `security_master(as_of)` (SIC sector, current GICS, sector SPDR if
  it existed at as_of; NOT fully PIT), `universe(name, as_of)`: `SP500` (PIT), `SECTOR_SPDR`,
  `SECTOR_SPDR_9`, `BROAD_INDEX_ETF`, `GEM_DUAL_MOMENTUM`, `ACCEL_DUAL_MOM`, `CROSS_ASSET`,
  `TREASURY_DURATION`, `TBILL_PROXY`, `LEVERAGED`, `EQUAL_WEIGHT_PAIR`, `CORE_ETF`, `MEGACAP`
  (fixed lists only include ETFs with a completed bar by as_of), and `SP500_TOP<N>_DV` (e.g.
  `SP500_TOP50_DV`): PIT S&P 500 members ranked by mean raw close x volume over the 20 sessions
  strictly before the session of as_of (first call ~12 s, then memoized per session).
  **`universe('SP500', as_of)` returns the tickers as they were at as_of** (e.g. `FISV` in 2015,
  `UTX` in 2010; `.` -> `-`), not the fja05680 "Updated" later tickers; `bars()` maps them to
  the issuer.
* `universe_survivorship_bias(name, as_of)` (`survivorship.universe_bias_report`) -> members,
  with_data, coverage, missing, `stale` (last bar 1-3 sessions old), `member_session_coverage`
  (share of member x session pairs over the trailing 20 sessions with a bar), survivorship_bias
  flag, fidelity. **Derived universes** (`SP500_TOP50_DV` ...) report the BASE SP500 coverage
  (`base_universe`, `base_coverage`, `derived_members`): they rank only members with data, so
  they inherit its bias (red-team #5, #13). Reports must print it for SP500 universes.
* Extras: `instrument()` parses OCC symbols (SPX/SPXW European cash-settled AM/PM),
  `rate_series()`, `clear_cache()`. Reads are memoized per Portal instance.

Sector SPDR inception (first trading): XLB XLE XLF XLI XLK XLP XLU XLV XLY 1998-12-22, XLRE
2015-10-08, XLC 2018-06-19. SIC->sector is a coarse documented approximation (`universes._SIC_RULES`).

## 7. Data QA (`atlasbt/data/qa.py`)

Checks (per symbol, `qa_report(portal, symbols, freq)` -> `cache/qa/qa_<freq>_<date>.csv`):
OHLC consistency (high >= max(open,close), low <= min(open,close), positive prices); bad ticks
(`bad_ticks(close, k=6, min_move=0.08, revert=0.25, market=SPY, market_move=0.03)`: |log return|
> max(6 x rolling median |r|, 8%), at least 75% reversed on the next bar, and not on a day SPY moved
> 3% -- the same detector the identity layer uses to reject collided segments, red-team #10); calendar gaps
(NYSE sessions between first and last bar without a bar); zero-volume days/bars; stale runs
(>= 5 identical closes); split cross-check (overnight gaps between CONSECUTIVE sessions whose ratio
matches a common split ratio but has no recorded split, and recorded splits without a jump);
adjusted-close cross-check vs Yahoo Adj Close; intraday RTH completeness vs the calendar.

Results 2026-10-28, 70 core symbols (39 ETFs + 31 megacaps), daily (bad-tick counts below were
computed with the earlier 10x / 25 % detector):
* OHLC violations: 0 for all 70. Bad-tick flags: 7 symbols, 1-2 each (UPRO, TQQQ, UNH, XOM, PG, KO;
  all real crash/rebound days on inspection, kept). Zero-volume days: WMT 53, JNJ 11, BAC 6, a few
  others 1-2 (1970s-80s Yahoo rows).
* Calendar gaps: 68 for the 1962-start names (XOM PG JNJ KO CVX MRK) -- the 1968 "paperwork crisis"
  Wednesday closures are not in the XNYS calendar model, plus early Yahoo holes; 0 for every ETF.
* Stale runs: HD 28, BAC 34, UNH 14, PEP 14 (1980s rows with repeated closes); 0-1 elsewhere.
* Split cross-check: every recorded split shows the expected gap; 2 unrecorded-split suspects, both
  genuine crashes (AAPL 2000-09-29 -52%, NFLX 2004-10-15 -41%) -> heuristic, review before acting.
* Our PIT total-return adjustment vs Yahoo Adj Close: max daily return difference <= 3.3e-5 (KO),
  typically ~1e-6 -> raw reconstruction and dividend handling agree with Yahoo.
* Intraday 1m (51 symbols x 20 sessions): RTH completeness 99.7-100% (mean 99.97%), 0 OHLC
  violations, 0-6 zero-volume RTH bars per symbol, PRE 148-6,354 and POST 160-3,464 bars per symbol
  (pre/post bars exist only for minutes with trades).

## 8. Survivorship & known gaps (blunt)

### 8.1 Security identity layer (`identity.py`; red-team #1-#4, #10)

Tickers are reused and both price sources mis-attribute history: Yahoo serves TODAY's company
under a ticker (BBT, IR, CPWR, EP, GENZ, SII, CCU ... would get another company's prices) and
DoltHub is not reliably as-traded (VIAC holds CBS 2011-2017, CB interleaves ACE and old Chubb,
FI 2014-2021 is Frank's International). Stock prices are therefore served only for an
**established security identity**; otherwise `bars()` raises `DataUnavailableError` and the
member counts as missing (fail closed).

* **security_id** = `CIK##########` (SEC registrant). Evidence "ticker T denoted CIK c on date d":
  ZipLime's 8-K corpus (every 8-K item + 2.02 announcements, 1,477,443 evidence rows in
  `reference/identity/evidence`). Only `as_of` evidence (ticker observed at filing; **2006+**)
  labels dates (plus the SEC current ticker map as as-of evidence on its snapshot date, for
  tickers that have 8-K evidence): bracketed by evidence of the same CIK <= 548 days apart, or within 120 days of
  the nearest evidence. Carried tickers (a later ticker copied onto old filings, e.g. Clear Channel
  'CCU' carried to 2018 while CCU was Cervecerias Unidas) count only within 120 days, only if the
  S&P member's name matches the CIK's SEC names, and only if the issuer was not renamed in between.
* **Before 2006** (`pre_evidence_cik`): the ticker's earliest evidenced issuer is accepted only if
  the S&P member of that membership interval confirms it -- Wikipedia's CIK of a current
  constituent, the recorded member name (Wikipedia add/remove rows), or a verified successor -- AND
  the issuer's SEC name in force at the date equals its name at its first evidence (no rename:
  SBC Communications was renamed AT&T Inc. and took 'T' in 2005-11, so 'T' in 2004 fails closed;
  Philip Morris -> Altria 2003 'MO' fails closed too -- a known false negative). Served as T2 with an
  `identity_log` note.
* **Issuer succession on the same line** (`is_successor`, `_succession_ok`): holding-company
  reorganizations / re-domiciliations (ORCL 777676->1341439 2006, DUK 30371->1326160 2006, MDT
  64670->1613103 2015, DIS 1001039->1744489 2019) keep the Yahoo line: accepted when the ticker's
  evidence passes from the old to the new CIK with no third issuer, names match, the Yahoo raw
  close is continuous across the hand-over (no unexplained day > 25 %), DoltHub as-traded prices
  of the predecessor agree, and the successor had **no ticker of its own before** (an acquirer
  that adopts the target's ticker carries its own price line: ACE -> CB 2016, SBC -> T 2005 are
  rejected). Successors with an earlier ticker (MRK/Schering-Plough 2009, APA 2021 holdco name
  change) fail closed for the predecessor's dates.
* **Lineage**: the CIK's ZipLime tickers; a share-class ticker without evidence (FOXA, BRK-B) is
  its own lineage; SEC current tickers of the CIK only by price agreement.
* **Segment acceptance**: Yahoo under a lineage ticker whose CURRENT owner (SEC ticker map) is the
  CIK (or its verified successor) and median price > $1; DoltHub rows on dates labelled with the
  CIK; unlabelled DoltHub rows only by price agreement with an accepted segment (median |log
  ratio| <= 1.5 %, 99th pct <= 5 %, >= 20 common sessions) or as renamed history (the run precedes
  the CIK's own use of the ticker, the CIK is evidenced under another ticker meanwhile, the next
  verified segment continues the price within 10 % within 10 sessions: CBS 2011-2017 from DoltHub
  VIAC). **QA feeds identity**: for DoltHub, clustered spike-and-revert days (>= 3 in 20 sessions,
  or more than max(3, 1 %)) reject the segment as two securities interleaved and isolated glitches
  are dropped; for Yahoo the history up to the last cluster is cut (thin, tick-quantized 1970s-90s
  data: COO before 1994-02, NVR 1993-07, MNST 1993-05, FICO 1992-10, ODFL 2000-01, CVNA 2022-01)
  and a cluster within the last 2 years rejects the ticker; isolated Yahoo flags are kept and
  logged. Names are compared on tokens and on squeezed strings ('ExxonMobil' = 'Exxon Mobil'). Cross-source disagreement rejects the lower-priority source.
* **Splice validation**: a source change with an unexplained close-to-close move > 20 % (or > 6
  MADs) drops the lower-priority chunk; single-day fills must lie within 10 % of both neighbours.
* **Actions follow the segments** actually served (Yahoo actions of the Yahoo ticker used, DoltHub
  splits/dividends for DoltHub chunks). Every split is validated against the raw open / previous
  close; unexplained DoltHub jumps become inferred splits only for 2,3,4,5,10 and 1/2..1/20 with a
  matching volume change (FISV 2013-12-17 and 2018-03-20, COG 2012-01-26 and 2013-08-15).
* **Explicit identity end**: the series ends at the last verified bar; DELIST is emitted at the
  next session if the issuer no longer trades.
* Persisted per CIK/lineage in `identity/series` + `identity/actions` (`--only identity`, ~5.5 min,
  3,835 builds; rebuilt on the fly when `BUILD_VERSION` changes). `attrs['identity_log']` lists
  every accept / reject / drop / inference.

**Known cases (regression tests `tests/redteam/data`, `tests/data/test_qa_survivorship.py`)**

| Ticker / date | Outcome |
|---|---|
| BBT 2019-12 | served from Yahoo TFC (same CIK 92230), equals DoltHub as-traded BBT |
| PARA / VIAC / CBS | CIK 813828 lineage; CBS 2011-2017 from DoltHub VIAC rows accepted as renamed history |
| IR 2019, UTX 2019, HRS 2019, SYMC 2018, MMC 2015 | renamed history from Yahoo TT, RTX, LHX, GEN, MRSH |
| COG 2015 / CTRA 2022, FISV 2018 | DoltHub as-traded, inferred splits validated |
| FI 2018 | Frank's International (not Fiserv); `universe('SP500')` returns FISV for Fiserv |
| ORCL 2005, DIS 2012, MDT 2010, DUK 2005 | holdco succession accepted |
| KO / GE / MSFT 1996 | pre-evidence, confirmed by member CIK, T2 |
| XOM 1996 | Exxon Corp (CIK 34088; traded as XON, fja member 'XOM'), Yahoo XOM line = Exxon, T2 |
| XOM 2026, ECHO 2026 | new holdco 2115436 (recent succession) / EchoStar via SEC ticker-map evidence |
| CPWR, EP, GENZ, WEN 2005, AIT 1998, MCIC, CCU 2005, CNC 2000, SII 2005, CB (old Chubb), FOXA 2019-03-12, BK 1998, T 2004, MO 1996 | **fail closed** (no verified segment, name veto, rename, or acquirer-adopted ticker) |

### 8.2 S&P 500 price coverage after fail-closed

`universe_survivorship_bias('SP500', date)` (member has an identity-verified bar on the session or
within the 3 previous sessions; recomputed 2026-10-29):

| date | 1996-06 | 2000-06 | 2005-06 | 2008-06 | 2010-06 | 2012-06 | 2015-06 | 2018-06 | 2020-06 | 2022-06 | 2024-06 | 2026-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| members | 487 | 492 | 496 | 497 | 499 | 497 | 499 | 506 | 505 | 503 | 503 | 503 |
| with verified prices | 136 | 192 | 233 | 287 | 312 | 373 | 407 | 486 | 501 | 497 | 501 | 502 |
| coverage | **27.9 %** | **39.0 %** | **47.0 %** | 57.8 % | 62.5 % | 75.0 % | 81.6 % | 96.1 % | 99.2 % | 98.8 % | 99.6 % | 99.8 % |

(2021-06 99.2 %, 2023-06 99.4 % with 21 members' last bar 1-3 sessions old; member-session coverage
over the trailing 20 sessions equals the point coverage within 0.3 pp everywhere.)

* Coverage FELL versus the previous spliced build (1996 40 %, 2000 48 %, 2005 55 %, 2008 63 %,
  2010 68 %, 2012 78 %, 2015 85 %): those extra names were partly the WRONG company's prices.
  **Cross-sectional S&P 500 backtests before ~2018 are survivorship-biased** (missing names are
  disproportionately acquired or failed) -- reports must print `universe_survivorship_bias` and treat
  pre-2018 cross-sectional results as T2 at best. No free fix exists (CRSP / Norgate are paid).
* DoltHub `stocks.ohlcv` (advertised 2011+) starts only in 2018 for many delisted names.
* Derived universes (`SP500_TOP50_DV` ...) inherit the base coverage (§6).
* **Delistings**: DELIST at the session after the verified series ends when the issuer is not in
  the current Nasdaq Trader directory; amount = last close (T2; the real cash-out / merger
  consideration is unknown).
* Yahoo is survivorship-biased for everything (no delisted tickers except stubs). ETF universes
  are fine (all sheet ETFs still trade) except that leveraged/inverse fund history is real only from
  inception.

**Other known gaps**
* Intraday history: Yahoo 1m only since 2026-09-30 (grows ~1 session/day if `--set daily` runs);
  5m 60 days; 60m 2 years. Multi-year 1m exists only as Dukascopy CFD mid quotes (T2, opt-in, no
  volume) -- **intraday strategies needing volume/VWAP/RVOL over years are NOT testable** with free
  data unless a key-gated provider (Alpaca SIP since 2016) is enabled.
* No NBBO/SIP quotes (IEX single-venue sample days and Dukascopy CFDs only). Spreads must be
  estimated (ENGINE).
* Options: sparse EOD chains, ~3 snapshots/week 2020-2024, DTE 11-66 only (no 0-10 DTE, no LEAPS),
  no QQQ/IWM/SPX (synthetic T3 fallback), no volume/OI, capture minute unknown, 2019 weekly only.
* Corporate actions: no declaration/record/pay dates (known_at = previous close, pay_date None);
  spin-offs, mergers' stock components and symbol changes are not modelled as actions (Yahoo does not
  provide them); DoltHub dividends for delisted names are split-adjusted amounts (T3 pre-2014).
* Fundamentals: XBRL only (2009+); no pre-2009 PIT fundamentals; Q4 quarter values not derived;
  no point-in-time shares outstanding before XBRL.
* Earnings estimates: weekly consensus from 2017-10 only; earlier only the final estimate from
  eps_history (2016+); none before 2016. Pre-announcement earnings DATES (EARNINGS_SCHEDULE) are
  assumed known 7 days ahead (no history of when calendars were published).
* Index changes: announcement date known for 623 of 1,544 events; the rest use a conservative
  previous-session fallback.
* Borrow: no historical borrow fees/availability (default model, T2); FINRA data excluded by terms.
* Sector: SIC-based mapping and current GICS only -- no PIT GICS history (2018 Communication
  Services reshuffle and 2023 GICS changes are not reflected historically).
* FRED now refuses the plain UA (local fallback works; updates need the scout's route or a FRED
  API key); www.sec.gov needs `SEC_USER_AGENT` with a contact.

## 9. Operations

* Full core download: `python -m atlasbt.data.fetch --set core` (intraday first, since Yahoo 1m
  disappears after ~30 days). Daily refresh: `--set daily` (cron, after 20:30 ET).
* Bulk mirrors: download the HF CSVs (see §4.2) into a fresh directory, then
  `ATLASBT_OHLCV_CSV=... ATLASBT_OPTION_CSV=... ATLASBT_EPS_ESTIMATE_CSV=... python -m atlasbt.data.fetch --set bulk`.
* Identity series: `--only identity` after any bars refresh (persists verified series per CIK for
  every S&P membership interval + megacaps; otherwise they are built on the fly, ~0.1-2 s each).
* Long backfills: `--only dukascopy_backfill` (CFD 1m since 2017, hours per symbol),
  `--only options` (DoltHub SQL for dates after the mirror).
* Budget: `ATLASBT_CACHE_BUDGET_GB` (default 40) is checked every 25 symbols.
* Tests: `python -m pytest tests/data -q` (no network; fixtures in `tests/data/fixtures`, recorded
  2026-10-28 from: yfinance (AAPL/TSLA/SPY daily rows incl. all split rows; SPY 1m 2026-10-23), Cboe
  VIX/SKEW CSV rows, FRED DTB3 rows, French FF3 daily zip (truncated), DoltHub SQL responses (SPY
  chain 2024-01-03; AAPL earnings tables), SEC AAPL submissions/companyfacts subsets, fja05680 rows,
  Wikipedia wikitext rows, Fed statement page 2024-01-31 + 2008 links, Nasdaq Trader rows,
  Dukascopy .bi5 files, ZipLime parquet subset, IEX TOPS packets (2019-12-24). Synthetic frames are
  named `synthetic_*` and only exercise mechanics.)

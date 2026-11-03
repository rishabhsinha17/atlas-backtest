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
                 sp500, ziplime, hod, iex, dukascopy, nasdaqtrader, borrow, iborrowdesk (keyed),
                 dividends_decl (EDGAR full-text search), buybacks (EDGAR full-text search + filing text),
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
| Alpaca SIP 1m/5m bars | T1_BAR | consolidated trades, 2016-> (§4.15) |
| Alpaca SIP NBBO quotes (decision windows) | T0_QUOTE | consolidated NBBO, cleaned (§4.15) |
| Corporate actions (Yahoo) | T2_PROXY | known_at approximated |
| DoltHub dividends for symbols Yahoo lacks | T2 (2014+) / T3 (pre-2014) | split-adjusted amounts un-adjusted with a split table that starts 2014 |
| Option chains (DoltHub, historicaloptiondata sample) | T2_PROXY | real quotes, capture minute unknown, sparse strikes |
| EARNINGS from ZipLime (exact 8-K acceptance) | T1 | to-the-second release time |
| EARNINGS from DoltHub calendar only | T2 | clock time approximated (BMO 08:00, AMC 16:30 ET) |
| INDEX_ADD/DEL | T2 | announcement time-of-day approximate (17:15 ET) or fallback |
| FOMC | T1 if time printed on page, else T2 | |
| Borrow, IBorrowDesk rows (keyed API) | T1 | IB stock-loan fee/availability, daily close reading, known 16:00 ET |
| Borrow, default model | T2 | no rows for the symbol/date (no key, before 2015-07-13, stale > 7 days) |
| DIVIDEND_DECL (EDGAR 8-K full-text match) | T2 | exact SEC acceptance time; hit -> dividend association inferred |
| BUYBACK (EDGAR 8-K text) | T2 | amount / intent parsed from free text (precision 27/30) |
| French / AQR factors | T2 | revised, not PIT -- validation only |

## 4. Sources

Rate limits are enforced in `config.HOST_MIN_INTERVAL` (+ exponential backoff, Retry-After honoured).
**Offline / reproducibility**: `ATLASBT_OFFLINE=1` makes `atlasbt.data.http` serve only its disk
cache (ttl ignored) and raise `FetchError` for anything else -- backtests and tests set it. Portal
use never fetches in any mode: the identity layer's SEC registrant-name lookups are cache-only
(`cache_only=True`); `fetch --only sec_submissions` (part of `--set core`) precaches the
submissions JSON of every CIK it may consult (1,263 CIKs).
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
* **Full-text search** (`fts_search`, `efts.sec.gov/LATEST/search-index`, 2001+): answers the plain
  UA (verified 200 on 2026-10-30); 100 hits per page, offsets <= 10,000; hits carry accession, form,
  items, exhibit type and filing date but NO text. `acceptance_map(cik)` attaches the acceptance
  time from the submissions JSON (`filing_known_at`: fallback filed date 17:30 ET, never early).
  `fetch_document` (filing text on `www.sec.gov/Archives`) refuses without `SEC_USER_AGENT`
  (`SecContactRequired`; the plain UA gets HTTP 403). The contact string is read from the
  environment only (never written to code, docs, cache metadata or logs).
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
* **Symbol convention (ARCHITECTURE §3)**: the internal OCC-style symbol encodes
  `Instrument.expiry` = the LAST TRADING DAY (SPX AM monthly Feb 2016 = `SPX   160218P01920000`,
  Thursday; it settles at the next session's open SOQ); the root (SPX vs SPXW) tells the series.
  `Portal.instrument(symbol)` decodes internal symbols only (date = expiry; settlement from the
  root: SPX/NDX/RUT... CASH_AM, SPXW/XSP/NDXP/RUTW CASH_PM, others PHYSICAL) and round-trips
  `Instrument.option(...)` exactly. Vendor symbols (vendor date = settlement Friday for SPX AM,
  or a holiday Friday) are converted ONCE at ingestion by `option_meta.to_internal`
  (DoltHub SQL / CSV mirror, historicaloptiondata): `symbol` = internal, `vendor_symbol` = the
  vendor's. Stored files that held vendor-convention symbols were re-keyed
  (`python -m atlasbt.data.fetch --only option_rekey`, idempotent); legacy frames are converted on
  read with the same function, so one contract never appears under two symbols.
* **Annotation** (`option_meta.annotate_chain`, applied when the Portal serves real chains):
  `expiry` = LAST TRADING DAY (holiday Fridays -> previous session; AM-settled index monthlies
  such as root SPX are listed by vendors on the settlement Friday -> expiry = Thursday,
  `settlement_date` = Friday); extra columns `root` (SPX vs SPXW), `settlement_date`, `kind`
  (MONTHLY / WEEKLY) and `underlying_price_source` (`vendor` for historicaloptiondata,
  `pit_bar_close_at_ts` for DoltHub: the raw Yahoo close of the snapshot session, known exactly at
  ts). Quote lookups match by internal symbol or by (root, expiry, strike, right).
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
  (Databento OPRA samples, §4.15c; absent underlyings return False / NA). `fidelity_of` is the best tier among the sources actually present
  (`portal.OPTION_SOURCE_FIDELITY`): exchange-timestamped quotes (databento / polygon / alpaca) T0,
  EOD captured at a documented time (cboe_eod) T1, EOD with approximate capture time (dolthub,
  hod) T2; with no real source, EOD falls back to the synthetic chain -> T3 when the underlying has
  daily bars; intraday -> NA. `has()` stays real-data only.
* Quote quality: SPY quotes are tight (2024-01-03 ATM 4.46/4.48), but single-stock DoltHub spreads
  look wider than the NBBO (AAPL near-ATM median relative spread 4.2% in 2019, 2.5-4.8% by year)
  and the 2019 Saturday snapshots may reflect after-hours quotes -> fills at DoltHub bid/ask are
  conservative for single names.
* Intraday option quotes: Databento OPRA (keyed, §4.15c) for the fetched plans only. Otherwise QQQ/IWM/SPX chains are
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

### 4.14 Borrow (`providers/borrow.py`, `providers/iborrowdesk.py`)
* **IBorrowDesk** (re-checked 2026-10-30, UA `atlasbt-research/0.1`): the bare host only redirects to
  `www.iborrowdesk.com`; the site's `/api/ticker/{T}` endpoints are the web app's internal interface
  and (per its `/llms.txt`) block scripted user agents -- the earlier probe's dropped connection was
  that block, and we no longer call them. Programmatic access is the **keyed v2 API** (paid: $29/month
  incl. 5,000 units; unit packs extra): `GET /api/v2/daily/borrow?symbols=..&start=..` = daily
  open/high/low/close of fee (% p.a.), rebate and shares available, **2015-07-13 ->**, one unit per
  symbol per 365 days returned, <= 150 units and <= 100 symbols per request, 60 req/min.
  `GET /api/v2/coverage` is free (read 2026-10-30: daily_earliest_available 2015-07-13).
  `iborrowdesk.fetch_daily` (env `IBORROWDESK_API_KEY`) batches by worst-case units, waits >= 1.1 s
  between requests, backs off on 429/5xx, stops on 400/401/402/403, sends the key only as a header
  (never in params, cache metadata or logs), keeps US lines, and drops a response whose name shares
  no word with the SEC registrant name. Stored PIT: `borrow/iborrowdesk/<TICKER>` with
  `time` (= known_at) = 16:00 ET of the date (the daily close is IB's last in-session reading),
  fee, rebate, available (+ open/high/low), figi, name. The API resolves a ticker to the security
  trading under it TODAY (with its history across renames), so only current tickers are requested
  and the files are ticker-keyed CURRENT stores (`Portal.resolve_symbol` serves a security key only
  when its CIK is the ticker's current registrant).
* **Fetch target** (`iborrowdesk.short_universe`): PIT S&P 500 members whose membership overlaps
  2015-07-13.. (base tickers) + CORE_ETFS + LEVERAGED_ETFS = 812 symbols; all-time history <= 12
  units/symbol -> worst case 9,744 units in 68 requests (from 2017: 8,120); billed units are lower
  for names that stopped trading (billing counts the span actually returned) -- the $29 plan's
  5,000 units plus unit packs. **Not fetched: NEEDS USER
  (`IBORROWDESK_API_KEY`).** Coverage today: 0 symbols x 0 years.
* IBKR's own public short-stock files (`ftp3.interactivebrokers.com`, user `shortstock`) are a current
  snapshot only (no history); FTP does not resolve from this environment anyway. FINRA short data
  is excluded by its terms.
* `Portal.borrow(symbol, as_of)`: the last stored row with `time <= as_of` -> (available > 0,
  fee / 100, T1_BAR); before the first row or when that row is older than
  `Portal.MAX_BORROW_STALE_DAYS` (7) -> the default model. Without rows the code path is unchanged
  (same memoised S&P-membership shortcut, bit-for-bit identical results; perf/fastportal tests).
* **Default model** (T2_PROXY, env-overridable): ETFs & S&P 500 members 0.30% p.a., other stocks
  1.0%, leveraged/inverse ETFs 2.0%; always shortable. The engine flags T2 on short sales only when
  the borrow it used is the default.

### 4.17 Dividend declarations (`providers/dividends_decl.py`, events `DIVIDEND_DECL`)
Sources checked 2026-10-30: Nasdaq's dividend-history API answers the honest UA but covers only
Nasdaq-listed symbols (SPY / DIA / IWM / JNJ "not available", QQQ declaration "N/A") and
`api.nasdaq.com/robots.txt` is `Disallow: /` -> not used (one probe of AAPL served as an
independent check below, nothing stored); the cached DoltHub `dividend` table has ex-date and
amount only; ETFs (SPY, QQQ, IWM, DIA) file no 8-Ks -> no free PIT declaration source for them.
**Method**: per issuer CIK (the 1,053 SEC-mapped S&P 500 PIT-union tickers), EDGAR full-text search
for declaration phrases (`DECL_QUERY`: "declared a quarterly" OR "declared a cash dividend" OR
...) in 8-K filings and exhibits, kept only for announcement documents (8-K body or EX-99.x under
Items 2.02 / 7.01 / 8.01; contracts, indentures, charters excluded). Each regular CASH_DIVIDEND the
portal serves (ex-date E, amount) is matched to the LATEST such document accepted in
(E - 75 days, open of E) with no earlier ex-date in between; known_at = that filing's SEC
acceptance time (an upper bound on the press-release time). Unmatched dividends get no event.
Payload: ex_date, amount (from the corporate action), accn, items, file_type, filename, lead_days,
cik. Fidelity **T2_PROXY** (timestamp exact, association inferred from timing: the search index has
no text). Events whose CIK is not the issuer the ticker denoted on the declaration date are dropped
(`identity_filter`, reused tickers). `dividends_decl.rematch()` re-derives the events offline from the stored hits
(`reference/sec_fts/DIVIDEND_DECL_HITS`, 14,446) and dividends (`.../DIVIDEND_DECL_DIVS`, 41,513).
* **Table** (2026-10-30): 8,059 events, 466 issuers, known_at 2004-09-21 .. 2026-10-13.
  Recall vs the portal's regular cash dividends of the 647 dividend-paying issuers: 2002-09 9.2%,
  2010-17 19.4%, **2018-26 24.9%** (4,569 of 18,354). Structural: many issuers announce
  dividends by press release only (no 8-K): MSFT, JPM, JNJ, XOM, PEP, WMT, MA, HD, UNH, NVDA have
  none; AAPL, PG, ORCL, V, COST, AVGO are complete 2018+.
* **Validation**: AAPL 54 of 54 declaration dates (2012-2026) equal Nasdaq's independently recorded
  declaration dates (and Apple's earnings-release dates). Lead (acceptance -> ex-date open): median
  20.7 days; 1.4% under 2 days (late mentions; conservative direction). An earlier 30-event sample
  showed an M&A agreement exhibit (LMT 2020-12) mis-associated -> the document-type filter. Text
  check of a random 30 (filings read with `SEC_USER_AGENT`, 2026-10-30): 29 of 30 documents declare
  the matched dividend (28 state the exact amount; 2 use number formats the check missed); 1 is a
  mis-association in the leak direction (UDR 2022-10-26 earnings release restating the Q3 dividend,
  matched to the Q4 ex-date 2023-01-06: known_at ~2 months early, same amount) -> **precision ~97%**.
* Consumer: S042 (declared dividends T2; dividends without a declaration record are skipped).

### 4.19 Analyst consensus, next-12-months EPS (`providers/estimates.py`; S041)
DoltHub `earnings.eps_estimate` weekly snapshots (mostly Sundays), periods 'Current Year' /
'Next Year' with consensus and `count` (number of estimates). Loaded 2026-10-30 from the HF CSV mirror
(`siddharthmb/stocks-earnings-eps_estimate`, 2017-10-26 .. 2026-01-25; deleted after loading) and
extended by the DoltHub SQL API (39 snapshot dates to 2026-10-25): 667,532 raw rows, S&P 500 PIT-union
tickers only (`reference/estimates/fy_consensus_raw`). **PIT**: capture time and DoltHub publication
lag are undocumented -> a snapshot dated d is known at the close of the first NYSE session after d
(Sunday -> Monday 16:00 ET). **Identity**: each snapshot's ticker is mapped to the issuer it denoted on
that date (identity layer); unknown identity -> dropped (5,890 snapshots). Served through
`Portal.fundamentals` as `eps_consensus_ntm` = w * FY1 + (1 - w) * FY2 (w = remaining share of FY1),
`eps_num_analysts_ntm` = min(FY1, FY2 counts), plus `eps_consensus_fy1/fy2`
(`fundamentals/estimates/<TICKER>`, 869 tickers, 1.30 M rows); latest snapshot known at as_of, none if
older than 21 days; a security key collects its issuer's snapshots across renamed tickers (FB -> META).
Fidelity **T2_PROXY** (blended NTM, not I/B/E/S; weekly granularity). Coverage: of the S&P 500 PIT
members, 497 / 506 have an NTM value at 2018-06-29 (442 with >= 5 analysts), 496 / 505 at 2021-12-31
(412), 500 / 503 at 2025-06-30 (468), 502 / 503 at 2026-10-23 (456). Checks against known revisions: NVDA FY1 4.52 -> 4.77 (2023-05-28 snapshot, after the 05-24
guidance) -> 7.46 (06-04); META (FB) FY1 13.91 -> 13.33 -> 12.93 after the 2022-02-02 report; AAPL
FY1 13.65 -> 12.16 over Mar-Apr 2020 (COVID).

### 4.18 Buyback authorizations (`providers/buybacks.py`, events `BUYBACK`)
S045 needs known_at, authorized amount as % of market cap and an earnings-bundling flag.
Candidates: EDGAR full-text search per issuer CIK for authorization phrases
(`AUTH_QUERY`) in 8-K filings/exhibits, known_at = acceptance, `with_earnings` = Item 2.02 in the
8-K -> `reference/sec_fts/BUYBACK_CANDIDATES` (2026-10-30: 11,307 documents in 8,999 filings of 838 issuers, accepted 2001-01-17 .. 2026-10-29; 5,787 filings carry Item 2.02, 3,212 do not). A phrase hit is not an
authorization (earnings releases routinely mention existing programs) and has no amount, so
candidates are never served as events. **Text stage** (`SEC_USER_AGENT`, www.sec.gov at <= 4 req/s;
2026-10-30): the 4,331 documents of the 3,212 candidate filings WITHOUT Item 2.02 were read (S045
excludes earnings-bundled releases, so those are not read; 7 pre-2002 `.txt` documents 404'd; raw
cache 232 MB). `parse_authorization` takes the largest NEW / increased authorization in dollars
(share counts x the prior close), skipping sentences that date the program > 60 days before the
filing, use past wording ("previously", "remaining", "had approved", "years ago") or are slide/table
text (> 350 chars); for "increase the existing $200M program to $350M" the amount is the stated
total. pct_mcap = amount / (shares outstanding filed before known_at x last close before known_at),
rows above 100% dropped, issuer identity checked as in §4.17. `with_earnings` is also set when the
issuer filed an 8-K Item 2.02 within +-24 h (63 events). Fidelity **T2_PROXY**.
* **Table** `events/sec_fts/BUYBACK`: 695 events, 294 issuers, known_at 2009-08-28 .. 2026-09-15
  (no events before 2009: XBRL shares outstanding start then); pct_mcap median 5.8% (5th-95th
  1.3%-18.6%); S045-eligible (>= 5%, not with earnings): 373 events / 214 issuers, 190 since 2018.
* **Precision** (random 30, judged by reading each sentence): 27/30 = 90% are a new or increased
  authorization announced in that filing; misses: an existing authorization restated (MMC 2009), a
  10b5-1 plan under an existing program (PDCO 2012), an accelerated repurchase executing an
  existing program (INTC 2020). The previous parser version (before the stale / past-wording /
  identity filters) scored 22/30. Recall is not measured (no reference list).

### 4.15 Alpaca SIP intraday (`providers/alpaca.py`) -- LIVE-VERIFIED 2026-10-30
Free paper-account key (env `ALPACA_API_KEY` / `ALPACA_API_SECRET`, sourced from `~/.atlasbt/secrets.env`;
read from the environment only -- never logged, cached, stored in metadata or committed). Raw responses are
**not** written to the HTTP cache (multi-GB JSON); the parquet store is the cache. Endpoints: multi-symbol
`GET data.alpaca.markets/v2/stocks/{bars,quotes}?symbols=..` with `feed=sip`, `adjustment=raw`, `asof=-`,
`limit=10000` (page limit across ALL symbols), `page_token` pagination.

**Verified against the live API** (probes 2026-10-30; recorded pages in `tests/data/fixtures/alpaca_*.json`):
* History starts **2016-01-04** (nothing earlier); the free plan serves up to 15 min ago.
* `t` = bar **START** (UTC): SPY's 14:30Z bar of 2016-01-04 opens at 200.49 = the official open; the 1Day
  bar is stamped at ET midnight. Minute bars include extended hours 04:00-20:00 ET (no session filter);
  `frames.intraday_bars` tags PRE/RTH/POST from the NYSE schedule (early closes: POST from 13:00).
* The **closing auction prints in the 16:00 (POST) minute** (AAPL 2016-01-04: 8.5 M shares at 105.35 in the
  21:00Z bar), so the last RTH 1m close is the last continuous trade, not the official close, and RTH 1m
  volume sums to ~90 % of the consolidated daily volume. Daily official prices stay the daily bars'.
* `asof`: by default Alpaca maps renamed tickers to today's entity (`META` for 2016 returns FB's tape). We
  always send `asof=-` (tape as traded) and decide which tape belongs to which security ourselves (below).
* **Quotes are the consolidated NBBO stream**, not per-exchange BBOs: bid and ask exchanges differ on 72-81 %
  of updates (a venue BBO always has one exchange). No NBBO rebuild is needed; the stream is cleaned
  (`clean_nbbo`): drop one-sided / non-positive sides; drop updates whose conditions are not a subset of
  {R regular, ? not reported (2016-~2021 data), O opening} (non-firm, closed, halted, slow/manual); drop
  **crossed** (ask < bid) and **locked** (ask = bid) updates (0.1-4 % of updates, mostly premarket and fast
  markets); keep the last update per timestamp; drop consecutive duplicates of (bid, ask, sizes) -- lossless
  for the NBBO state. ~70 % of raw updates survive.
* **Quote sizes**: round lots (x100 shares) for sessions **before 2025-11-03**, **shares from 2025-11-03**
  (SIP round-lot reform: SPY 2025-10-31 sizes 1, 2, 6..; 2025-11-03 multiples of the new 40-share lot; KO /
  INTC / JPM switch the same day on every tape; the 2025-10-31 POST session is still in lots). Stored sizes
  are always shares. NBBO update counts drop sharply on the new regime (KO: 1,388 -> 19 updates in the last
  10 s of the session) -- consistent with odd-lot-sized quotes no longer moving the round-lot NBBO.
* Rate limit header `X-RateLimit-Limit: 200`/min; we use a rolling-minute token bucket capped at **180/min
  per account** (split between concurrent fetch processes with `ATLASBT_ALPACA_RPM`), pause until
  `X-RateLimit-Reset` on 429, exponential backoff on 5xx / network errors, never retry 4xx.

**Identity routing** (reused tickers never splice). Universe symbols are tickers or security keys
`TICKER~CIK##########` (`Portal.universe`). For every wanted (symbol, session) the identity-verified RAW daily
close / volume that `bars(symbol, '1d')` serves is compared with Alpaca's 1Day bar of each candidate tape
ticker (the symbol's own ticker first, then the CIK's evidenced lineage and SEC tickers); a tape is accepted
for the session only if |ln close ratio| <= 0.5 % and |ln volume ratio| <= ln 2.5. Sessions without an accepted
tape get no intraday bars (fail closed). Intraday rows are stored under the universe symbol
(`bars_1m/alpaca/<SYMBOL or TICKER_CIK..>`), so FB (2016-2022) and META (2022-) each carry the right tape,
`DIS~CIK0001001039` and `DIS~CIK0001744489` each get only their own sessions, BRK-B maps to `BRK.B`.
Routes and per-symbol match rates are stored in `reference/alpaca/tape_routes_<set>` /
`route_summary_<set>`. Intraday bars are RAW; `bars(..., adjusted=True)` applies the same actions of the
same identity frame as the daily bars (SPY 2016-06-01: adjusted 1m close = adjusted daily close).

**What is fetched** (`python -m atlasbt.data.fetch --set intraday`; resumable: per (symbol, month) / (window,
month) ledgers in `cache/logs/alpaca_progress_*.jsonl`, an incomplete month is always refetched):

| task | symbols | frequency / sessions | dates |
|---|---|---|---|
| `alpaca_universe` | PIT S&P 500 members and top-60 by 20-day $ volume (the `SP500_TOP50_DV` definition) per session -> `reference/alpaca/universe_pit` | | 2016-01-04 -> |
| `alpaca_bars_etf` | SPY QQQ IWM DIA, 11 sector SPDRs, TLT GLD EFA EEM (S001-S030, S116, S031-S037) | 1m PRE+RTH | 2016-01-04 -> |
| `alpaca_bars_top` | every stock ever in the PIT top-60 (S001-S022, S027, S030 + 10-name margin) | 1m PRE+RTH | its top-60 sessions -30/+5 sessions |
| `alpaca_quotes` | windows below | cleaned NBBO | 2016-01-04 -> |
| `alpaca_bars_sp500_5m` | every PIT S&P 500 member (S039/S040) | 5m RTH | member sessions -30/+5 from 2017-11 (S&P coverage start 2018-02) |

POST bars are not stored (no strategy reads them; ~20 % of rows); 5m for 1m symbols is resampled by the Portal.

**Quote windows** (`alpaca.QUOTE_WINDOWS`, relative to the actual open/close, so early closes shift):
`pre_0920` [open-10m, open] top-50 members of the day (S005/S006 09:29 premarket midpoint); `close_m36`
[close-36m, close-35m] ETF_CORE (S015 15:25 midpoint); `close_m30` (S015 15:30 entry), `close_m5` (15:55
flat-by exits) ETF_CORE; `close_m10` (S031-S037 15:50 entries) and `open_p5` (09:35 exits) for the 19 ETFs.
Each recorded window is a closed interval [w0, w1] in `reference/alpaca/quote_windows`. **The Portal serves
Alpaca quotes only for requests inside a recorded window** (an engine 1m fill window (T, T+1m] is), for
premarket last-quote lookups ending at the open whose last minute is recorded, and for availability scans
longer than 7 days; any other request (5m / daily fill windows, other minutes) gets an empty frame, so the
engine falls back to bar fills instead of simulating a resting order on a partial quote path.

Coverage actually fetched: see §5.

**Planner changes needed for intraday** (`atlasbt/run/plan.py`, owned by the runner; verified by running the shipped
and a patched planner side by side, scratch script `plan_patched.py`):
1. `PlanContext.bar_range`: intraday ranges start at `now - 800 days` (the Yahoo era) -> must start at
   2016-01-01, otherwise every Alpaca-backed intraday window is cut to 2024-08-21..
2. `_plan_instance`, `kind == "quotes"`: the branch always returns NOT_TESTABLE ("required quotes exist on only N
   common session(s)") even for N in the thousands -> when N >= the minimum, append the first/last quote session to
   `starts`/`ends` and continue; for a PIT `universe_name` use the UNION of the members' quote sessions (Alpaca
   pre-open windows exist only on the sessions a stock is in the top-60, exactly when S005/S006 need them).
3. `quote_range` reads the whole quotes parquet of every member; it should read `reference/alpaca/quote_windows`
   (one row per recorded window) for Alpaca.
4. PIT intraday universes (`SP500_TOP50_DV`): members are resolved at the LAST date and the window starts when 80 %
   of TODAY's members have their first intraday bar. Alpaca bars exist for each stock on its top-60 sessions
   +-30/+5 (not its whole history), so this rule dates the window by today's members' first top-60 entry; the
   right test is per-session coverage of THAT session's members (as `sp500_full_coverage_start` does for daily).
5. Plan caches (`results/plan_cache.json`) memoise `bars|SYM|1m` / `quotes|SYM` as None from before the fetch:
   delete those keys (or plan into a fresh results dir).

### 4.15b Other key-gated providers -- UNVERIFIED (written from official docs, never run with a key)
| Provider | Env | What | Notes |
|---|---|---|---|
| `polygon.py` | POLYGON_API_KEY | aggregates `/v2/aggs/ticker/...` | **api.polygon.io is blocked (403) by this environment's egress proxy** |
| `tiingo.py` | TIINGO_API_KEY | EOD raw + divCash/splitFactor (incl. delisted tickers), IEX intraday | IEX intraday = single venue (T2) |
| `databento.py` (equity helpers only) | DATABENTO_API_KEY | `fetch()` ohlcv-1m / bbo-1m csv, budget-guarded like OPRA (§4.15c), per-request cap DATABENTO_MAX_COST (default $1) | XNAS.ITCH = Nasdaq venue only; the OPRA part is LIVE-VERIFIED (§4.15c) |
| `alphavantage.py` | ALPHAVANTAGE_API_KEY | TIME_SERIES_INTRADAY month=YYYY-MM, extended hours, adjusted=false | 25 req/day free |
Without keys `available()` is False and fetch functions return `{'skipped': ...}` (tested).

### 4.15c Databento OPRA option quotes (`providers/databento.py`) -- LIVE-VERIFIED 2026-10-30
Key `DATABENTO_API_KEY` (environment only, sourced from `~/.atlasbt/secrets.env`; never logged/stored). The
account runs on free credits, so **every data request is priced first** (`metadata.get_cost`), written to the
ledger `cache/raw/databento_ledger.csv` (timestamp, dataset, schema, symbols, window, cost, status) and refused
if the cumulative spend would exceed `DATABENTO_BUDGET_USD` (default **$90**). Listings come from the free
`symbology.resolve` (parent `<ROOT>.OPT` -> every OSI symbol listed that day). Small raw responses are cached
(`cache/raw/hist.databento.com/`); large ones are not (disk) -- the parquet store plus the progress ledger
`cache/logs/databento_progress.jsonl` make `python -m atlasbt.data.fetch --only databento_options` resumable
without paying twice (env `ATLASBT_DATABENTO_PLANS`, `_START`, `_END`, `_DRY_RUN=1` prices only).

**Verified semantics** (pilot 2024-01-02, SPXW 0DTE, $0.0013): dataset `OPRA.PILLAR` starts 2023-03-28;
`cbbo-1m` = OPRA consolidated best bid/offer (all exchanges) **sampled on the minute grid**: `ts_recv` is the
sample instant (exact minute), the quote is the book state at that instant, a record is emitted every minute
for every requested contract (404/404 minutes), after the close the last quote is repeated unchanged;
`ts_event` tracks the last trade, NOT the last quote change, so `age_s = ts - ts_event` is "time since last
trade". Zero bids are real ("no bid") quotes. Price: ~$1.5e-7 per contract-minute with a floor of ~$3e-6 per
contract per request (1-10 minutes cost the same); a full session of one contract $6e-5; the whole SPXW chain
for one minute $0.019. `cbbo-1s` exists only from 2025-02-20; `cmbp-1` (tick NBBO) is ~25x cbbo-1m.
**The sample stamped exactly at the closing bell is unreliable** for non-expiring SPX/SPXW series (30-DTE ATM
SPXW call 69.9/75.9 at 16:00:00 vs 75.2/75.8 at 15:59 and 71.2/71.6 at 16:01 on 2024-01-31), so "closing"
samples are taken at close - 1 minute (15:59 ET).

**Storage / serving.** Kind `option_quotes`, source `databento`, one file per (underlying, ET session):
`store/option_quotes/databento/<UND>/<YYYY-MM-DD>.parquet` with `ts, symbol (internal OCC: expiry = last
trading day; SPX AM monthlies listed by OPRA under the settlement Friday are moved to the Thursday by
option_meta, SPX vs SPXW keep their roots/settlement), bid, ask, bid_size, ask_size, age_s, chain`.
`known_at = ts`. The Portal serves them as **T0_QUOTE** (`OPTION_SOURCE_FIDELITY['databento']`):
* `option_chain(u, as_of)`: the latest sample instant <= as_of whose rows are flagged `chain` (fetched as a
  decision snapshot, see plans) if it is <= 15 min old and strictly newer than the EOD (DoltHub/hod) snapshot;
  otherwise the EOD / synthetic path is unchanged. Valuation-only rows (`chain=False`, e.g. daily SPX box
  marks, 0DTE samples in the last 20 minutes) are never served as a chain, so end-of-day consumers are not
  handed a truncated chain. `underlying_price`: index underlyings (SPX) = put-call parity of the same snapshot;
  equities/ETFs = close of the last 1m (else 5m) bar known at ts.
* `option_quote` / `option_quotes`: every sample (chain or not); at equal ts OPRA wins. A contract that has
  OPRA samples on a session never receives a synthetic substitute for a window without samples.
* `has(u, 'options', '1m')` / `fidelity_of` -> T0 when the underlying has OPRA files; for EOD (`'1d'`) OPRA
  counts only when no DoltHub chain exists for the underlying (SPX, QQQ, IWM).
* **SPX minute levels** (no index feed in OPRA): `bars_1m/opra_parity/SPX`, **T2_PROXY**: at every 1-minute sample
  the level is the median of K + C_mid - P_mid over the 5 SPXW 0DTE strikes with the smallest |C - P| (carry over
  the remaining hours < 3e-5, ignored); bar [t-1m, t) has close = level(t), open = level(t-1m), high/low = max/min
  of the two samples (no intra-minute path), volume 0. Check vs the Cboe official close on the pilot week
  (2024-01-29..02-02): -0.10, +0.31, -0.17, -0.37, +0.69 index points (<= 0.014%).

**Plans** (`databento.PLANS`; fetch selection uses data known later, e.g. the day's high/low, only to choose WHAT
to download -- strategies never see it):

| plan | strategies | contracts | samples |
|---|---|---|---|
| `spx0dte` | S161-S164 | SPXW expiring that session, strike in [low(1-m), high(1+m)], m = max(3%, 3.5 VIX_prev/sqrt(252)) | every minute open..close (chain until close-20m) |
| `spxeod` | S172 | month-end session + next: SPX & SPXW, DTE 1-60 within +-15%, 61-300 within +-6% (chain); other sessions: the series S172 would pick at each of the last month-ends (DTE 30/90/180) +-75 pts (marks) | 15:59 |
| `earnings` | S158/S159 | every PIT S&P 500 member with a scheduled BMO/AMC release: expiries [max(next session, E+2d), E+11d], strikes +-25% | entry session E 15:00-15:31, next session 09:30-09:46 |
| `s160` | S160 | decision session R (first 10:00 after the actual release): 10:00-10:01 puts 0.8-1.05 S / calls 0.95-1.2 S DTE 14-60; 10:00-close the puts (0.7-1.02 S) of the 30-DTE expiry S160 selects (choose_expiry replicated); 15:59 marks of those puts until 6 days before expiry | see left |
| `s171` | S171 | SPY/QQQ/IWM/DIA DTE 0-60, +-10% | 15:24-15:27 (inside the Alpaca NBBO window close_m36) -- not fetched yet: no 2023+ Alpaca NBBO for ETF_CORE in the store |
| `validation` | OPTIONS.md §7.4b | SPY/QQQ/IWM DTE 5-70 +-12%, one session every 2nd month | 15:59, stored apart (`reference/databento_validation`), never served |

Coverage fetched: see §5 (Databento OPRA).

### 4.19 Survivorship patch: delisted S&P 500 members (`providers/alpaca_eod.py`, `providers/tiingo.py`; 2026-10-30)
Measured first: the full DoltHub `stocks.ohlcv` mirror (18,893 symbols, re-downloaded 2026-10-30) has
**no symbol whose last bar is before 2018-01-01** -- DoltHub cannot patch anything delisted before
2018 (the bulk re-ingest added nothing; the store already held every relevant DoltHub row).
* **Alpaca SIP daily bars** (`alpaca_eod`, 2016-01-04 ->, key-gated, read-only reuse of
  `alpaca.fetch_bars_raw`, `asof=-` = as traded under the symbol, own 60 req/min bucket):
  bars_1d/alpaca_eod/<T> for the 106 tickers of members (and their CIK lineage) without verified
  bars 2016-2018. Zero-volume filler bars (Alpaca repeats the last print on no-trade sessions: DD
  2017-09 .. 2019-05) are dropped. **Validation** (47 survivors, 94 comparisons vs Yahoo raw and
  DoltHub, `reference/alpaca_eod/validation`): 100 % pass, median |log close ratio| 1e-8 (identical
  closes), median volume difference 7.8 %, 26/26 Yahoo splits reproduced by Alpaca raw gaps;
  per symbol, a ticker whose overlap with DoltHub disagrees is not stored. Identity: rows are
  labelled by CIK evidence exactly like DoltHub rows (rank between DoltHub and unlabelled rows);
  DoltHub rows of a ticker on >= 5 consecutive sessions with no trade on the SIP tape lose their
  identity (DowDuPont 2017-09 filed under 'DD' while old DuPont no longer traded); an unexplained x3
  move inside an Alpaca/Tiingo chunk is an identity break (AmSurg -> 'EVHC' 2016-12-02, Valaris new
  equity 2021-05-03): the rest of the chunk is dropped.
* **Tiingo EOD** (`tiingo`, user's free key, header auth only -- the key never reaches the raw cache;
  free tier 50 symbols/h, 1,000 req/day, 500 symbols/month -> we stay at <= 40/h, 400/day, 400/month
  with a persistent ledger `cache/logs/tiingo_requests.jsonl`, resumable). A Tiingo line is keyed by
  the ticker the security LAST traded under and carries its whole history; its supported-ticker list
  (`reference/tiingo/lines`) shows one row per line. Findings: many delisted names exist only from
  2016-01-04 (HAR, LLTC, CVC, HOT, BRCM), names delisted before ~2015 are mostly absent (BMC, CEPH,
  MOLX, TLAB, NOVL), and a reused ticker serves its CURRENT holder (CA, MON, APC, DOW, DD) -> only
  tickers whose latest line starts before 2016 are requested (`needed_symbols`). Identity rule: rows
  are used for CIK c only if the line's owner (issuer the ticker denoted at the line's end) is c,
  inside c's evidence span; renamed history and every row of a multi-line ticker need price
  agreement with a verified segment. Tiingo is a BACKFILL: where a trusted source (Yahoo, Alpaca,
  DoltHub) covers the same issuer later, the symbol is gated on the first 250 overlapping sessions
  and only rows up to that junction are stored. Actions from Tiingo divCash / splitFactor.
  **Validation** (`reference/tiingo/validation`): see §8.2 for the numbers of the final run.
* **Identity fixes** (`identity.py`, `sp500.py`): share-class lines without own evidence take the
  evidenced sibling class's date labels (VIAB -> VIA = Viacom 2006-2019, DISCK -> DISCA, CMCSK ->
  CMCSA; letters B/C/K only); fja's closed intervals listed under a ticker adopted only after leaving
  the index are rebased to the PIT ticker by SEC CIK evidence (`pit_rebase`, `fja_base` keeps the
  original: AABA -> YHOO 1999-2017, VIAV -> JDSU 2000-2013, ATGE -> DV 2009-2012, FNMA -> FNM,
  ABKFQ -> ABK, IAC -> IACI, KATE -> LIZ); splits that the prices contradict (no gap) or that
  re-encode a same-day spin-off already carried as SPINOFF are dropped (GE 2024-04-02 1.253, SSP
  2025-12-08 1.33).
* Not usable / not used: Stooq (connection reset by the egress proxy), Nasdaq Data Link WIKI (free
  key required; not requested), qlib_us_1d (Yahoo split-adjusted snapshot of 2020, no split list,
  starts 2012-06-18 for most dead names -- kept for cross-checks only), Hugging Face re-uploads of
  the paywalled paperswithbacktest Stocks-Daily-Price (terms).
* Fetch: `fetch --only alpaca_eod`, `--only tiingo_eod` (keys from the environment), then
  `--only identity`.

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
| fundamentals / estimates | 869 | 1.30 M | | weekly NTM / FY1 / FY2 consensus + analyst counts 2017-10-26 .. 2026-10-25 (§4.19) |
| events / sec_fts BUYBACK | 294 | 695 | <1 MB | repurchase authorizations 2009-08 .. 2026-09 (§4.18) |
| events / sec_fts DIVIDEND_DECL | 466 | 8,059 | <1 MB | dividend declarations, known_at 2004-09-21 .. 2026-10-13 (§4.17; built 2026-10-30) |
| reference / sec_fts BUYBACK_CANDIDATES | 838 | 11,307 | <1 MB | phrase hits (search index only), not events (§4.18) |
| borrow / iborrowdesk | 0 | 0 | | keyed API not fetched (NEEDS USER: IBORROWDESK_API_KEY) |
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
  priority: daily yahoo > dolthub > tiingo > cboe; intraday **alpaca** > yahoo > polygon > databento >
  alphavantage > tiingo (dukascopy only if opted in); a coarser native store of a lower-priority source
  (Yahoo 5m) is not used when a higher-priority 1m store exists (Alpaca 1m is resampled instead).
  Identity-keyed intraday stores (`TICKER~CIK..`) are read under the key itself. `attrs`: source, fidelity, adjusted.
  **Daily requests for stocks go through the security-identity layer (§8.1). Identity is a
  property of the BAR DATE, never of `as_of`**: each requested session is attributed to the issuer
  (CIK) the ticker denoted on that date (`ticker_timeline`); `as_of` only filters knowledge
  (known_at, adjustment actions). A bare ticker whose window spans two issuers raises
  `DataUnavailableError` ("... never spliced -- request a security key"); a **security key**
  `TICKER~CIK##########` (what `universe()` returns for reused or renamed tickers) serves exactly
  that security's verified history, whatever ticker it traded under. `attrs`: `security_id`,
  `identity_log`, `source` (e.g. `yahoo+dolthub`), fidelity T2 when a bar date's identity came
  from pre-2006 / copied-ticker / membership evidence or another ticker's / DoltHub's history.
  Unknown identity -> `DataUnavailableError` (fail closed). ETFs, indices and Cboe series are
  served directly. `security_id(symbol, as_of)` = issuer on that bar date.
  Intraday: `sessions=('PRE','RTH','POST')` selects session parts (Yahoo has pre/post). Missing
  native frequencies are resampled causally from 1m: RTH buckets aligned to the 09:30 ET open,
  `known_at` = bucket end, capped at the actual close on half days (tested on 2023-11-24).
* `quotes()` (alpaca > databento > iex > dukascopy); Alpaca only inside recorded decision windows (§4.15).
* `instrument(symbol).exchange` = primary listing venue (NYSE / NASDAQ / AMEX / ARCA / BATS) from
  the Nasdaq Trader directory (current); `exchange_at(symbol, as_of)` -> (venue, source, inferred):
  a PIT mover table (ORCL Nasdaq -> NYSE 2013-07-15, TXN NYSE -> Nasdaq 2012-12-12, PEP NYSE ->
  Nasdaq 2017-12-20, SPY/MDY AMEX -> ARCA 2008-12-01, QQQ AMEX -> Nasdaq 2004-12-01; sector SPDRs
  and DIA AMEX -> ARCA late 2008 marked approximate/inferred), else the current directory value
  marked `inferred=True` for dates before its snapshot. `security_master()` carries
  `exchange_pit`, `exchange_source`, `exchange_inferred`.
* `has(symbol, kind, freq=None, sessions=None)`: symbol `'*'` = any data of that kind;
  `has('*', 'events', 'EARNINGS'|'INDEX_ADD'|'INDEX_DEL'|'FOMC'|'EARNINGS_RELEASE'|'EARNINGS_SCHEDULE'|'DIVIDEND_DECL')`
  True when the source is loaded (BUYBACK: from events/sec_fts/BUYBACK, §4.18; HALT: False -- no source integrated). DIVIDEND_DECL / BUYBACK are served
  from one stored table each (`Portal.STORED_EVENT_KINDS`, symbol + cik columns: security keys get
  only their issuer's events);
  `has('*', 'fundamentals', field)`; `has(sym, 'bars', '1m', sessions=('PRE',))` checks session
  availability. **Bars availability for a date**: `has(sym, 'bars', freq, start=, end=, as_of=)`
  is True iff `bars(sym, freq, start, end, as_of)` serves >= 1 bar -- it is computed by calling
  bars() (`bars_coverage(sym, start, end, as_of)` returns available / first / last / sessions /
  expected_sessions / coverage / security_id / fidelity / reason), so it can never disagree with
  what bars() serves (identity, fail-closed). Without a window, `has(sym, 'bars', '1d')` means
  "identity-verified bars exist for SOME date" (a historical ticker such as GENZ is servable only
  for its own era) -- engines deciding a backtest window should use the window form.
  `universe(name, as_of, require_bars=True)` keeps only members with a served bar in the last 5
  sessions (same rule as the survivorship report).
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
  it existed at as_of; NOT fully PIT) -- one row per `universe('SP500', as_of)` member exactly as spelled (security
  keys included), resolved for THAT issuer by CIK from the cached SEC submissions (name in force at
  as_of, SIC -> sector, exchanges; Wikipedia GICS only when the current constituent is the same
  CIK); rows without a sector carry `sector_unknown_reason` (all are members whose issuer cannot
  be established -- bars() fails closed for them too). Sector known for members: 1998-06 63.9 %,
  2003-06 89.5 %, 2008-06 96.8 %, 2013-06 99.0 %, 2019-06 99.4 %, 2025-06 100 %, `universe(name, as_of)`: `SP500` (PIT), `SECTOR_SPDR`,
  `SECTOR_SPDR_9`, `BROAD_INDEX_ETF`, `GEM_DUAL_MOMENTUM`, `ACCEL_DUAL_MOM`, `CROSS_ASSET`,
  `TREASURY_DURATION`, `TBILL_PROXY`, `LEVERAGED`, `EQUAL_WEIGHT_PAIR`, `CORE_ETF`, `MEGACAP`
  (fixed lists only include ETFs with a completed bar by as_of), and `SP500_TOP<N>_DV` (e.g.
  `SP500_TOP50_DV`): PIT S&P 500 members ranked by mean raw close x volume over the 20 sessions
  strictly before the session of as_of (first call ~12 s, then memoized per session).
  **`universe('SP500', as_of)` returns SECURITY keys, not point-in-time tickers.** fja05680's
  "Updated" file lists survivors under TODAY's ticker (CVX, CBRE, WELL, GOOGL, ES, JPM, SPGI, MO
  ... before they traded under those symbols), so a member is identified by its membership
  interval (S&P keeps a member through renames): open interval -> Wikipedia's CIK of the current
  constituent; closed interval -> the issuer the ticker denoted on its last member session. The
  bare ticker is returned when it denotes that same issuer on as_of and was never reused; else
  `TICKER~CIK##########` (e.g. `BBT~CIK0000092230` for BB&T in 2019, `MO~CIK0000764180` for
  Philip Morris in 1996). `universe(..., require_bars=True)` keeps members with a served bar in
  the last 5 sessions. Events, fundamentals and instrument() accept keys (events filtered by the
  issuer's CIK, XBRL facts of the key's registrant).
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

**Corporate-action cross-check** (`ca_crosscheck.py`, ENGINE agent; method and full numbers in docs/ENGINE.md
§3): classified Yahoo actions vs DoltHub splits/dividends and DoltHub's as-traded overnight gaps, identity-guarded,
flag-only (severity `error` / `warn`, `flags()` for fail-closed consumers). `python -m atlasbt.data.ca_crosscheck`
-> `cache/qa/ca_crosscheck_<date>.csv`. 2026-10-30, 657 symbols: splits 43 of 171 comparable events disagree
(25.1 %, mostly Yahoo splits that DoltHub omits, confirmed by DoltHub's own prices), dividends 1,439 of 29,300 (4.9 %). Errors in the
served actions: GE 2024-04-02 (GE Vernova spin-off served both as SPINOFF and as a 1.253 SPLIT -- needs a fix in
`actions_classify`) and SSP 2025-12-08 (1.33 "split" without any price gap).

## 8. Survivorship & known gaps (blunt)

### 8.1 Security identity layer (`identity.py`, `Portal.ticker_timeline`; red-team #1-#4, #10, audit 2026-10-29)

Tickers are reused and both price sources mis-attribute history: Yahoo serves TODAY's company
under a ticker (BBT, IR, CPWR, EP, GENZ, SII, CCU ... would get another company's prices) and
DoltHub is not reliably as-traded (VIAC holds CBS 2011-2017, CB interleaves ACE and old Chubb,
FI 2014-2021 is Frank's International). Stock prices are therefore served only for an
**established security identity**; otherwise `bars()` raises `DataUnavailableError` and the
member counts as missing (fail closed).

**Identity is a property of the bar date.** (The first version resolved the ticker once, at
`min(end, as_of)`: an engine run loading 2019-2026 with as_of = 2026 bought 'BBT' on 2019-12-02
at Berkshire Hills' $31 instead of BB&T's $54 -- ~35k wrong-company member-sessions. Fixed.)
`ticker_timeline(T)` = segments of NYSE sessions with the issuer T denoted, built from all of
history like a security master (identical for every as_of). Inside a segment the date is
attributed; after a segment the security is carried forward (it keeps trading, e.g. BB&T as TFC)
until the next issuer's segment; before the first segment nothing is attributed (an earlier
company may have used T). A bare ticker never splices two issuers (error with the keys to use);
security keys (`T~CIK..`) serve one security.

* **security_id** = `CIK##########` (SEC registrant). Evidence "ticker T denoted CIK c on date d":
  ZipLime's 8-K corpus (every 8-K item + 2.02 announcements, 1,477,443 evidence rows in
  `reference/identity/evidence`) plus the SEC current ticker map as evidence on its snapshot date.
  Only `as_of` evidence (ticker observed at filing; **2006+**) labels dates directly: bracketed by
  evidence of the same CIK <= 548 days apart, or within 120 days of the nearest evidence. Where two
  issuers hand a ticker over, the switch is placed on the as-traded DoltHub identity break (> 20 %)
  near the evidence midpoint (ARNC: Arconic Inc -> Arconic Corp on 2020-04-01).
* **Copied (carried) ticker evidence** (a later ticker copied onto old filings) is used only
  within 120 days, only when a member name is recorded AND matches the CIK's SEC name in force on
  that date (CCI 1998 = Citicorp, not Crown Castle; UNM 1997 = UNUM, not Provident -> fail
  closed), and only if the issuer was not renamed before its first as-of evidence.
* **Before 2006** (`pre_evidence_cik`): the ticker's earliest evidenced issuer is accepted only if
  the member of that S&P membership interval (left-censored 1996 intervals extend back) confirms
  it -- Wikipedia's CIK of a current constituent, the recorded member name, or a verified
  successor -- AND the issuer's SEC name in force at the date equals its name at its first
  evidence (no rename: SBC -> AT&T Inc. took 'T' in 2005-11, so bare 'T' in 2004 fails closed).
* **Ticker hand-over inside one fja interval**: fja keeps a single open 'IR' interval from 2010
  although Ingersoll-Rand plc became Trane ('TT') and Gardner Denver took 'IR' on 2020-03-02; the
  member key then follows the bar-date issuer (`IR~CIK0001466258` in 2019, `IR~CIK0001699150` in
  2021).
* **Membership continuity**: a ticker renamed back and forth while in the index (FISV -> FI ->
  FISV) or renamed in the index before ZipLime's as-of ticker caught up (BLL -> BALL 2022-05-10)
  is attributed to the CIK the neighbouring interval's ticker denotes.
* **Issuer succession on the same line** (`is_successor`, `_succession_ok`): holding-company
  reorganizations / re-domiciliations (ORCL 2006, DUK 2006, MDT 2015, DIS 2019, XOM 2025) keep the
  Yahoo line: accepted when the evidence passes from the old to the new CIK with no third issuer,
  names match, the Yahoo raw close is continuous across the hand-over, DoltHub as-traded prices of
  the predecessor agree, and the successor had no ticker of its own before (an acquirer that
  adopts the target's ticker carries its own price line: ACE -> CB 2016, SBC -> T 2005).
  Member keys (open S&P intervals map to today's registrant) serve e.g. MRK, APA, SPGI, DXC, MO,
  NEE, GOOGL, AON over their whole Yahoo line.
* **Lineage**: the CIK's ZipLime tickers; a share-class ticker without evidence (FOXA, BRK-B) is
  its own lineage; SEC current tickers of the CIK only by price agreement.
* **Segment acceptance**: Yahoo under a lineage ticker whose CURRENT owner (SEC ticker map) is the
  CIK (or its verified successor) and median price > $1; DoltHub rows on dates labelled with the
  CIK; unlabelled DoltHub rows only by price agreement with an accepted segment (median |log
  ratio| <= 1.5 %, 99th pct <= 5 %, >= 20 common sessions) or as renamed history (CBS 2011-2017
  from DoltHub VIAC). **QA feeds identity**: for DoltHub, clustered spike-and-revert days (>= 3 in
  20 sessions, or more than max(3, 1 %)) reject the segment and isolated glitches are dropped; for
  Yahoo only TRUE glitches count -- the price returns to the same print the next session
  (|r_t + r_t+1| < 0.5 %) in a window without a sustained move (|net| < 15 %); >= 3 in 20 sessions
  cut the history up to the cluster (thin tick-quantized 1970s-90s data: COO before 1994-01, NVR
  1993-07, MNST 1993-05, COHR 1993-06, WM 1990-02, FICO 1988-05); genuine volatile regimes are
  never cut (CVNA 2021-2022 regression test). Names compare on tokens and squeezed strings.
* **Splice validation**: a source change with an unexplained close-to-close move > 20 % (or > 6
  MADs) drops the lower-priority chunk -- unless that chunk is >= 20x longer (a 1-bar Yahoo stub
  never evicts years of DoltHub: QRVO); across a data gap the allowed move scales with sqrt(gap).
* **Actions follow the segments** actually served (Yahoo actions of the Yahoo ticker used, DoltHub
  splits/dividends for DoltHub chunks, plus known spin-offs for DoltHub chunks: MRO/MPC 2011).
  Every split is validated against the raw open / previous close, allowing for same-day
  distributions (EXPE 2011 1:2 reverse split + TRIP); unexplained DoltHub jumps become inferred
  splits only for 2,3,4,5,10 and 1/2..1/20 with a matching volume change.
* **Spin-offs are value-based** (`actions_classify`, audit): SPINOFF `amount` = the children's
  first ex-date close x distribution ratio (known at that close); the adjustment factor is
  S / (S + D) with S = parent ex-date close per pre-ex share, i.e. the true total return. Yahoo's
  double encodings (dividend AND fractional split: DHR/FTV 2016, ABT/HSP 2004) collapse to one
  event; composite reverse split + spin-off events carry both (DD/CTVA 2019, ITT 2011, EXPE 2011).
  Without a child price: a cited close (Exelis 2011-11-01 $11.70, Rochester Business Journal),
  else Yahoo's dividend amount, else the value implied by the ex-date open.
  Ex-date adjusted returns now: HPQ 2015 +5.0 %, DHR 2016 +2.3 %, ABT 2004 +2.8 %, MRO 2011 +2.6 %,
  EXPE 2011 -2.7 %, DISCA 2014 -2.7 %, ITT 2011 +6.2 %; DD 2019-06-03 +10.2 % follows from the
  prints (DoltHub and Yahoo agree: DD open 66.00 -> close 76.10, CTVA close 24.81), not
  independently cited.
* **Explicit identity end**: a security's series ends at its last verified bar; DELIST is emitted
  at the next session if the issuer no longer trades (amount = last close: merger consideration
  is not in any free source, T2).
* Persisted per ticker (`identity/timeline`) and per CIK/lineage (`identity/series`,
  `identity/actions`) by `--only identity`; rebuilt on the fly when `BUILD_VERSION` changes.
  `attrs['identity_log']` lists every accept / reject / drop / inference.

**Known cases (regression tests `tests/redteam/data`, `tests/redteam/data_audit`, `tests/data`)**

| Ticker / date | Outcome |
|---|---|
| BBT 2019-12, run 2019-2026 (engine path) | key `BBT~CIK0000092230` fills at BB&T's as-traded price; bare 'BBT' over 2019-2026 is never spliced |
| IR 2019 (engine path), CB 2010, CCU 2005, AIT 1998 with as_of 2026 | PIT issuer or fail closed -- never the later owner |
| PARA / VIAC / CBS | CIK 813828 lineage; CBS 2011-2017 from DoltHub VIAC rows accepted as renamed history |
| UTX 2019, HRS 2019, SYMC 2018, MMC 2015 | renamed history from Yahoo RTX, LHX, GEN, MRSH |
| COG 2015 / CTRA 2022, FISV 2018, FI 2024, BALL 2024, QRVO 2020 | as-traded / membership continuity / stub-safe splice |
| FI 2018 | Frank's International (not Fiserv) |
| ORCL 2005, DIS 2012, MDT 2010, DUK 2005, XOM 1996 (Exxon) | holdco succession / Yahoo line |
| KO / GE / MSFT 1996 | pre-evidence, confirmed by member CIK, T2 |
| CCI 1998 (Citicorp), UNM 1998 (UNUM), CPWR, EP, GENZ, WEN 2005, MCIC, CNC 2000, SII 2005, FOXA 2019-03-12, BK 1998, bare T 2004, bare MO 1996 | **fail closed** (member keys serve SBC as `T~CIK0000732717` and Philip Morris as `MO~CIK0000764180`, the fja members) |

### 8.2 S&P 500 price coverage after fail-closed

`universe_survivorship_bias('SP500', date)` over the member SECURITY keys (member has an
identity-verified bar on the session or within the 3 previous sessions; recomputed 2026-10-29
after the bar-date identity redesign):

| date | 1996-06 | 2000-06 | 2005-06 | 2008-06 | 2010-06 | 2012-06 | 2015-06 | 2018-06 | 2020-06 | 2022-06 | 2024-06 | 2026-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| members | 487 | 492 | 496 | 497 | 499 | 497 | 499 | 506 | 505 | 503 | 503 | 503 |
| with verified prices | 171 | 222 | 259 | 299 | 328 | 389 | 419 | 489 | 503 | 500 | 503 | 502 |
| coverage | **35.1 %** | **45.1 %** | **52.2 %** | 60.2 % | 65.7 % | 78.3 % | 84.0 % | 96.6 % | 99.6 % | 99.4 % | 100 % | 99.8 % |

(2021-06 99.6 %, 2023-06 100 % with 22 members' last bar 1-3 sessions old; member-session coverage
over the trailing 20 sessions equals the point coverage within 0.3 pp.)

* Coverage is below the first, end-anchored build (1996 40 %, 2000 48 %, 2005 55 %, 2008 63 %,
  2010 68 %, 2012 78 %, 2015 85 %): those extra names were partly the WRONG company's prices; the
  remaining gap is delisted members absent from every free source.
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
  declaration timestamps exist separately as DIVIDEND_DECL events for ~25% of S&P cash dividends
  2018+ (§4.17; none for ETFs);
  spin-offs, mergers' stock components and symbol changes are not modelled as actions (Yahoo does not
  provide them); DoltHub dividends for delisted names are split-adjusted amounts (T3 pre-2014).
* Fundamentals: XBRL only (2009+); no pre-2009 PIT fundamentals; Q4 quarter values not derived;
  no point-in-time shares outstanding before XBRL.
* Earnings estimates: weekly consensus from 2017-10 only; earlier only the final estimate from
  eps_history (2016+); none before 2016. Pre-announcement earnings DATES (EARNINGS_SCHEDULE) are
  assumed known 7 days ahead (no history of when calendars were published).
* Index changes: announcement date known for 623 of 1,544 events; the rest use a conservative
  previous-session fallback.
* Borrow: no historical borrow fees/availability in the store (default model, T2) until an
  IBorrowDesk API key is supplied (§4.14; keyed provider ready); FINRA data excluded by terms.
* Buybacks: 2009+ only, S&P union issuers, precision ~90%, earnings-bundled 8-Ks not parsed (§4.18).
* Analyst revisions (S041): consensus only from 2017-10-26 (weekly; NTM blended, T2; §4.19). iNAV
  (S110): no free PIT source.
* Sector: SIC-based mapping and current GICS only -- no PIT GICS history (2018 Communication
  Services reshuffle and 2023 GICS changes are not reflected historically).
* FRED now refuses the plain UA (local fallback works; updates need the scout's route or a FRED
  API key); www.sec.gov needs `SEC_USER_AGENT` with a contact.

## 9. Operations

* Full core download: `python -m atlasbt.data.fetch --set core` (intraday first, since Yahoo 1m
  disappears after ~30 days). Daily refresh: `--set daily` (cron, after 20:30 ET).
* Bulk mirrors: download the HF CSVs (see §4.2) into a fresh directory, then
  `ATLASBT_OHLCV_CSV=... ATLASBT_OPTION_CSV=... ATLASBT_EPS_ESTIMATE_CSV=... python -m atlasbt.data.fetch --set bulk`.
* SEC registrant names: `--only sec_submissions` (in `--set core`) before `--only identity`.
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

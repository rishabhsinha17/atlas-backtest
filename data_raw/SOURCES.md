# atlas_bt raw data sources (scout + sampler report)

Compiled 2026-10-28. Everything under `data_raw/` is real, downloaded, unmodified data. The only additions are the `ken_french/extracted/` unzips, the `SOURCE_COMMIT.txt` files that record git provenance, and one metadata CSV pulled from an npm package (noted below). Machine-readable inventory: `inventory.json` (40 entries). Total on disk: about 1.2 GB.

Network context: egress was restrictive at first. Midway through, the integrator said full access had been granted. From then on every request used the User-Agent `atlasbt-research/0.1`, with pauses between calls. Two exceptions: FRED, which refuses that UA, so it got curl's default UA, and SEC, covered below. The scripts ran from the scratchpad with `python3 -I`. No downloaded code was executed.

Legend for `adjusted`: **raw** means as-traded. **split-adj** means split-adjusted only. **total-adj** means adjusted for both splits and dividends.

---

## 1. Coverage by priority (summary)

| Priority | Status | Best on-disk source(s) | Bulk path for DATA agent |
|---|---|---|---|
| 1 Daily OHLCV, ETFs | **Good** for live ETFs. All top-priority ETFs are on disk (SPY back to 1993, sector SPDRs back to 1998, leveraged ETFs from inception) with dividends and splits | `yfinance/daily` (39 symbols), `qlib_us_1d` (8,994 symbols, 2000–2020-11) | yfinance for ETFs. DoltHub `stocks.ohlcv` (2011–now, all listed symbols) |
| 1 Survivorship-free universe | **Partial / gap.** Nothing here is clean CRSP-like data. qlib has only 178 dead names. DoltHub `stocks.symbol` has 24,418 symbols with `last_seen`, which includes delisted ones, but OHLCV only starts in 2011 | DoltHub, plus fja05680 PIT membership | paperswithbacktest Stocks-Daily-Price (7,764 symbols, 1962–2026) is **gated (paid)** |
| 2 Intraday 1m/5m | **Samples only** | yfinance 1m with pre/post (last 7d), Lean SPY minute and tick (Oct 2013), jwigginton 1m S&P 500 (2 weeks of 2024), Dukascopy CFD ticks, IEX TOPS pcap (1 day) | IEX HIST: free pcap for every day since 2016-12 (10–16 GB/day now). Dukascopy ticks since 2017. fabhaus 5m (2024-01..2026-03, 478 GB, not gated). paperswithbacktest 1m (gated) |
| 2 Tick / NBBO | **IEX-only top of book** (not consolidated NBBO). Lean SPY tick fixture. No SIP NBBO found | `iex_hist/20191224_IEXTP1_TOPS1.6.pcap.gz` | IEX HIST |
| 3 Options EOD chains | **One full-market day** (2019-08-15, all US options with bid/ask/IV/greeks). DoltHub chain is a sparse subset (about 3 expiries and 22 strikes per symbol per day, 2019-02..now). Lean SPX/SPY option minute fixtures | `historicaloptiondata_free_samples`, DoltHub `options.option_chain` | DoltHub via `dolt clone post-no-preference/options`. fabhaus 5m option bars (trades only, no quotes) |
| 3 Intraday option quotes | **Fixtures only** (Lean SPX/SPXW 2021-01-04/14 quote minute bars, SPY 2023-08-03) | `gh_quantconnect_lean_data/Data/indexoption`, `.../option` | none found free |
| 4 Rates | **Complete** | FRED DTB3 (1954–), DGS*, EFFR, SOFR, DFF. Treasury CMT and bill rates. NY Fed EFFR/SOFR/OBFR/BGCR/TGCR. KF RF | all on disk |
| 5 Vol indices | **Complete** | Cboe: VIX (1990), VIX9D (2011), VIX3M (2009), VIX6M (2008), VIX1Y, VIX1D (2022), SKEW (1990), VVIX (2006), and others. Strategy indices: PUT, BXM, CNDR, BFLY, etc. | on disk |
| 6 Factors | **Complete** | Ken French daily and monthly 3F/5F/Mom/ST/LT reversal, 49/12/10 industries, 25 size×B/M, 10 momentum portfolios, ME breakpoints, SIC definitions. AQR BAB/QMJ/HML-Devil/UMD daily. q5 factors daily | on disk |
| 6 S&P 500 PIT constituents | **Good** | fja05680 (1996-01-02..2026-08-18, daily membership), Wikipedia changes table (412 rows, 1976..2026-10-06) | on disk |
| 6 Sector/industry | **Snapshot only** (not PIT) | FinanceDatabase (sector/industry/FIGI/CUSIP, delisted flag), Wikipedia GICS, SEC `sic` per CIK, KF Siccodes49 | SEC submissions `sic` |
| 6 Shiller | **Done** | `shiller/ie_data.xls`, `gh_datasets_s-and-p-500` | |
| 7 Fundamentals with filing dates | **Samples.** SEC companyfacts gives `filed` date per fact (AAPL sample) | `sec_edgar/` | data.sec.gov companyfacts for each CIK (10 req/s) |
| 7 Earnings dates + BMO/AMC | **Good** | ZipLime (455k 8-K 2.02 releases 2003–2026 with second-level ET acceptance time and session), DoltHub earnings_calendar (2020–2026 with when=BMO/AMC), DoltHub eps_history (reported vs estimate 2016–), eps_estimate (consensus snapshots 2017–), Nasdaq API sample | |
| 7 Dividends/splits | **Good** | DoltHub dividend (502k rows, 1970–) and split (4,168 rows, 2014–). yfinance actions. EODHD AAPL with declaration/record/pay dates. Lean factor files | |
| 7 Index add/delete announcements | **Partial.** Effective dates only (Wikipedia, fja05680). No announcement dates | | none found |
| 7 FOMC dates/times | **Good** | federalreserve.gov calendars 1994–2026+ (HTML), statement sample with "For release at 2:00 p.m.", vtasca/fomc-statements-minutes CSV (text plus dates, scheduled to 2027) | |
| 7 Buybacks | **Gap.** Only indirect: ZipLime events/ has every 8-K item (Item 8.01 etc.), and companyfacts has repurchase tags | | |
| 7 Short borrow fees | **Gap.** iborrowdesk timed out. FINRA short interest is reachable but its ToS forbid redistribution and ML use (see ZipLime/us-short-interest audit), so it was not downloaded | | |
| 7 Trading halts | **Partial.** Nasdaq halts RSS by date (`haltdate=MMDDYYYY`). 2020-03-09 gave 905 halts; coverage before about 2012 is sparse | `nasdaqtrader/` | loop over dates |

---

## 2. Sources FOUND (downloaded or sampled)

### 2.1 Core bulk sets (downloaded in full)

| Source | URL / clone | Contents & coverage (verified by reading files) | Freq | Fields | Adjusted | Known issues | License |
|---|---|---|---|---|---|---|---|
| **cboe_indices** | `https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYM}_History.csv` | 45 indices. VIX 1990-01-02..2026-10-27. VIX9D 2011-01-04... VIX3M 2009-09-18... VIX6M 2008-01-02... VIX1Y 2007-01-03... VIX1D 2022-05-13... SKEW 1990-01-02... VVIX 2006-03-06... VXN, RVX, VXD, OVX, GVZ, VXEEM, VXTLT (2004), VXEFA, VXSLV, VXGDX, VXAPL, VXAZN, VXGOG, VXIBM, VXGS. Strategy indices: PUT (1991), BXM (2002 file), BXMD, BXY, CNDR, BFLY, CLL, PPUT, PUTY, BXMW, CMBO, RXM (1986). COR1M, COR3M (2006), DSPX (2014), VXTH, SPX (1975), OEX, XEO, DJX, RUT (2020 only) | daily | DATE, OPEN, HIGH, LOW, CLOSE (some files have a single value column) | n/a | Early rows are often O=H=L=C. Dates are MM/DD/YYYY. The VIX3M file starts in 2009, while yfinance ^VIX3M goes back to 2006 | Cboe site terms |
| **fred** | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=ID` | 27 series: DTB3 1954-01-04.., DTB4WK, DTB6, DTB1YR, DGS1MO 2001.., DGS3MO 1981.., DGS6MO, DGS1 1962.., DGS2 1976.., DGS5, DGS10 1962.., DGS30, EFFR 2000-07-03.., SOFR 2018-04-03.., DFF 1954-07-01.., DPRIME, T10Y2Y, T10Y3M, DFII10, T10YIE, VIXCLS, VXVCLS, DTWEXBGS, USREC, CPIAUCSL, BAMLH0A0HYM2/BAMLC0A0CM (last 3y only). All run to 2026-10-26/27 | daily (CPI/USREC monthly) | observation_date, value | n/a | Holidays are blank. Values are revised (use ALFRED for vintages). **FRED rejects a custom UA** (HTTP/2 INTERNAL_ERROR); curl's default UA works | public domain mostly; ICE BAML restricted |
| **ken_french** | `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/<file>.zip` | 25 zips. FF3 daily 19260701..20260930, FF3 monthly 192607..202609, weekly. FF5 daily 19630701..20260930 and monthly. Momentum daily 19261103... ST/LT reversal. 49/12/10 industry (daily and monthly). 25 portfolios 5x5. 10 prior-12-2 portfolios. Size portfolios. ME breakpoints. Siccodes49/12 | daily/weekly/monthly | Mkt-RF, SMB, HML, RMW, CMA, RF, Mom, ... | n/a | Several sub-tables per CSV, separated by text lines. Percent units. -99.99 = missing. RF source switched to ICE BofA 1M T-bill from 2024-06 (per file header) | free with citation |
| **gh_fja05680_sp500** | `git clone https://github.com/fja05680/sp500` | S&P 500 membership on each change date, 1996-01-02..2026-08-18. Ticker start/end table. Changes since 2019. Current list with GICS and CIK | event | date, tickers | n/a | Ticker-based (no permno). The `(Updated)` file maps to current tickers. The other file suffixes dead tickers as `-YYYYMM` | MIT |
| **aqr_factors** | `https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/<name>.xlsx` | BAB, QMJ, HML-Devil daily (sheets: MKT, SMB, HML FF, HML Devil, UMD, RF, factor; one column per country incl. USA). TSMOM monthly. Century of Factor Premia monthly | daily/monthly | DATE + country columns | n/a | Large xlsx files (26–32 MB) | AQR data terms (research use, attribution) |
| **global_q_factors** | `https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_{daily,monthly}_2025.csv` | q5 factors, 1967-01-03..2025-12-31 | daily/monthly | R_F, R_MKT, R_ME, R_IA, R_ROE, R_EG | n/a | Percent | academic use |
| **nyfed_reference_rates** | `https://markets.newyorkfed.org/api/rates/{unsecured/effr,unsecured/obfr,secured/sofr,secured/bgcr,secured/tgcr}/search.csv?startDate=1990-01-01&endDate=...` | EFFR 2000-07-03..2026-10-26 (6,613 rows). OBFR 2016-03-01... SOFR/BGCR/TGCR 2018-04-02.. | daily | rate, percentiles, volume, target range | n/a | Newest first | public |
| **us_treasury** | `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{Y}/all?type=daily_treasury_yield_curve\|daily_treasury_bill_rates&field_tdr_date_value={Y}&_format=csv` | Samples: yield curve 2024–2026, bill rates 2002 and 2024–2026, XML 1990 | daily | 1 Mo..30 Yr. Bill rates: bank discount and coupon equivalent for 4–52 week | n/a | One year per request. The yield curve starts in 1990, bill rates in 2002 | public domain |
| **hf_ziplime_earnings_calendar** | `https://huggingface.co/datasets/ZipLime/earnings-calendar` (rev f8d9be6) | 455,361 8-K Item 2.02 announcements, 2003-04-25..2026-10-23, 12,657 tickers. 262k with EPS/revenue (2009–2026). events/ holds all 8-K items (4.38M, 1993–2026). acceptance.parquet | event | announced_at (UTC, second precision), announced_time_et, session (pre_market/market_hours/after_close/non_session), first_tradeable_session, EPS basic/diluted, revenue, SUE, ticker_confidence | n/a | Ticker mapping carries a confidence flag. No estimates/consensus. The card says EDGAR `submissions` JSON `acceptanceDateTime` is really Eastern time for much of the pre-2021 corpus despite its "Z" suffix (unverified by me; AAPL 2015–2026 rows look like true UTC) | Apache-2.0 |
| **dolthub_post_no_preference** (full small tables) | `https://www.dolthub.com/csv/post-no-preference/<repo>/master/<table>` | earnings_calendar 122,113 rows 2020-01-22..2026-12-04 (when: AMC 50,915, BMO 39,487, NULL 31,711). eps_history 169,386 (period_end 2016-07-31..2026-09-30, reported vs estimate). dividend 502,467 (ex_date 1970-01-19..2026-10-23). split 4,168 (2014-03-28..). us_treasury 9,198 (1990-01-02..2026-10-27) | event/daily | see inventory | ohlcv appears raw | Scraped (Zacks/Nasdaq-style) community data. Split table only starts in 2014 | DoltHub repo (community, no explicit license) |
| **wikipedia** | `https://en.wikipedia.org/w/index.php?title=<page>&action=render` (and `&action=raw`) | S&P 500 current (503 with GICS, date added, CIK). **Historical components changes table: 412 rows, 1976-07-01..2026-10-06** (dense from about 2000). S&P 400, S&P 600, Nasdaq-100 pages | event | Effective Date, Added (Ticker, Security), Removed (Ticker, Security), Reason | n/a | Effective dates only, no announcement dates. REST API `/api/rest_v1` gave 429 for our UA. SP600 raw gave 429 (render worked) | CC BY-SA |
| **federalreserve_fomc** | `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`, `fomchistorical{1994..2020}.htm`, `/feeds/press_monetary.xml` | Meeting calendars 1994–2026+. 47 statement links on the current calendar page. Sample statements: 2024-01-31 says "For release at 2:00 p.m. EST". 2005 boarddocs page shows only the release date | event | HTML | n/a | Needs parsing. Release time is explicit only on newer pages (2:15 pm was the norm before 2013) | public domain |
| **qlib_us_1d** | GitHub release `https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_us_1d_latest.zip` (450 MB) | 8,994 instruments incl. SPY/QQQ/IWM/XLK/TLT/GLD/TQQQ/UPRO/BIL/SHV/XLC/XLRE. Calendar 1999-12-31..2020-11-10. Instruments for sp500 (755 rows with intervals) and nasdaq100 | daily | open, high, low, close, volume, factor, change | **total-adj and normalized** (first close = 1). Raw = value / factor (SPY 2020-11-10 → 354.0 ✓) | Yahoo-sourced. Only 178 instruments end before the last date, so it is effectively survivorship-biased. Ends 2020-11. float32 | Qlib (MIT), data from Yahoo |
| **yfinance** | `pip install yfinance` (1.7.0), `Ticker.history(period='max', auto_adjust=False, actions=True)` | Daily for 39 symbols: SPY 1993-01-29.., QQQ 1999-03-10, IWM 2000-05-26, DIA, 11 sector SPDRs (1998-12-22; XLRE 2015-10-08, XLC 2018-06-19), RSP 2003, TLT/IEF/SHY 2002-07-30, BIL 2007, SHV 2007, GLD 2004, EFA 2001, VNQ 2004, SSO/QLD/SH 2006-06-21, SDS 2006-07-13, UPRO 2009-06-25, TQQQ 2010-02-11, AGG, LQD, HYG, EEM, ^GSPC 1927, ^VIX, ^VIX9D, ^VIX3M (2006), ^IRX 1960. All to 2026-10-27. 1-minute with pre/post (04:00–19:59 ET) for SPY/QQQ/AAPL, 2026-10-19..27. SPY option chain snapshot (2 expiries) | daily, 1m | OHLC, Adj Close, Volume, Dividends, Stock Splits, Capital Gains | OHLC **split-adj** (not dividend-adj). Adj Close is total-adj | Yahoo re-adjusts history after each split, so raw prices must be rebuilt from Stock Splits. 1m data is kept for only about 30 days. No delisted symbols | Yahoo ToS (personal use) |

### 2.2 Samples (small real extracts; full source documented for the DATA agent)

| Source | URL / endpoint | What the sample is | Full coverage & schema | Adjusted | Issues / rate limits | License |
|---|---|---|---|---|---|---|
| **dolthub (remote tables)** | SQL API `https://www.dolthub.com/api/v1alpha1/post-no-preference/{options,earnings,stocks,rates}/master?q=<SQL>` | `sample_options_option_chain_SPY_2024-01-03.json` (104 rows), `sample_stocks_ohlcv_10sym_2024-01-02_08.json`, `sample_options_volatility_history_2024-01-03.json`, `sample_earnings_eps_estimate_AAPL.json` | **options.option_chain** 2019-02-09..2026-10-27: date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho. Sparse: about 3 expiries × 22 strikes per underlying; at least 506 underlyings on 2024-01-03 (the DISTINCT query timed out after 506). **volatility_history**: hv/iv current, week ago, month ago, 52w hi/lo. **stocks.ohlcv** 2011-01-03..2026-10-27 (date, act_symbol, OHLCV). **stocks.symbol** 24,418 symbols incl. `last_seen` (delisted). **earnings**: eps_estimate (2017-10-26..), income_statement, balance_sheet_*, cash_flow_statement, rank_score, sales_estimate | ohlcv looks raw | Max 1000 rows per query. Queries time out after ~20–60 s; aggregates and non-PK filters on big tables fail, so filter by `date` first. CSV export of `stocks.symbol` returned 504. For bulk, `dolt clone` | community |
| **sec_edgar** | `https://data.sec.gov/submissions/CIK##########.json`, `/api/xbrl/companyfacts/CIK...json`, `/api/xbrl/companyconcept/...`, `/api/xbrl/frames/...` | AAPL and MSFT submissions. AAPL companyfacts (503 us-gaap tags). AAPL EPS-diluted concept. Frames EPSDiluted CY2023Q4 | Submissions: `filings.recent` (about 1,000 most recent, older ones in `filings.files[]`), with accessionNumber, filingDate, reportDate, **acceptanceDateTime**, form, **items** (e.g. "2.02,9.01"), primaryDocument. AAPL has 45 8-K/2.02 rows in recent filings, e.g. 2026-07-30T20:30:28Z. companyfacts gives each fact with start, end, val, accn, fy, fp, form, **filed**, frame (PIT-able) | n/a | data.sec.gov accepted UA `atlasbt-research/0.1`. **www.sec.gov (company_tickers.json, Archives/full-index) returned 403** without a contact email in the UA. SEC policy needs "Name email" in the UA; the DATA agent should confirm the contact with the user. 10 req/s max | public domain |
| **iex_hist** | Index `https://iextrading.com/api/1.0/hist` (all dates) or `?date=YYYYMMDD`. Files on GCS bucket `iex` | `20191224_IEXTP1_TOPS1.6.pcap.gz` (160 MB, half day; verified SPY QuoteUpdate 13:34:55 UTC bid 2000@321.55) plus the full index JSON | 2,498 dates 2016-12-12..2026-10-27. TOPS 1.5/1.6 (top of book and trades), DEEP from 2017-05-15 (depth), DPLS from 2024-10-01. Recent days are 10–16 GB per feed | raw | IEX-only quotes and trades, **not consolidated NBBO**. pcapng + IEX-TP parsing needed | IEX terms (free, attribution) |
| **dukascopy** | `https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5` and `{BID,ASK}_candles_min_1.bi5` (month is 0-based) | SPYUSUSD, QQQUSUSD, USA500IDXUSD on 2024-01-16: hourly ticks for 14–20 UTC (23/27 OK) plus BID/ASK 1-min candles. `instrument_metadata_from_dukascopy-node-1.51.0.csv`: 666 US stock/ETF/index CFDs with first-tick timestamps, extracted from the MIT npm package `dukascopy-node` | US stock/ETF CFD ticks start about 2017-01-26 (SPY 2017-02-16, XLK/XLE/XLU/XLV/XLY 2017-11-15). USA500.IDX ticks start 2012-01-16. Tick record: >IIIff = ms, ask, bid, askvol, bidvol (LZMA). Price/1000 for US CFDs. Candle record: >IIIIIf = sec, O, C, L, H, vol | raw | **Dukascopy CFD quotes, not exchange data**. RTH-only for stock CFDs. Frequent 503s and timeouts, so pace ≥2 s and retry. "startDayForMinuteCandles 2000-01-01" in the metadata is a placeholder | Dukascopy terms |
| **gh_quantconnect_lean_data** | `git clone https://github.com/QuantConnect/Lean` (sparse `Data/`) | equity/usa/daily (~25 symbols, SPY 1998-01-02..2021-03-31). SPY minute trade+quote for 7 days in Oct 2013 (04:00–20:00). SPY tick. factor_files/map_files (~30). 11 coarse whole-market daily universe files (2014-03). SPY option minute 2023-08-03. SPX index option minute 2021-01-04/14 (quote/trade/OI). AAPL 2014 daily option quotes. SPX index daily | daily/min/tick | Equity prices are integers ×10000. Quote bars carry bid OHLC + size and ask OHLC + size | raw + factor files | Test fixtures only (few days) | Apache-2.0 (data from AlgoSeek/QC) |
| **historicaloptiondata_free_samples** | `https://historicaloptiondata.com/free-data/` (Dropbox links) | **Sample_L3_20190815.zip**: every US listed option for 2019-08-15 (137 MB CSV; bid, ask, last, volume, OI, IV bid/ask/mean, delta, gamma, theta, vega) + optionstats (IV30..IV360 call/put/mean, volumes, OI) + stockquotes. stockhistory_sample.zip (2019 EOD stocks, dividends, splits) | Paid product covers 2002– | raw | SPX/SPXW Dec-2023 CSVs and the hourly sample: Dropbox served HTML, not data | free sample from Delta Neutral |
| **hf_samples** | `https://huggingface.co/datasets/<id>/resolve/main/<file>` | jwigginton/timeseries-1mn-sp500 (full 11 MB: 503 S&P 500 stocks, 1m RTH, 2024-02-26..2024-03-11). fabhaus/equities_5m_options 2024-01 shard (full month). fabhaus/equities_5m_stockprices (first 4 MB, truncated). thillsss SPX 1m (first 2 MB, from 2008-01-02 CT). gauss314 options-IV-SP500 (first 2 MB, from 2019-10-14). CodyJiang NVDA chains (first 2 MB). Pinxitu (first 2 MB, actually ES futures). papylove (actually 1m stock bars for 10 symbols). vtasca FOMC communications CSV (full) | see section 4 | varies | Truncated files end mid-line | per dataset (see section 4) |
| **eodhd_demo** | `https://eodhd.com/api/{eod,splits,div,intraday,fundamentals,calendar/earnings}/AAPL.US?api_token=demo` | AAPL raw OHLC + adjusted_close 1980-12-12..2026-10-27. Splits. Dividends with declaration/record/payment dates. 1m intraday 2024-01-02. Fundamentals JSON. Earnings calendar with before_after_market | AAPL only on the demo key (SPY 403, options 403) | raw + adj_close | demo key is for evaluation only | EODHD ToS |
| **twelvedata_demo** | `https://api.twelvedata.com/time_series?symbol=AAPL&interval=1min&outputsize=5000&apikey=demo` | AAPL last 5,000 1-min bars | AAPL only (SPY 401) | raw | demo only | Twelve Data ToS |
| **nasdaq_api** | `https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD`, `/api/quote/{SYM}/historical?assetclass=etf&fromdate=..&todate=..&limit=9999` | Earnings calendar 2024-01-25 (117 rows: time, EPS, surprise, consensus, #ests). SPY daily OHLCV for the last 10y (2,513 rows) | historical capped at about 10y | unknown | Flaky (timed out earlier, worked on retry). Nasdaq ToS restrict automated use | Nasdaq ToS |
| **nasdaqtrader** | `https://www.nasdaqtrader.com/dynamic/symdir/{nasdaqlisted,otherlisted}.txt`, `https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts[&haltdate=MMDDYYYY]` | Current symbol directories (with ETF flag, test issue). Halts RSS for now, 2020-03-09 (905 items) and 2010-01-04 (1 item) | Halts by date, sparse early | n/a | Pipe-delimited files with a footer line | Nasdaq |

### 2.3 Smaller / reference sets (downloaded)

| Source | URL | Contents | Range | Notes / license |
|---|---|---|---|---|
| gh_datasets_finance-vix | github datasets/finance-vix | VIX daily OHLC | 1990-01-02..2026-09-22 | Cboe mirror, PDDL |
| gh_datasets_s-and-p-500 | github datasets/s-and-p-500 | Shiller monthly: SP500, Dividend, Earnings, CPI, Long rate, PE10 | 1871-01..2026-09 | PDDL |
| gh_datasets_bond-yields-us-10y | github datasets/bond-yields-us-10y | 10y yield monthly | 1953-04..2026-08 | PDDL |
| gh_datasets_s-and-p-500-companies(-financials) | github datasets/... | Current constituents with GICS sector; snapshot of financial ratios | snapshot | PDDL |
| gh_datasets_nasdaq-listings, nyse-other-listings | github datasets/... | Current listings | snapshot | PDDL |
| shiller | `https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls` | Shiller ie_data (P, D, E, CPI, GS10, CAPE) | 1871..2026 | academic |
| damodaran | `https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls` | Annual S&P/T-bill/T-bond returns | 1928..2025 | academic |
| hilpisch_eikon_eod | `https://hilpisch.com/tr_eikon_eod_data.csv` | Closes for AAPL, MSFT, INTC, AMZN, GS, SPY, .SPX, .VIX, EUR=, XAU=, GDX, GLD | 2010-01-01..2018-06-29 | textbook data, cross-check only |
| gh_cran_qrmdata | github cran/qrmdata (sparse) | SP500_const (505 then-current S&P stocks, adj close + sector), SP500, VIX, ZCB_USD (1–30y zero curve), DJ_const | about 1995..2015-12 | GPL. xts objects; pyreadr drops the date index, so read with R. Survivorship-biased |
| gh_cran_PCRA | github cran/PCRA (sparse) | CboeOptionStrategies (monthly BXM, PUT, CNDR, ..., SPTR, VIX, VXO, GS3M), FRBinterestRates | 1986-06..2021-12 | GPL. **The CRSP/SPGMI-derived files in PCRA were deliberately not kept**: their docs forbid redistribution |
| gh_JerBouma_FinanceDatabase | github JerBouma/FinanceDatabase (sparse) | US equities by exchange (NYQ, NMS, NGM, NCM, ASE, PCX, BTS): sector, industry_group, industry, ISIN, CUSIP, FIGI, market-cap bucket, delisted flag | snapshot | MIT. Not PIT |
| gh_matplotlib_mplfinance_sampledata | github matplotlib/mplfinance | S&P 500 index 1-min, 5 days Nov 2019. SPY daily 2008–2018 | | BSD |

### 2.4 Short samples (first rows, read from the files)

```
cboe_indices/VIX9D_History.csv
DATE,OPEN,HIGH,LOW,CLOSE
01/04/2011,16.060000,16.060000,16.060000,16.060000

fred/DTB3.csv
observation_date,DTB3
1954-01-04,1.33

ken_french/extracted/F-F_Research_Data_Factors_daily.csv
,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01

yfinance/daily/SPY.csv
Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits,Capital Gains
1993-01-29 00:00:00-05:00,43.96875,43.96875,43.75,43.9375,24.053539276123047,1003200,0.0,0.0,0.0

yfinance/intraday_1m/SPY_1m_7d_prepost.csv
2026-10-19 04:00:00-04:00,767.28,769.81,767.17,767.9,767.9,0,0.0,0.0,0.0

qlib_us_1d  instruments/all.txt: "SPY\t1999-12-31\t2020-11-10"; features/spy/close.day.bin float32 [start_idx=0, 1.0, 0.990, ...]; factor 0.0068..0.0100

dolthub earnings__earnings_calendar.csv
act_symbol,date,when
A,2020-02-18,After market close

dolthub option_chain (SQL sample)
{'date': '2019-02-09', 'act_symbol': 'A', 'expiration': '2019-02-15', 'strike': '65.00', 'call_put': 'Call', 'bid': '10.50', 'ask': '11.25', 'vol': '0.2705', 'delta': '1.0000', ...}

hf_ziplime_earnings_calendar earnings/part-00000.parquet
ticker  announced_at               session     first_tradeable_session period_end eps_diluted revenue
AIR     2010-09-15 20:44:21+00:00  after_close 2010-09-16              2010-08-31 0.35        412197000.0

sec_edgar submissions AAPL 8-K/2.02:
('2026-07-30', '2026-07-30T20:30:28.000Z', '8-K', '2.02,9.01', '2026-07-30')

us_treasury/daily_treasury_yield_curve_2024.csv
Date,"1 Mo","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
12/31/2024,4.40,4.39,4.37,4.32,4.24,4.16,4.25,4.27,4.38,4.48,4.58,4.86,4.78

nyfed_reference_rates/effr_search.csv
10/26/2026,EFFR,3.88,3.85,3.88,3.89,3.94,109,3.75,4.00,...

global_q_factors/q5_factors_daily_2025.csv
date,R_F,R_MKT,R_ME,R_IA,R_ROE,R_EG
1967-01-03,0.0187,0.0778,0.4891,0.1399,-0.7193,-0.8845

wikipedia Historical_components_of_the_SP500 (changes table)
October 6, 2026 | TWLO Twilio | WBD Warner Bros. Discovery | Paramount Skydance acquired Warner Bros. Discovery

gh_fja05680_sp500/sp500_ticker_start_end.csv
ticker,start_date,end_date
AABA,1999-12-08,2017-06-19

historicaloptiondata L3_options_20190815.csv
UnderlyingSymbol,UnderlyingPrice,Flags,OptionSymbol,Type,Expiration, DataDate,Strike,Last,Bid,Ask,Volume,OpenInterest,T1OpenInterest,IVMean,IVBid,IVAsk,Delta,Gamma,Theta,Vega,AKA
A,69.59,*,A190816C00042500,call,08/16/2019,08/15/2019,42.5,0,25.05,29.5,0,0,0,1.0356,0,16.8732,1,0,-0.9762,0,A190816C00042500

dukascopy SPYUSUSD 2024-01-16 15h (ms, ask, bid, askvol, bidvol): (18, 474513, 474487, 0.0112, 0.0132) -> 474.513/474.487

Lean equity/usa/minute/spy/20131007_quote.zip (ms since midnight, bid O/H/L/C, bidsize, ask O/H/L/C, asksize; price x10000)
14400000,1672700,1672700,1671500,1672500,300,1680000,1680000,1672600,1672700,20000

IEX TOPS 2019-12-24: SPY QuoteUpdate 13:34:55.892 UTC bid 2000 @ 321.55

hf_samples/jwigginton_timeseries-1mn-sp500
A 2024-02-26 09:30:00 131.779999 132.119995 131.649994 131.910004 24137.0

hf_samples/fabhaus_equities_5m_options/2024-01.jsonl
{"close": 36.5, "datetime": "2024-01-18T14:30:00Z", "expiration_date": "2026-06-18", "option_symbol": "AAPL260618C00185000", "strike_price": 185.0, "trade_count": 2, ...}
```

---

## 3. Reachable but NOT downloaded (documented for later)

| Candidate | Status | Notes |
|---|---|---|
| paperswithbacktest/Stocks-Daily-Price (HF) | 200 metadata, **gated=manual** | 7,764 symbols, 1962-01-02..2026-08-05, OHLCV + adj_close, 26M rows. Gate requires a paid PwB subscription. Same org: ETFs-Daily-Price, Indices, Bonds, Stocks-1Min-Price (2010–2026, 5.7B rows, 76 GB), Stocks-Quarterly-Earnings, Daily-Dividends, Semimonthly-ShortInterest, Daily-EdgarFilings, Daily-MarketCap — all gated |
| d-massive/flat_files_us_stocks_sip (HF) | gated=manual | 0.9 GB parquet of SIP day aggregates (Massive/Polygon). License unknown |
| fabhaus/equities_5m_stockprices (HF) | open, license "other" | 5m bars for about 13.3k US tickers, 2024-01..2026-03, 478 GB jsonl (sampled 4 MB) |
| fabhaus/equities_5m_options (HF) | open | 5m option OHLCV/VWAP (trades), 2024-01..2026-03, 60 GB (1 month downloaded) |
| siddharthmb/stocks-options, stocks-ohlcv, stocks-earnings-* (HF) | open | CSV mirrors of the DoltHub post-no-preference tables (option_chain.csv 9.1 GB, ohlcv.csv 1.2 GB), last updated 2026-01. Easier bulk than DoltHub |
| younginpiniti/us-stocks-daily-all (HF) | open, no license | 0.6 GB parquet chunks |
| gauss314/options-IV-SP500 (HF) | Apache-2.0, 523 MB | Daily IV surface summary per S&P 500 name |
| thillsss/SPX-MES-VIX-data (HF) | no license, 310 MB | SPX 1m from 2008 (CT), MES/MNQ 1m |
| CodyJiang/nvda-option-chains (HF) | no license | OptionsDX-format NVDA chains 2020–2022; probably redistributed vendor data |
| skfolio/skfolio-datasets (GitHub) | clone OK | nasdaq_dataset.csv.gz (about 1,455 Nasdaq stocks daily), sp500_implied_vol_dataset.csv.gz, ftse100 |
| scikit-learn/examples-data (GitHub) | clone OK | financial-data/*.csv: 56 large caps daily OHLCV 2003–2007 |
| stefan-jansen/zipline-reloaded (GitHub) | clone OK | resources/market_data/treasury_curves.csv, SPY_benchmark.csv. tests/resources/quandl_samples/QUANDL_ARCHIVE.zip (WIKI sample with split/div) |
| Hvass-Labs/FinanceOps, mementum/backtrader, gbeced/pyalgotrade, robertmartin8/PyPortfolioOpt, rreichel3/US-Stock-Symbols (GitHub) | clone OK | Small textbook/test datasets (Yahoo daily per-year CSVs, backtrader 5-min sample, etc.) |
| home.treasury.gov full history | 200 | Loop years 1990–2026 for the yield curve and 2002–2026 for bill rates |
| www.treasurydirect.gov TA_WS auctions | 200 | Auction results JSON |
| cftc.gov COT history | 200 | `fut_fin_txt_YYYY.zip` |
| cdn.finra.org RegSHO daily short volume | 200 | **Not downloaded.** FINRA ToS prohibit redistribution and ML/AI use (see ZipLime/us-short-interest licensing audit) |
| alphaquery.com, multpl.com, ivolatility.com, jkpfactors.com, optionsdx.com, barchart.com (202), cdn.quantconnect.com | reachable | No direct free bulk file found or tried. Site scraping not attempted |

---

## 4. Candidates TRIED that FAILED

| Candidate | URL | Result |
|---|---|---|
| Financial Modeling Prep demo | `financialmodelingprep.com/api/v3/quote/AAPL?apikey=demo` | 401 |
| Kibot free IBM 1-min (guessed path) | `www.kibot.com/Files/2/IBM_adjusted.txt` | 404 |
| Hilpisch over plain HTTP | `http://hilpisch.com/...` | 000 (proxy is HTTPS-only; HTTPS worked) |
| Shiller at Yale (HTTP) | `http://www.econ.yale.edu/~shiller/data/ie_data.xls` | 000 (wsimg copy worked) |
| Pastor-Stambaugh liquidity (guessed path) | `faculty.chicagobooth.edu/lubos.pastor/research/liq_data_1962_2023.txt` | 404 |
| Bokeh sample data | `bokeh-sampledata.s3.amazonaws.com`, `sampledata.bokeh.org` | 403 / 403 |
| Qlib Azure blob | `qlibpublic.blob.core.windows.net/...` | 409 (public access disabled; GitHub release worked) |
| GitHub web UI release page | `github.com/chenditc/investment_data/releases` | 403 (release asset downloads do work) |
| GitHub LFS media | `media.githubusercontent.com/media/datasets/finance-vix/...` | 404 (not an LFS file; inconclusive) |
| Deutsche Börse PDS bucket | `deutsche-boerse-xetra-pds.s3.amazonaws.com` | 403 (and not US data) |
| OFR short-term funding API (guessed mnemonic) | `data.financialresearch.gov/v1/series/timeseries?mnemonic=FNYR-EFFR_A-A` | 400 |
| iShares holdings CSV (current and `asOfDate=` 2006/2015/2024) | `www.ishares.com/us/products/{id}/...ajax?fileType=csv...` | 200 but HTML page, not CSV |
| Yahoo raw chart endpoint | `query1.finance.yahoo.com/v8/finance/chart/SPY` | 429 (yfinance library works) |
| Quandl / Nasdaq Data Link | `www.quandl.com` → `data.nasdaq.com` | proxy CONNECT 403 |
| Google | `www.google.com` | proxy CONNECT 403 |
| investing.com / marketwatch / wsj / macrotrends / slickcharts | | 403 / 401 / 403 / 403 / 403 |
| stooq (twice) | `stooq.com/q/d/l/?s=spy.us&i=d` | 000 timeout |
| iborrowdesk (twice) | `iborrowdesk.com/api/ticker/GME` | 000 timeout |
| FRED with custom UA | `fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3` | HTTP/2 INTERNAL_ERROR / empty reply (default curl UA OK) |
| SEC www host without contact email | `www.sec.gov/files/company_tickers.json`, `/Archives/edgar/full-index/2024/QTR1/master.idx` | 403 |
| Wikipedia REST API | `en.wikipedia.org/api/rest_v1/page/html/...` | 429 (index.php render/raw OK). SP600 raw 429 |
| DoltHub aggregate SQL on big tables | `SELECT MIN(date)... FROM ohlcv/option_chain` | query timeout ("context deadline exceeded") |
| DoltHub CSV export stocks.symbol | `www.dolthub.com/csv/post-no-preference/stocks/master/symbol` | 504 stream timeout |
| Dukascopy instruments JSON | `freeserv.dukascopy.com/2.0/index.php?path=common/instruments` | 403 (used the npm package metadata instead) |
| Dukascopy hourly files | various | intermittent 503 / timeouts (e.g. AAPLUSUSD, MSFTUSUSD, VOLIDXUSD, DOLLARIDXUSD all 503 during probing) |
| Cboe index files | VXST, VXV, COR9D, SPXEW, NDX `_History.csv` | 403 (not published under those names) |
| historicaloptiondata Dropbox CSVs | SPX_20231201_to_20231229.csv, SPXW_..., Sample_HourlyOptions_20251024_1415.csv | 200 but Dropbox HTML page (dl=1 and raw=1) |
| EODHD demo | SPY.US eod, AAPL options | 403 / 403 |
| Twelve Data demo | SPY daily | 401 |
| HF paperswithbacktest/*, d-massive/* | file download | gated (manual approval / paid) — not attempted beyond metadata |
| ZipLime/us-short-interest (HF) | | publishes no data by design (licensing audit) |

About 150 distinct URLs and repos were probed in total. The full probe log is in `data_raw/_probe_log_first_sweep.tsv` (URL, HTTP code, bytes, content type, final URL) for the first sweep. Everything else is recorded above.

---

## 5. Biggest gaps / recommendations

1. **Survivorship-free daily equities with delisted names (pre-2011):** no free clean source was found. Options: (a) DoltHub `stocks.ohlcv` + `symbol.last_seen` for 2011+. (b) the gated paperswithbacktest dataset, if the user subscribes. (c) EODHD or Norgate-type paid data. qlib and yfinance are survivorship-biased.
2. **Raw (unadjusted) ETF prices:** yfinance OHLC is split-adjusted. Rebuild raw prices with the `Stock Splits` column, or cross-check against Lean (raw + factor files, SPY to 2021), EODHD (raw, AAPL only) or the DoltHub ohlcv (appears raw).
3. **Consolidated NBBO / SIP intraday:** none free. IEX TOPS (IEX-only BBO) and Dukascopy CFD quotes are the only tick sources. For 1m bars beyond 30 days: fabhaus 5m (2024+), paperswithbacktest 1m (gated), or a paid vendor.
4. **Historical option chains:** full-market EOD exists only as one free day (2019-08-15). DoltHub's chain is sparse (about 3 expiries and about 22 strikes) but spans 2019–2026 and includes bid/ask/IV/greeks, which may be enough for ATM/monthly strategies. No free intraday option *quotes* beyond Lean fixtures.
5. **Short borrow fees, buyback announcements, index-change announcement dates:** not found free. FINRA short data exists but its licence forbids redistribution and ML use.
6. **SEC bulk:** data.sec.gov works. The DATA agent needs a declared UA with a contact email (user to confirm) for www.sec.gov endpoints and should stay under 10 req/s. Mind the ZipLime warning about the time zone of `acceptanceDateTime`.

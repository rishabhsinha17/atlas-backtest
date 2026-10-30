# Project status

Last updated: 2026-10-30.

## Summary

| Area | State |
|---|---|
| Engine, execution, accounting, margin, sizing | complete; red-teamed (35 verified findings fixed) |
| Data layer (point-in-time, identity, adjustments) | complete; red-teamed twice (15 findings, then an independent identity audit; all fixed) |
| Options (pricing, chains, lifecycle, margin) | complete; red-teamed (10 findings fixed) |
| Calibrated spread model | complete; red-teamed (9 findings fixed) |
| Indicators, validation statistics, reporting | complete; red-teamed (14 confirmed findings fixed) |
| 172 strategies | all implemented; every family conformance-reviewed against the sheet and fixed |
| Test suite | about 1,900 tests (offline); bit-for-bit golden tests for the performance work |
| End-to-end replications | pass (below) |
| Full atlas run (all testable strategies) | **not finished.** It needs about 1–2 days on 4 cores, and the build container was recycled when idle. Run it with `python -m atlasbt.run launch --commit HEAD --no-overlay final --workers 4`. |

## How it was verified

Every subsystem went through the same loop. A builder agent implemented it. An independent red-team agent,
which had not written the code, attacked it and wrote failing tests. A separate verifier agent then tried to
invalidate each finding against the papers, exchange rules and the sheet. Only confirmed findings were fixed,
and the red-team tests stay in `tests/redteam/` as regressions.

| Review | Findings → outcome |
|---|---|
| Analytics (indicators, DSR/PBO/SPA, metrics) | 15 → 14 confirmed and fixed, 1 invalid |
| Engine: money (accounting, corporate actions, margin, sizing) | 23 → 22 confirmed and fixed (incl. option strikes not adjusted on splits; marketable limits bypassing Reg-T), 1 invalid |
| Engine: time and execution | 13 → 13 confirmed and fixed (incl. reusing a bar's favourable extreme after a same-bar entry) |
| Options library | 10 → 10 confirmed and fixed (incl. SPX AM/PM legs mixed in one package) |
| Data layer | 15 → fixed (reused tickers served another company's prices; fake splice returns; unadjusted alias splits) |
| Independent identity re-audit | engine path resolved identity at the run's end date → redesigned to resolve identity per bar date, keyed by `TICKER~CIK` |
| Spread model | 9 → fixed (pre-2010 small-cap costs biased low; venue tick schedules; liquid-ETF intraday profile) |
| Conformance vs sheet: intraday / swing / calendar+allocation / factors+stat-arb / options events / options income | 5 / 9 / 10 / 11 / 11 / 10 findings → fixed |
| End-to-end runner (cross-layer) | 12 bugs → fixed (incl. quote-only fills on sample days; Corwin-Schultz spreads 30× too high; OCC two-digit-year decoding pre-2000 expiries as 2091) |

## Replications (known answers; full detail in `REPLICATION.md`)

| Check | Result |
|---|---|
| SPY buy & hold 1993–2026, dividends reinvested, vs adjusted-close total return | −0.02%/yr tracking difference |
| SPY buy & hold, dividends as cash | ledger reconciles to < 1 bp of NAV |
| 60/40 SPY/IEF vs independent pandas calculation | correlation 1.000, −0.08%/yr |
| Cboe PUT index vs engine put-write (synthetic SPX chain) | correlation 0.93, 6.0% vs 7.45%/yr, TE 4.1%/yr |
| 12-2 momentum decile spread vs Fama-French UMD | correlation 0.89 (320 months) |
| Book/market spread vs Fama-French HML | correlation 0.82 (195 months) |
| Random coin-flip entries on SPY | gross −1.0 bp (t = −0.32); 0 fills better than mid; 1.0 bp per round trip vs 0.3 bp quoted half-spread |
| Perfect-foresight strategy | blocked by the lookahead guard on every path |

## Testability

126 of 172 strategies are testable on free data (`plan.csv`). Expected fidelity for the testable ones: 5 at T0,
81 at T1, 38 at T2, 2 at T3. The 46 not testable, by cause:

| Cause | Strategies | What unlocks them |
|---|---|---|
| No multi-year 1-minute or 5-minute bars for stocks | S001–S014, S016–S022, S027, S030, S039, S040 | Alpaca key (free SIP minute bars since 2016) |
| Intraday decision times (15:45/15:50) need 1-minute bars across the ETF universe | S031–S037 | Alpaca key; or the opt-in daily proxy (T2) these strategies provide |
| Quotes required (09:29 / 15:25 midpoints) | S015 | Alpaca key |
| Intraday option quotes required | S158–S160, S164, S171 | Databento OPRA |
| SPX minute bars / SPX quotes required | S161–S163, S172 | Databento / Cboe data |
| No free point-in-time source | S041 (analyst revisions), S045 (buyback announcements), S110 (iNAV) | paid data |
| Missing ETF history in cache | S108 (Vanguard sector ETFs) | add them to the fetch set |

## Known limitations

* **Survivorship.** Verified S&P 500 price coverage is 35% (1996), 60% (2008), 84% (2015) and ≥ 96% from 2018.
  Primary stock results use windows with ≥ 95% coverage (2018+). Earlier "extended" runs are flagged
  `EXTENDED_SURVIVORSHIP_BIASED` and never ranked.
* **Options.** Free real chains (DoltHub) start in 2020, are sparse (about 3 expiries per day) and lack
  SPX/QQQ/IWM. Earlier or missing contracts are priced from a synthetic surface (T3), which runs about
  1.5%/yr rich against the Cboe PUT index.
* **Spreads.** The model is fitted to 2026 consolidated quotes. Earlier eras are anchored to published studies
  (Hasbrouck 2009, Hendershott-Jones-Menkveld, Furfine), with a deliberately pessimistic floor for 2010–2025.
* **Sectors and industries.** These are today's classification applied historically (flagged T2).
* **Borrow fees.** No history; a default schedule is used (flagged T2).
* **Key-gated providers** (Alpaca, Polygon, Tiingo, Databento, Alpha Vantage) were written from official docs
  but not live-tested.

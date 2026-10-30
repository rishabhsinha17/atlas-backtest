"""Historical regulatory / clearing fee rate tables (dated, sourced).

Every table is a list of ``(effective_date, values...)`` rows sorted by date;
``RateTable.at(date)`` returns the row in force on ``date`` (the last row whose
effective date is <= date). Dates are the *charge/trade dates* on which a new
rate applies.

Sources (retrieved 2026-10-28 with User-Agent
'atlasbt-research/0.1'):

* SEC Section 31: sec.gov "Fee Rate Advisories" index
  (https://www.sec.gov/rules-regulations/fee-rate-advisories) and each linked
  transaction-fee advisory / press release (2002-30, 2003-27, 2003-57, 2004-24,
  2004-59 ... 2026-2). Pre-2002 statutory rate: 1/300 of one percent
  (Investor and Capital Markets Fee Relief Act, Pub. L. 107-123, set
  $15 per $1,000,000 retroactive to 2001-12-28, SEC press release 2002-7).
* FINRA Trading Activity Fee: FINRA By-Laws Schedule A Section 1 and every
  dated revision served at finra.org/rev/<id> (2008-01-01 .. 2027+), NASD
  Notice to Members 04-84 (2004-11-01 change).
* OCC clearing fee: SR-OCC-2016 (flat fee), SR-OCC-2019 ($0.05 -> $0.055 on
  2019-04-01), SR-OCC-2020-009 ($0.045 from 2020-09-01), SR-OCC-2021 ($0.02
  from 2021-06-01), SR-OCC-2024 ($0.025 from 2025-01-01).
* Cboe SPX/SPXW customer fees: Cboe Exchange Fees Schedule dated
  2026-10-01 (cdn.cboe.com/resources/membership/Cboe_FeeSchedule.pdf).
* IBKR Pro commission schedule: interactivebrokers.com/en/pricing/
  commissions-stocks.php and commissions-options.php (retrieved 2026-10-28).

Rows flagged ``approx=True`` are approximations (documented in docs/ENGINE.md);
fee lines that rely on them should be treated as T2 inputs.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


def _d(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


@dataclass(frozen=True)
class RateRow:
    effective: pd.Timestamp
    values: dict
    source: str = ""
    approx: bool = False


class RateTable:
    """Step-function lookup by effective date."""

    def __init__(self, name: str, rows: Sequence[RateRow]):
        rows = sorted(rows, key=lambda r: r.effective)
        self.name = name
        self.rows = list(rows)
        self._dates = [r.effective.value for r in self.rows]

    def row_at(self, date) -> RateRow:
        d = _as_naive_date(date)
        i = bisect.bisect_right(self._dates, d.value) - 1
        if i < 0:
            return self.rows[0]
        return self.rows[i]

    def at(self, date, key: str) -> Any:
        return self.row_at(date).values[key]

    def __len__(self) -> int:
        return len(self.rows)


def _as_naive_date(date) -> pd.Timestamp:
    t = pd.Timestamp(date)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


# --------------------------------------------------------------------------
# SEC Section 31 (USD per $1,000,000 of covered sales)
# --------------------------------------------------------------------------
_S31 = [
    ("1990-01-01", 1e6 / 300 / 100, "Exchange Act s.31 statutory 1/300 of 1% (pre Fee Relief Act)", True),
    ("2001-12-28", 15.00, "Pub.L. 107-123 / SEC PR 2002-7 ($15 per $1M, retroactive 2001-12-28)", False),
    ("2002-04-01", 30.10, "SEC PR 2002-30 (mid-year adjustment)", False),
    ("2003-03-22", 25.20, "SEC PR 2003-27 (Fee Rate Advisory #12 FY2003)", False),
    ("2003-04-01", 46.80, "SEC PR 2003-27 (mid-year adjustment)", False),
    ("2004-02-22", 39.00, "SEC PR 2004-10 (FY2004 advisory #7)", False),
    ("2004-04-01", 23.40, "SEC PR 2004-24 (mid-year adjustment)", False),
    ("2005-01-07", 32.90, "SEC PR 2004-167", False),
    ("2005-04-01", 41.80, "SEC PR 2005-22 (mid-year adjustment)", False),
    ("2005-12-22", 30.70, "SEC PR 2005-161/163", False),
    ("2007-03-17", 15.30, "SEC PR 2007-24", False),
    ("2008-01-25", 11.00, "SEC PR 2007-270", False),
    ("2008-04-01", 5.60, "SEC PR 2008-25 (mid-year adjustment)", False),
    ("2009-04-10", 25.70, "SEC PR 2009-56", False),
    ("2010-01-15", 12.70, "SEC PR 2009-270", False),
    ("2010-04-01", 16.90, "SEC PR 2010-29 (mid-year adjustment)", False),
    ("2011-01-21", 19.20, "SEC PR 2010-255", False),
    ("2012-02-21", 18.00, "SEC PR 2012-15", False),
    ("2012-04-01", 22.40, "SEC PR 2012-35 (mid-year adjustment)", False),
    ("2013-05-25", 17.40, "SEC PR 2013-74", False),
    ("2014-03-18", 22.10, "SEC PR 2014-30", False),
    ("2015-02-14", 18.40, "SEC PR 2015-8", False),
    ("2016-02-16", 21.80, "SEC PR 2016-2", False),
    ("2017-07-04", 23.10, "SEC PR 2017-111", False),
    ("2018-05-22", 13.00, "SEC PR 2018-67", False),
    ("2019-04-16", 20.70, "SEC PR 2019-30", False),
    ("2020-02-18", 22.10, "SEC PR 2020-7", False),
    ("2021-02-25", 5.10, "SEC PR 2021-8", False),
    ("2022-05-14", 22.90, "SEC PR 2022-60", False),
    ("2023-02-27", 8.00, "SEC PR 2023-15", False),
    ("2024-05-22", 27.80, "SEC Fee Rate Advisory #2 FY2024", False),
    ("2025-05-14", 0.00, "SEC Section 31 Advisory FY2025", False),
    ("2026-04-04", 20.60, "SEC Section 31 Advisory FY2026", False),
]
SEC31 = RateTable("SEC Section 31", [
    RateRow(_d(d), {"per_million": v}, src, ap) for d, v, src, ap in _S31
])


# --------------------------------------------------------------------------
# FINRA Trading Activity Fee (sales only)
# --------------------------------------------------------------------------
_TAF = [
    # date, per_share, max_per_trade, per_option_contract, source, approx
    ("1990-01-01", 0.0, 0.0, 0.0, "no TAF before NASD Schedule A TAF (2002/2003)", True),
    ("2003-01-01", 0.0001, 10.00, 0.002, "SR-NASD-2002-99 (initial TAF $0.10/1000 sh, $10 cap); option rate approx", True),
    ("2003-05-30", 0.00005, 5.00, 0.002, "SR-NASD-2002-148 ($0.00005/sh, $5 cap)", True),
    ("2003-09-01", 0.0001, 10.00, 0.002, "SR-NASD-2003-93 (restored $0.0001/sh, $10 cap)", True),
    ("2004-11-01", 0.000075, 3.75, 0.002, "NASD NtM 04-84 ($0.075/1000 sh, $3.75 cap)", False),
    ("2008-01-01", 0.000075, 3.75, 0.002, "FINRA Sch.A s.1 rev 2008-01-01", False),
    ("2011-07-01", 0.000090, 4.50, 0.002, "FINRA Sch.A s.1 rev 2011-07-01 (SR-FINRA-2011-020)", False),
    ("2012-03-01", 0.000095, 4.75, 0.002, "FINRA Sch.A s.1 rev 2012-03-01 (SR-FINRA-2012-008)", False),
    ("2012-07-01", 0.000119, 5.95, 0.002, "FINRA Sch.A s.1 rev 2012-07-01 (SR-FINRA-2012-023)", False),
    ("2022-01-01", 0.000130, 6.49, 0.00218, "FINRA Sch.A s.1 rev 2022-01-01 (SR-FINRA-2020-032)", False),
    ("2023-01-01", 0.000145, 7.27, 0.00244, "FINRA Sch.A s.1 rev 2023-01-01 (SR-FINRA-2020-032)", False),
    ("2024-01-01", 0.000166, 8.30, 0.00279, "FINRA Sch.A s.1 rev 2024-01-01 (SR-FINRA-2020-032)", False),
    ("2026-01-01", 0.000195, 9.79, 0.00329, "FINRA Sch.A s.1 rev 2026-01-01", False),
    ("2026-10-01", 0.0, 0.0, 0.0, "FINRA Sch.A s.1 rev 2026-10-01 (TAF fee holiday)", False),
    ("2027-01-01", 0.000195, 9.79, 0.00329, "FINRA Sch.A s.1 rev 2027-01-01", False),
]
FINRA_TAF = RateTable("FINRA TAF", [
    RateRow(_d(d), {"per_share": ps, "max_per_trade": mx, "per_contract": pc}, src, ap)
    for d, ps, mx, pc, src, ap in _TAF
])


# --------------------------------------------------------------------------
# OCC clearing fee (per contract, customer), cleared trades
# --------------------------------------------------------------------------
_OCC = [
    ("1990-01-01", 0.05, "pre-2016 tiered schedule approximated at $0.05", True),
    ("2016-05-01", 0.05, "SR-OCC-2016 flat per-contract fee $0.05 (date approx)", True),
    ("2019-04-01", 0.055, "SR-OCC-2019: $0.05 -> $0.055", False),
    ("2020-09-01", 0.045, "SR-OCC-2020-009: $0.055 -> $0.045", False),
    ("2021-06-01", 0.020, "SR-OCC-2021: $0.045 -> $0.02", False),
    ("2025-01-01", 0.025, "SR-OCC-2024: $0.02 -> $0.025 single tier", False),
]
OCC_CLEARING = RateTable("OCC clearing", [
    RateRow(_d(d), {"per_contract": v}, src, ap) for d, v, src, ap in _OCC
])


# --------------------------------------------------------------------------
# Options Regulatory Fee (aggregate of member-exchange ORFs a retail broker
# passes through). ORF is assessed by each options exchange on customer
# contracts cleared by its members regardless of execution venue, so a broker
# that is a member of many exchanges passes through the SUM. The per-exchange
# rates vary month to month (e.g. Phlx $0.0034 -> $0.0024 on 2025-08-01,
# NOM $0.0014 -> $0.0005 -> $0.00005 in 2025, Cboe BZX/EDGX $0.0001-0.0002).
# This table is an order-of-magnitude APPROXIMATION and is flagged as such.
# --------------------------------------------------------------------------
_ORF = [
    ("1990-01-01", 0.0, "no ORF before 2008 (approx)", True),
    ("2008-12-01", 0.0045, "Cboe ORF introduced (approx)", True),
    ("2012-01-01", 0.010, "multiple exchanges adopt ORF (aggregate approx)", True),
    ("2016-01-01", 0.020, "aggregate of member-exchange ORFs (approx)", True),
    ("2025-08-01", 0.010, "2025 ORF reductions (Phlx, NOM, MRX filings) (aggregate approx)", True),
]
ORF_AGGREGATE = RateTable("ORF aggregate", [
    RateRow(_d(d), {"per_contract": v}, src, ap) for d, v, src, ap in _ORF
])


# --------------------------------------------------------------------------
# Cboe proprietary index options (customer capacity 'C'), per contract.
# Base fee by premium + execution surcharge (SPX: $0.21 not incl. SPXW;
# SPXW electronic $0.14). Source: Cboe Fees Schedule 2026-10-01.
# Earlier history not reconstructed: the 2026 schedule is applied to all
# dates and flagged approximate before 2026-10-01.
# --------------------------------------------------------------------------
_CBOE_SPX = [
    ("1990-01-01", 0.36, 0.45, 0.21, 0.14, "Cboe Fees Schedule 2026-10-01 applied retroactively (approx)", True),
    ("2026-10-01", 0.36, 0.45, 0.21, 0.14, "Cboe Fees Schedule 2026-10-01", False),
]
CBOE_SPX = RateTable("Cboe SPX/SPXW customer", [
    RateRow(_d(d), {"lt1": a, "ge1": b, "spx_surcharge": s1, "spxw_surcharge": s2}, src, ap)
    for d, a, b, s1, s2, src, ap in _CBOE_SPX
])


# --------------------------------------------------------------------------
# Equity exchange access fees (taker) per share, Reg NMS Rule 610 cap.
# 2007-2025: $0.0030 cap (most venues charge ~the cap to remove).
# Rule 610 amendments (2024): $0.001 cap for NMS stocks priced >= $1 with
# compliance on 2025-11-03 (approx date).
# --------------------------------------------------------------------------
_ACCESS = [
    ("1990-01-01", 0.0030, "Reg NMS Rule 610 access fee cap $0.003 (applied pre-2007 as approx)", True),
    ("2007-02-05", 0.0030, "Reg NMS Rule 610 access fee cap", False),
    ("2025-11-03", 0.0010, "Rule 610 amendment access fee cap $0.001 (compliance date approx)", True),
]
EQUITY_TAKER_FEE = RateTable("Equity taker access fee", [
    RateRow(_d(d), {"per_share": v}, src, ap) for d, v, src, ap in _ACCESS
])


# --------------------------------------------------------------------------
# FINRA CAT fees passed through by brokers (approx start 2024-11-01)
# --------------------------------------------------------------------------
_CAT = [
    ("1990-01-01", 0.0, 0.0, "no CAT pass-through", True),
    ("2024-11-01", 0.000035, 0.0035, "CAT fee pass-through (approx)", True),
    ("2026-01-01", 0.000003, 0.0003, "IBKR schedule 2026 (FINRA CAT fee)", False),
]
CAT_FEE = RateTable("CAT fee", [
    RateRow(_d(d), {"per_share": a, "per_contract": b}, src, ap) for d, a, b, src, ap in _CAT
])


# Retail zero-commission era start (Schwab cut to $0 on 2019-10-07; TD, E*Trade
# followed within days; Robinhood $0 since 2013 but chosen industry date).
ZERO_COMMISSION_ERA = pd.Timestamp("2019-10-07")

# Reg SHO short-sale price tests
UPTICK_RULE_END = pd.Timestamp("2007-07-06")       # Rule 10a-1 eliminated (SEC Rel. 34-55970)
RULE_201_START = pd.Timestamp("2011-02-28")        # Rule 201 compliance date (extended from 2010-11-10)

# Settlement cycle (for documentation / cash-account checks)
SETTLEMENT_CYCLE = [
    (pd.Timestamp("1995-06-07"), 3),   # T+3
    (pd.Timestamp("2017-09-05"), 2),   # T+2
    (pd.Timestamp("2024-05-28"), 1),   # T+1
]


# --------------------------------------------------------------------------
# IBKR Pro commission history (opt-in: ``IBKRProTiered(historical=True)`` /
# fee schedule names ``IBKR_PRO_TIERED_HISTORICAL`` / ``IBKR_PRO_FIXED_HISTORICAL``).
#
# Verified points (retrieved 2026-10-30, User-Agent atlasbt-research/0.1):
# * IBG annual report 2013 (interactivebrokers.com/download/2013_IBG_AR.pdf, "Rates were
#   obtained on February 3, 2014"): US stocks $0.0005-$0.0035/share, US options
#   $0.075-$0.70 per contract; "Our transparent cost-plus commission is subject to USD 0.35
#   minimum on equities per order for monthly volume exceeding 100,000,000 shares and USD 0.70
#   minimum per order for monthly volume up to 100,000,000 shares. USD 1.00 minimum on options."
# * IBG annual report 2014 (rates obtained 2015-02-02): same minimums; "USD 1.00 minimum on
#   stocks and options" (Fixed).
# * IBG annual report 2016 (rates obtained 2017-03-02): options "$0.00 base, $0.70 per contract".
# * interactivebrokers.com/en/pricing/commissions-stocks.php and commissions-options.php
#   (live, 2026-10-30): Tiered min USD 0.35, Fixed USD 0.005/sh min USD 1.00, max 1% of trade
#   value; options USD 0.65 (premium >= 0.10) / 0.50 / 0.25, min USD 1.00.
# The exact date of the $0.70 -> $0.65 option cut and the tiered-minimum cut is NOT verified
# (web.archive.org was unreachable from this environment). Both are placed on 2019-09-26, the
# IBKR Pro/Lite repricing (IBKR Lite introduced on that date per the 2019 IBG annual report),
# and flagged approx. Rows before 2014-02-03 apply the earliest verified schedule (approx).
# --------------------------------------------------------------------------
_IBKR_HIST = [
    # date, opt rate (prem>=0.10, <=10k contracts/mo), tiered min (<=100M sh/mo), tiered min (>100M),
    # fixed min, source, approx
    ("1990-01-01", 0.70, 0.70, 0.35, 1.00,
     "IBG AR 2013 (2014-02-03) schedule applied retroactively (approx)", True),
    ("2014-02-03", 0.70, 0.70, 0.35, 1.00, "IBG AR 2013 (rates obtained 2014-02-03)", False),
    ("2019-09-26", 0.65, 0.35, 0.35, 1.00,
     "IBKR Pro/Lite repricing (date of the $0.65 / $0.35 change approx; verified 2026-10-30)", True),
]
IBKR_PRO_HISTORY = RateTable("IBKR Pro commission history", [
    RateRow(_d(d), {"opt_top": o, "tiered_min": m1, "tiered_min_high_volume": m2, "fixed_min": fm}, src, ap)
    for d, o, m1, m2, fm, src, ap in _IBKR_HIST
])


# --------------------------------------------------------------------------
# IBKR Pro USD interest schedule (opt-in: ``FinancingConfig(schedule="ibkr_pro")``).
# Benchmark (BM) = Fed Funds Effective (FRED DFF, ``portal.risk_free_rate(as_of, "ON")``):
# IBKR "US Dollar Fed Funds Effective" (interestPaid examples, interactivebrokers.com
# index.php?f=5424); before 2007-09 IBKR used USD LIBOR (not in the cache: DFF is used and
# the run is flagged).
#
# Margin loans (blended over tiers, BM + spread):
# * 2013-02-01 .. 2020-02-10 verified from the "US Margin Loan Rates" tables in the IBG annual
#   reports 2012/2013/2016/2019 (blended APR at $25K/$200-300K/$1.5M/$3.5M minus DFF on the
#   quoted date reproduces BM+1.5% <= 100K, +1.0% to 1M, +0.5% to 3M, +0.25% above to 0.02%).
# * 2021-01-04 (IBG AR 2020) and 2026-10-30 (live margin-rates page): BM+1.5% <= 100K, +1% to
#   1M, +0.75% to 50M, +0.5% above. The change date between 2020-02-10 and 2021-01-04 is not
#   verified; it is placed on 2020-02-11 (the earliest possible date: the newer tiers cost
#   more on large loans, i.e. conservative) and flagged approx.
# Credit on cash (IBKR Pro): 0% on the first USD 10,000, BM - 0.5% above, floored at 0
# (Attachment C "Brokerage Websites July 2017": USD 0-10,000 0%, 10,000.01+ BM-0.5% at
# BM 1.16% -> 0.66%; IBG AR 2017: 0.92% at BM 1.42% on 2018-02-05; live page 2026-10-30).
# Eligibility: July 2017 "only accounts with NAV exceeding USD 100,000 will be eligible";
# from the 2019 annual report (rates as of 2020-02-10) accounts below USD 100,000 are paid
# pro rata NAV/100,000. The switch is placed on 2020-02-10 (first verified date; the earlier
# all-or-nothing rule pays less, i.e. conservative before it).
# Short-sale proceeds (collateral) credit: USD 0-100K 0%, 100K-1M BM-1.25%, 1M-3M BM-0.5%,
# 3M+ BM-0.25% (identical in July 2017 and on the live short-sale-cost page 2026-10-30),
# same NAV eligibility; the borrow fee is charged separately (portal.borrow).
# Rows before the first verified date apply the earliest verified schedule (approx).
# --------------------------------------------------------------------------
FFE_BENCHMARK_START = pd.Timestamp("2007-09-01")    # IBKR USD benchmark LIBOR -> Fed Funds Effective (approx)
_INF = float("inf")
_IBKR_INT = [
    # date, debit tiers [(upper, spread)], credit threshold, credit spread, short tiers, nav proration, src, approx
    ("1990-01-01", ((1e5, 0.015), (1e6, 0.010), (3e6, 0.005), (_INF, 0.0025)), 10_000.0, 0.005,
     ((1e5, None), (1e6, 0.0125), (3e6, 0.005), (_INF, 0.0025)), False,
     "IBG AR 2012 (2013-02-01) / July 2017 schedule applied retroactively (approx)", True),
    ("2013-02-01", ((1e5, 0.015), (1e6, 0.010), (3e6, 0.005), (_INF, 0.0025)), 10_000.0, 0.005,
     ((1e5, None), (1e6, 0.0125), (3e6, 0.005), (_INF, 0.0025)), False,
     "IBG AR 2012/2013/2016/2019 margin tables; Attachment C July 2017 credit & short-proceeds tables", False),
    ("2020-02-10", ((1e5, 0.015), (1e6, 0.010), (3e6, 0.005), (_INF, 0.0025)), 10_000.0, 0.005,
     ((1e5, None), (1e6, 0.0125), (3e6, 0.005), (_INF, 0.0025)), True,
     "IBG AR 2019 (rates as of 2020-02-10): NAV < USD 100K paid pro rata", False),
    ("2020-02-11", ((1e5, 0.015), (1e6, 0.010), (5e7, 0.0075), (_INF, 0.005)), 10_000.0, 0.005,
     ((1e5, None), (1e6, 0.0125), (3e6, 0.005), (_INF, 0.0025)), True,
     "IBG AR 2020 (2021-01-04) / live margin-rates page 2026-10-30 (change date approx)", True),
]
IBKR_PRO_INTEREST = RateTable("IBKR Pro USD interest", [
    RateRow(_d(d), {"debit": deb, "credit_threshold": th, "credit_spread": cs, "short_proceeds": sp,
                    "nav_proration": pr}, src, ap)
    for d, deb, th, cs, sp, pr, src, ap in _IBKR_INT
])
IBKR_NAV_ELIGIBILITY = 100_000.0


def tiered_blend(balance: float, tiers, bm: float, sign: int = +1) -> float:
    """Annual interest AMOUNT on ``balance`` (>= 0) over IBKR-style tiers.

    ``tiers`` = ((upper_bound, spread), ...); a tier spread of None pays 0. ``sign=+1``:
    rate = BM + spread (debit); ``sign=-1``: rate = max(BM - spread, 0) (credit)."""
    total, lo = 0.0, 0.0
    for hi, spread in tiers:
        if balance <= lo:
            break
        amt = min(balance, hi) - lo
        if spread is not None:
            rate = bm + spread if sign > 0 else max(bm - spread, 0.0)
            total += amt * rate
        lo = hi
    return total


# --------------------------------------------------------------------------
# FINRA Rule 4210(f)(8)(B) pattern day trader: approved by SEC Rel. 34-44009 (2001-02-27),
# effective 2001-09-28 (NASD NtM 01-26 / NYSE IM 01-20). Minimum equity USD 25,000.
# --------------------------------------------------------------------------
PDT_EFFECTIVE = pd.Timestamp("2001-09-28")
PDT_MIN_EQUITY = 25_000.0
PDT_MAX_DAY_TRADES = 3          # a 4th day trade within 5 business days designates the account
PDT_WINDOW_SESSIONS = 5


def settlement_days(date) -> int:
    """Regular-way US equity settlement cycle in force on ``date`` (T+3 from 1995-06-07,
    T+2 from 2017-09-05, T+1 from 2024-05-28; T+5 before)."""
    d = _as_naive_date(date)
    n = 5
    for eff, k in SETTLEMENT_CYCLE:
        if d >= eff:
            n = k
    return n

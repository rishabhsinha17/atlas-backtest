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

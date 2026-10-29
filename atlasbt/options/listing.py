"""Historically faithful listed-expiry calendar, strike grids and tick sizes
for SYNTHETIC chains (fidelity T3_MODELED).  Every date below is cited in
``LISTING_FACTS`` (source + confidence) and in docs/OPTIONS.md.

Expiry convention (core.types.Instrument): ``expiry`` = LAST TRADING DAY.
  * Equity/ETF monthlies: 3rd Friday; if that Friday is an exchange holiday the
    last trading day is the preceding business day (e.g. Good Friday).
    (Pre-2015 the OCC expiration date was the Saturday after; last trading day
    was still the Friday, which is what we store.)
  * SPX AM-settled monthlies: settlement = SOQ on the 3rd Friday (or prior
    business day if holiday); last trading day = the session before it
    (Thursday).  ``settlement_date`` column holds the SOQ session.
  * SPXW (PM-settled): last trading day = settlement day.
  * Weekly holiday shifts: Monday expiries move to the next business day;
    Tue/Wed/Thu/Fri expiries move to the preceding business day; duplicates
    collapse onto the existing series.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from atlasbt.core import clock
from atlasbt.core.types import Fidelity

D = pd.Timestamp

# ---------------------------------------------------------------------------
# Facts (first EXPIRY date of each expiry type; series available from listing)
# ---------------------------------------------------------------------------
LISTING_FACTS = {
    # SPX
    "SPX_AM_MONTHLY": dict(first=D("1992-06-19"), src="SPX standard monthlies AM-settled (SOQ) since the 1992 change; "
                           "Cboe SPX specs", confidence="high"),
    "SPXW_FRI_AM": dict(first=D("2005-11-04"), last=D("2010-12-03"),
                        src="Cboe: first Weeklys (SPX, OEX) listed 2005-10-28 expiring 2005-11-04; SPX Weeklys were "
                            "AM-settled until the last AM weekly expired 2010-12-03 (Cboe circular Oct-2010)", confidence="high"),
    "SPXW_FRI_PM": dict(first=D("2010-12-10"), listed=D("2010-12-02"),
                        src="Cboe End-of-Week pilot (SEC approval 2010-09-14); PM-settled SPXW launched 2010-12-02",
                        confidence="high"),
    "SPXW_3RDFRI_PM": dict(first=D("2017-05-19"), listed=D("2017-05-01"),
                           src="SPXPM third-Friday PM series moved into SPXW class 2017-05-01 (SEC/FR 2017); earlier "
                               "SPXPM (C2, 2011-2017) not modeled", confidence="medium"),
    "SPXW_WED": dict(first=D("2016-03-02"), listed=D("2016-02-23"),
                     src="Cboe 2016-02-01 release; RG16-024 first expirations 2016-03-02/03-09", confidence="high"),
    "SPXW_MON": dict(first=D("2016-08-22"), listed=D("2016-08-15"),
                     src="Cboe 2016-07-11 release: listing 2016-08-15, first expirations 2016-08-22/08-29",
                     confidence="high"),
    "SPXW_TUE": dict(first=D("2022-04-26"), listed=D("2022-04-18"),
                     src="Cboe 2022-04-13 release: Tuesday expirations available from 2022-04-18; first expiry "
                         "inferred from the 1-week listing pattern of prior launches", confidence="medium"),
    "SPXW_THU": dict(first=D("2022-05-19"), listed=D("2022-05-11"),
                     src="Cboe 2022-04-13 release: Thursday expirations available from 2022-05-11; first expiry "
                         "inferred (1-week pattern)", confidence="medium"),
    # SPY / QQQ / IWM / DIA
    "ETF_FRI_WEEKLY": dict(first=D("2010-06-11"), listed=D("2010-06-04"),
                           src="Cboe: ETF Weeklys on SPY, QQQQ, DIA, IWM began 2010-06-04", confidence="high"),
    "SPY_WED": dict(first=D("2016-08-31"), listed=D("2016-08-24"),
                    src="SEC approval of BOX Wednesday SPY expirations 2016-08-24 (SR-BOX-2016-28); first expiry "
                        "ASSUMED one week later", confidence="low"),
    "SPY_MON": dict(first=D("2018-02-26"), listed=D("2018-02-16"),
                    src="Cboe SPY Monday Weeklys listing 2018-02-16 (SR-CBOE-2018-018)", confidence="medium"),
    "SPYQQQ_TUE": dict(first=D("2022-11-15"), listed=D("2022-11-14"),
                       src="MIAX alert 2022-11-14: Tuesday SPY/QQQ expirations Nov 15, 22, 29", confidence="high"),
    "SPYQQQ_THU": dict(first=D("2022-11-17"), listed=D("2022-11-16"),
                       src="MIAX alert 2022-11-14: Thursday SPY/QQQ listings for Nov 17, Dec 1", confidence="high"),
    "QQQ_MONWED": dict(first=D("2021-01-04"), src="UNVERIFIED start: 2021 Cboe filing (34-93255) describes QQQ "
                       "Mon/Wed expirations as existing; earliest date unknown -> conservative 2021", confidence="low"),
    "IWM_MONWED": dict(first=D("2021-10-04"), listed=D("2021-09-28"),
                       src="SEC Rel. 34-93157 (2021-09-28) approving IWM Mon/Wed expirations; first expiry assumed "
                           "next week", confidence="low"),
    "LEAPS": dict(first=D("1990-10-01"), src="Cboe LEAPS introduced 1990", confidence="high"),
    # XSP (Mini-SPX): single PM-settled root (integrator rule); weeklies modeled only once Cboe
    # stated XSP had expirations every weekday (2022-09-19 release); earlier XSP weeklies existed
    # but their start dates are unverified -> NOT modeled (conservative).
    "XSP_TUE": dict(first=D("2022-10-11"), listed=D("2022-10-03"),
                    src="Cboe 2022-09-19 release: XSP Tuesday expirations from 2022-10-03 (first expiry +1wk inferred)",
                    confidence="medium"),
    "XSP_THU": dict(first=D("2022-10-20"), listed=D("2022-10-12"),
                    src="Cboe 2022-09-19 release: XSP Thursday expirations from 2022-10-12", confidence="medium"),
    "XSP_MWF": dict(first=D("2022-10-12"), src="UNVERIFIED start: existed before; modeled only from the date XSP "
                    "dailies were complete (conservative)", confidence="low"),
    # RUT: AM monthlies (root RUT) + PM weeklies (root RUTW)
    "RUTW_MONWED": dict(first=D("2021-03-08"), listed=D("2021-03-01"),
                        src="Cboe release 'Cboe Exchanges to List Monday and Wednesday Expiring Weekly Options on RUTW' "
                            "(2021-03-01)", confidence="medium"),
    "RUTW_FRI": dict(first=D("2021-03-12"), listed=D("2021-03-01"),
                     src="UNVERIFIED start: Friday RUTW existed before 2021; modeled from 2021-03 (conservative)",
                     confidence="low"),
    "RUTW_TUE": dict(first=D("2024-01-16"), listed=D("2024-01-08"),
                     src="Cboe 2023-12-21 release: RUT daily expiries (Tue/Thu) from 2024-01-08", confidence="medium"),
    "RUTW_THU": dict(first=D("2024-01-18"), listed=D("2024-01-08"),
                     src="Cboe 2023-12-21 release", confidence="medium"),
    "RUTW_3RDFRI": dict(first=D("2024-11-15"), listed=D("2024-10-01"),
                        src="Cboe notice: standard third-Friday PM RUTW listed as of October 2024", confidence="low"),
    # NDX: AM monthlies only; PM weeklies (NDXP) launch date not verified -> not modeled.
    "NDX_AM_MONTHLY": dict(first=D("1994-01-01"), src="NDX index options AM-settled monthlies (long-standing)",
                           confidence="medium"),
    "RUT_AM_MONTHLY": dict(first=D("1994-01-01"), src="RUT index options AM-settled monthlies (long-standing)",
                           confidence="medium"),
}

# Index products with a listing model.  Any other index -> empty frame, Fidelity.NA.
INDEX_LISTING = {
    "XSP": {"monthly": ("XSP", "CASH_PM", None), "weekly_root": "XSP",
            "weekly": {0: "XSP_MWF", 1: "XSP_TUE", 2: "XSP_MWF", 3: "XSP_THU", 4: "XSP_MWF"}, "third_fri_pm": "XSP_MWF"},
    "NDX": {"monthly": ("NDX", "CASH_AM", "NDX_AM_MONTHLY"), "weekly_root": "NDXP", "weekly": {}, "third_fri_pm": None},
    "RUT": {"monthly": ("RUT", "CASH_AM", "RUT_AM_MONTHLY"), "weekly_root": "RUTW",
            "weekly": {0: "RUTW_MONWED", 1: "RUTW_TUE", 2: "RUTW_MONWED", 3: "RUTW_THU", 4: "RUTW_FRI"},
            "third_fri_pm": "RUTW_3RDFRI"},
}
KNOWN_INDEXES = {"SPX", "SPXW", "XSP", "NDX", "RUT", "VIX", "DJX", "OEX", "XEO", "MRUT", "NANOS", "NDXP", "RUTW"}

INDEX_ROOTS = {"SPX", "SPXW", "XSP", "NDX", "RUT"}
WEEKLY_ETFS = {"SPY", "QQQ", "IWM", "DIA"}


@dataclass(frozen=True)
class ProductSpec:
    underlying: str
    is_index: bool
    exercise: str           # 'AMERICAN' | 'EUROPEAN'
    multiplier: float = 100.0
    broad_based: bool = False


def product_spec(underlying: str) -> ProductSpec:
    u = underlying.upper().lstrip("^$")
    if u in ("SPX", "SPXW"):
        return ProductSpec("SPX", True, "EUROPEAN", 100.0, True)
    if u in ("NDXP",):
        return ProductSpec("NDX", True, "EUROPEAN", 100.0, True)
    if u in ("RUTW",):
        return ProductSpec("RUT", True, "EUROPEAN", 100.0, True)
    if u in KNOWN_INDEXES:
        return ProductSpec(u, True, "EUROPEAN", 100.0, True)
    broad = u in {"SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "VTI", "VOO", "IVV", "MDY"}
    return ProductSpec(u, False, "AMERICAN", 100.0, broad)


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------
def _cal():
    return clock.nyse()


def _is_session(d) -> bool:
    return _cal().is_session(d)


def _prev_business(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize()
    while not _is_session(d):
        d = d - pd.Timedelta(days=1)
    return d


def _next_business(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize()
    while not _is_session(d):
        d = d + pd.Timedelta(days=1)
    return d


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    off = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=off + 14)


def monthly_expiration_day(year: int, month: int) -> pd.Timestamp:
    """Equity monthly last trading day (3rd Friday, holiday -> prior session)."""
    return _prev_business(third_friday(year, month))


def _shift_weekly(d: pd.Timestamp) -> pd.Timestamp:
    if _is_session(d):
        return d
    return _next_business(d) if d.weekday() == 0 else _prev_business(d)


@dataclass(frozen=True)
class ExpiryRow:
    expiry: pd.Timestamp            # last trading day
    settlement_date: pd.Timestamp   # session whose open (AM) / close (PM/physical) settles
    kind: str                       # MONTHLY | WEEKLY | LEAPS | QUARTERLY
    root: str
    settlement: str                 # PHYSICAL | CASH_AM | CASH_PM
    exercise: str


def _months_ahead(as_of: pd.Timestamp, n: int):
    y, m = as_of.year, as_of.month
    for i in range(n):
        yy, mm = y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1
        yield yy, mm


def listed_expiries(underlying: str, as_of, horizon_days: int = 800, include_weeklies: bool | None = None,
                    weekly_horizon_days: int | None = None) -> pd.DataFrame:
    """Expiries listed (tradable) on session ``as_of`` for ``underlying``.

    Columns: expiry, settlement_date, kind, root, settlement, exercise.
    ``include_weeklies`` defaults True for SPX and SPY/QQQ/IWM/DIA, False for
    single names (weekly availability per stock is not modeled).
    """
    as_of = pd.Timestamp(as_of).normalize()
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)
    u = underlying.upper().lstrip("^$")
    spec = product_spec(u)
    u = spec.underlying if spec.is_index else u
    end = as_of + pd.Timedelta(days=horizon_days)
    rows: list[ExpiryRow] = []
    cols = ["expiry", "settlement_date", "kind", "root", "settlement", "exercise"]
    is_spx = u == "SPX"
    cfg = INDEX_LISTING.get(u)
    if spec.is_index and not is_spx and cfg is None:
        out = pd.DataFrame(columns=cols)
        out.attrs.update(fidelity=Fidelity.NA, reason=f"no listing model for index {u}")
        return out
    if include_weeklies is None:
        include_weeklies = (spec.is_index or u in WEEKLY_ETFS)
    # ---------------- monthlies / quarterlies / LEAPS
    months = set()
    for yy, mm in _months_ahead(as_of, 4):          # 4 nearest serial months
        months.add((yy, mm))
    for yy, mm in _months_ahead(as_of, 13):         # quarterly cycle out to ~1y
        if mm in (3, 6, 9, 12):
            months.add((yy, mm))
    if spec.is_index:                                # SPX: 12 serial months, quarterlies to 3y, Dec to 5y
        for yy, mm in _months_ahead(as_of, 12):
            months.add((yy, mm))
        for yy, mm in _months_ahead(as_of, 37):
            if mm in (3, 6, 9, 12):
                months.add((yy, mm))
        for yy, mm in _months_ahead(as_of, 61):
            if mm == 12:
                months.add((yy, mm))
    else:                                            # LEAPS: January, next ~2.5 years
        for yy, mm in _months_ahead(as_of, 32):
            if mm == 1 and as_of >= LISTING_FACTS["LEAPS"]["first"]:
                months.add((yy, mm))
    for yy, mm in sorted(months):
        if cfg is not None:
            root_m, st_m, fact = cfg["monthly"]
            settle = _prev_business(third_friday(yy, mm))
            if fact is not None and as_of < LISTING_FACTS[fact]["first"]:
                continue
            if st_m == "CASH_AM":
                rows.append(ExpiryRow(_cal().prev_session(settle), settle, "MONTHLY", root_m, "CASH_AM", "EUROPEAN"))
            else:
                rows.append(ExpiryRow(settle, settle, "MONTHLY", root_m, st_m, "EUROPEAN"))
            tf = cfg.get("third_fri_pm")
            if tf and st_m == "CASH_AM" and settle >= LISTING_FACTS[tf]["first"] \
                    and as_of >= LISTING_FACTS[tf].get("listed", LISTING_FACTS[tf]["first"]):
                rows.append(ExpiryRow(settle, settle, "MONTHLY", cfg["weekly_root"], "CASH_PM", "EUROPEAN"))
        elif is_spx:
            settle = _prev_business(third_friday(yy, mm))
            if as_of >= LISTING_FACTS["SPX_AM_MONTHLY"]["first"]:
                ltd = _cal().prev_session(settle)
                rows.append(ExpiryRow(ltd, settle, "MONTHLY", "SPX", "CASH_AM", "EUROPEAN"))
            else:
                rows.append(ExpiryRow(settle, settle, "MONTHLY", "SPX", "CASH_PM", "EUROPEAN"))
            if as_of >= LISTING_FACTS["SPXW_3RDFRI_PM"]["listed"]:
                rows.append(ExpiryRow(settle, settle, "MONTHLY", "SPXW", "CASH_PM", "EUROPEAN"))
        else:
            e = monthly_expiration_day(yy, mm)
            kind = "LEAPS" if (e - as_of).days > 270 and mm == 1 else ("QUARTERLY" if mm in (3, 6, 9, 12) else "MONTHLY")
            rows.append(ExpiryRow(e, e, kind, u, "PHYSICAL", "AMERICAN"))
    # ---------------- weeklies
    if include_weeklies:
        wh = weekly_horizon_days
        if wh is None:
            wh = 35 if as_of < D("2016-01-01") else 50
        wend = as_of + pd.Timedelta(days=wh)
        days = pd.date_range(as_of, wend, freq="D")
        for d in days:
            wd = d.weekday()
            if wd > 4:
                continue
            types = []
            if cfg is not None:
                key = cfg["weekly"].get(wd)
                if key:
                    f = LISTING_FACTS[key]
                    is3 = wd == 4 and d == third_friday(d.year, d.month)
                    if d >= f["first"] and as_of >= f.get("listed", f["first"]) and not is3:
                        types.append("IDX_PM")
            elif is_spx:
                if wd == 4:
                    if LISTING_FACTS["SPXW_FRI_AM"]["first"] <= d <= LISTING_FACTS["SPXW_FRI_AM"]["last"]:
                        types.append("FRI_AM")
                    elif d >= LISTING_FACTS["SPXW_FRI_PM"]["first"] and as_of >= LISTING_FACTS["SPXW_FRI_PM"]["listed"]:
                        is3 = d == third_friday(d.year, d.month)
                        # Weeklys were not listed in standard-expiration week; third-Friday PM joins SPXW 2017-05
                        if not is3 or as_of >= LISTING_FACTS["SPXW_3RDFRI_PM"]["listed"]:
                            types.append("PM")
                keys = {0: "SPXW_MON", 1: "SPXW_TUE", 2: "SPXW_WED", 3: "SPXW_THU"}
                if wd in keys:
                    f = LISTING_FACTS[keys[wd]]
                    if d >= f["first"] and as_of >= f.get("listed", f["first"]):
                        types.append("PM")
            else:
                if wd == 4 and u in WEEKLY_ETFS and d >= LISTING_FACTS["ETF_FRI_WEEKLY"]["first"] \
                        and as_of >= LISTING_FACTS["ETF_FRI_WEEKLY"]["listed"]:
                    types.append("PHYS")
                if u not in WEEKLY_ETFS and wd == 4 and d >= LISTING_FACTS["ETF_FRI_WEEKLY"]["first"]:
                    types.append("PHYS")   # single names: only if caller asked for weeklies
                key = None
                if u == "SPY":
                    key = {0: "SPY_MON", 1: "SPYQQQ_TUE", 2: "SPY_WED", 3: "SPYQQQ_THU"}.get(wd)
                elif u == "QQQ":
                    key = {0: "QQQ_MONWED", 1: "SPYQQQ_TUE", 2: "QQQ_MONWED", 3: "SPYQQQ_THU"}.get(wd)
                elif u == "IWM":
                    key = {0: "IWM_MONWED", 2: "IWM_MONWED"}.get(wd)
                if key:
                    f = LISTING_FACTS[key]
                    if d >= f["first"] and as_of >= f.get("listed", f["first"]):
                        types.append("PHYS")
            for t in types:
                e = _shift_weekly(d)
                if e < as_of:
                    continue
                if t == "FRI_AM":
                    ltd = _cal().prev_session(e)
                    if ltd >= as_of:
                        rows.append(ExpiryRow(ltd, e, "WEEKLY", "SPX", "CASH_AM", "EUROPEAN"))
                elif t == "PM":
                    rows.append(ExpiryRow(e, e, "WEEKLY", "SPXW", "CASH_PM", "EUROPEAN"))
                elif t == "IDX_PM":
                    rows.append(ExpiryRow(e, e, "WEEKLY", cfg["weekly_root"], "CASH_PM", "EUROPEAN"))
                else:
                    rows.append(ExpiryRow(e, e, "WEEKLY", u, "PHYSICAL", "AMERICAN"))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[(df["expiry"] >= as_of) & (df["expiry"] <= end)]
    # a weekly that coincides with a monthly of the same root+settlement collapses onto the monthly
    df["_rank"] = df["kind"].map({"MONTHLY": 0, "QUARTERLY": 0, "LEAPS": 0, "WEEKLY": 1})
    df = df.sort_values(["expiry", "root", "settlement", "_rank"]).drop_duplicates(["expiry", "root", "settlement"])
    return df.drop(columns="_rank").sort_values(["expiry", "root"]).reset_index(drop=True)


def has_0dte(underlying: str, session) -> bool:
    """True if a (PM-settled for SPX) series expires on ``session`` — e.g. SPXW
    on a Tuesday in 2023 -> True; Tuesday in 2015 -> False."""
    s = pd.Timestamp(session).normalize()
    if not _is_session(s):
        return False
    df = listed_expiries(underlying, s, horizon_days=1)
    df = df[df["expiry"] == s]
    if product_spec(underlying).is_index and len(df):
        df = df[df["settlement"] == "CASH_PM"]   # AM series stop trading the session before
    return len(df) > 0


# ---------------------------------------------------------------------------
# strikes and ticks
# ---------------------------------------------------------------------------
def strike_increment(underlying: str, strike: float, as_of, dte: int) -> float:
    """Approximate listed strike interval (documented simplification).

    SPX: 5 pts within the near-term band, 25 pts beyond / long-dated.
    SPY/QQQ/IWM/DIA: $1 (dte<=400), $5 beyond.  Single names (by era):
      pre-2012: 2.5 (<25), 5 (25-200), 10 (>200);
      2012+:    1 (<50), 2.5 (50-100), 5 (100-500), 10 (>500).
    """
    u = underlying.upper()
    as_of = pd.Timestamp(as_of)
    if u in ("SPX", "SPXW"):
        return 5.0 if dte <= 400 else 25.0
    if u in ("XSP",):
        return 1.0
    if u in ("NDX", "NDXP"):
        return 25.0 if dte <= 400 else 100.0
    if u in ("RUT", "RUTW"):
        return 5.0 if dte <= 400 else 25.0
    if u in WEEKLY_ETFS:
        return 1.0 if dte <= 400 else 5.0
    if as_of < D("2012-01-01"):
        return 2.5 if strike < 25 else (5.0 if strike < 200 else 10.0)
    return 1.0 if strike < 50 else (2.5 if strike < 100 else (5.0 if strike < 500 else 10.0))


def strike_grid(underlying: str, spot: float, as_of, dte: int, atm_vol: float = 0.25, n_sd: float = 4.0,
                min_pct: float = 0.15, max_pct: float = 0.80) -> np.ndarray:
    """Listed strikes around spot: span = clip(n_sd * vol * sqrt(T), min_pct, max_pct).
    Wider increments beyond the near band for SPX (25 pts beyond +-15%)."""
    T = max(dte, 1) / 365.0
    span = float(np.clip(n_sd * atm_vol * np.sqrt(T), min_pct, max_pct))
    lo, hi = spot * (1 - span), spot * (1 + span)
    u = underlying.upper()
    ks = []
    k = 0.0
    inc0 = strike_increment(u, spot, as_of, dte)
    k = np.floor(lo / inc0) * inc0
    while k <= hi + 1e-9:
        if k > 0:
            ks.append(round(k, 4))
        if u in ("SPX", "SPXW") and abs(k / spot - 1) > 0.15:
            inc = 25.0
        else:
            inc = strike_increment(u, max(k, 1e-9), as_of, dte)
        k = (np.floor(k / inc + 1e-9) + 1) * inc
    return np.unique(np.asarray(ks, float))


PENNY_ALL = {"SPY", "QQQ", "IWM"}


def tick_size(underlying: str, price, as_of) -> np.ndarray:
    """Minimum option price increment (historical approximation):
    pre-2007-01-26: $0.05 (<$3) / $0.10.  Penny Pilot from 2007-01-26:
    SPY/QQQ/IWM $0.01 at all prices; other classes $0.01 (<$3)/$0.05 (assumes
    the name is in the pilot — true for most liquid names after 2010).  SPX:
    $0.05 (<$3)/$0.10."""
    p = np.asarray(price, float)
    u = underlying.upper()
    as_of = pd.Timestamp(as_of)
    if u in ("SPX", "SPXW", "NDX", "NDXP", "RUT", "RUTW") or as_of < D("2007-01-26"):
        return np.where(p < 3.0, 0.05, 0.10)
    if u in PENNY_ALL:
        return np.full(p.shape, 0.01)
    return np.where(p < 3.0, 0.01, 0.05)

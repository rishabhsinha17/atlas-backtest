"""Point-in-time security identity and verified daily price series (red-team fixes #1-#4, #10).

Problem: Yahoo serves today's company under a ticker (reused tickers attach a DIFFERENT
company's history: BBT, IR, CPWR, EP, GENZ, ...), and DoltHub is not reliably as-traded either
(VIAC holds CBS back to 2011, CB mixes ACE and old Chubb, FI 2014-2021 is Frank's International).
So prices are only served when the security's identity is established.

Identity
  security_id = 'CIK##########' (SEC registrant). Evidence for "ticker T denoted CIK c on date d"
  comes from the ZipLime corpus of every 8-K item 1993-2026 (as-of ticker resolution; 4.4 M
  filings) plus its 8-K 2.02 announcements. Only 'as_of' evidence (ticker observed at filing,
  2006+) LABELS dates: a date is attributed to CIK c if bracketed by as-of evidence of c no more
  than 548 days apart, or within 120 days of the nearest as-of evidence. 'carried_back' /
  'carried_forward' evidence (a later/earlier ticker copied onto a filing) is used by the Portal
  only within 120 days and only if the S&P member's name matches the CIK's SEC names. Before a
  ticker's first as-of evidence, `pre_evidence_cik` proposes the earliest evidenced issuer; the
  Portal accepts it only if the member of that S&P membership interval confirms it (Wikipedia
  CIK of a current constituent, or the recorded member name) -> T2. Tickers without ZipLime
  evidence (share classes such as FOXA, BRK-B) use the SEC current ticker map, valid only from the
  first evidence date of that CIK. ETFs / indices bypass this layer.

Verified series for a CIK (build_series)
  Segments: Yahoo under every ticker of the CIK whose CURRENT owner is that CIK (Yahoo keeps the
  current entity's history under its current ticker); DoltHub as-traded rows of every ticker of
  the CIK on the dates that ticker is evidenced to denote the CIK; DoltHub rows of unknown identity
  only if they agree with an accepted segment (median |log ratio| <= 1.5 %, no day > 5 %, >= 20
  overlapping sessions). DoltHub runs failing the level-alternation collision test (spike-and-revert
  glitches > max(3, 1 %) of days) are rejected; isolated glitches are dropped (qa.bad_ticks).
  Splice validation: a source change with an unexplained close-to-close move > 20 % (or > 6 MADs)
  drops the lower-priority chunk; single-day fills must lie within 10 % of both neighbours.
  Actions follow the price segment (Yahoo actions of the Yahoo ticker used, DoltHub splits /
  dividends for DoltHub chunks); every split is validated against the raw price ratio around the
  ex-date (dropped if absent, inferred from an unexplained common-ratio jump in DoltHub chunks).
  FAIL CLOSED: no established identity -> no bars (DataUnavailableError), counted missing.
Everything rejected/dropped/inferred is logged in the series log (attrs['identity_log']).
"""
from __future__ import annotations

import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from . import config, frames, qa

log = logging.getLogger("atlasbt.data.identity")
ZL = config.DATA_RAW / "hf_ziplime_earnings_calendar" / "data"
BUILD_VERSION = "2026-10-30.6"     # bump when the series construction rules change
MAX_EVIDENCE_GAP_DAYS = 548
EDGE_DAYS = 120
COMMON_SPLITS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15,
                          1 / 20, 3 / 2, 2 / 3, 5 / 4, 4 / 5, 4 / 3, 3 / 4, 5 / 2, 2 / 5])
# ratios we are willing to INFER from an unexplained DoltHub jump (volume must scale too)
INFER_SPLITS = np.array([2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 1 / 20])
# lower = preferred on a date served by several segments; '?' = rows of unknown identity
SOURCE_RANK = {"yahoo": 0, "dolthub": 1, "alpaca_eod": 1.5, "tiingo": 1.8, "dolthub?": 2, "alpaca_eod?": 2.5,
               "tiingo?": 2.8}
AS_TRADED_SOURCES = ("dolthub", "alpaca_eod")


def norm(t) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


# ---------------------------------------------------------------------------- evidence
def build_evidence(store) -> pd.DataFrame:
    """(ticker, cik, date, conf) with conf 0 = as_of, 1 = carried; one row per ticker/cik/conf/month."""
    parts = []
    for p in sorted(glob.glob(str(ZL / "events" / "*.parquet"))):
        e = pd.read_parquet(p, columns=["cik", "ticker", "ticker_confidence", "filed_date"])
        parts.append(e.rename(columns={"filed_date": "date"}))
    for p in sorted(glob.glob(str(ZL / "announcements" / "*.parquet"))):
        e = pd.read_parquet(p, columns=["cik", "ticker", "ticker_confidence", "filed_date"])
        parts.append(e.rename(columns={"filed_date": "date"}))
    ev = pd.concat(parts, ignore_index=True).dropna(subset=["ticker", "cik"])
    ev["ticker"] = ev["ticker"].map(norm)
    ev["cik"] = ev["cik"].astype(int)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["conf"] = np.where(ev["ticker_confidence"] == "as_of", 0, 1).astype("int8")
    ev["m"] = ev["date"].dt.to_period("M")
    g = ev.groupby(["ticker", "cik", "conf", "m"], observed=True)["date"].agg(["min", "max"]).reset_index()
    a = g[["ticker", "cik", "conf", "min"]].rename(columns={"min": "date"})
    b = g[["ticker", "cik", "conf", "max"]].rename(columns={"max": "date"})
    out = pd.concat([a, b]).drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)
    store.write("reference", "identity", "evidence", out, key=["ticker", "cik", "conf", "date"], replace=True,
                extra={"origin": "ZipLime 8-K events + announcements (as-of ticker resolution)"})
    return out


class IdentityIndex:
    def __init__(self, store):
        self.store = store
        ev = store.read("reference", "identity", "evidence")
        if not len(ev):
            ev = build_evidence(store)
        tm = store.read("reference", "sec", "ticker_map")
        if len(tm) and len(ev):
            # the SEC current ticker map is itself as-of evidence on its snapshot date (only for
            # tickers that have 8-K evidence; share classes keep the first-evidence rule below)
            snap = pd.Timestamp(str(store.meta("reference", "sec", "ticker_map").get("fetched_at", ""))[:10] or
                                pd.Timestamp.now().normalize())
            known = set(ev["ticker"].unique())
            add = pd.DataFrame({"ticker": tm["ticker"].map(norm), "cik": tm["cik"].astype(int)})
            add = add[add["ticker"].isin(known)].drop_duplicates("ticker")
            add["conf"], add["date"] = np.int8(0), snap.normalize()
            ev = pd.concat([ev, add[ev.columns.intersection(add.columns)]], ignore_index=True)
        ev = ev.reset_index(drop=True).sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
        self._ev = ev
        t = ev["ticker"].to_numpy()
        u, first = np.unique(t, return_index=True)
        last = np.r_[first[1:], len(t)]
        self._tslice = {k: (a, b) for k, a, b in zip(u, first, last)}
        evc = ev.sort_values(["cik", "date"], kind="mergesort").reset_index(drop=True)
        self._evc = evc
        c = evc["cik"].to_numpy()
        uc, fc = np.unique(c, return_index=True)
        lc = np.r_[fc[1:], len(c)]
        self._cslice = {int(k): (a, b) for k, a, b in zip(uc, fc, lc)}
        self.first_seen = {int(k): evc["date"].iloc[a] for k, a in zip(uc, fc)}
        tm = store.read("reference", "sec", "ticker_map")
        self.sec_now = {norm(t_): int(c_) for t_, c_ in zip(tm["ticker"], tm["cik"])} if len(tm) else {}
        # NB: only the SEC *current* ticker map defines the current owner of a ticker (the
        # security_master also holds historical tickers resolved via ZipLime -> not "current")
        asof = self._ev[self._ev["conf"] == 0]
        self.first_asof = asof.groupby("cik")["date"].min().to_dict()

    @property
    def by_ticker(self):
        return _SliceView(self._ev, self._tslice)

    @property
    def by_cik(self):
        return _SliceView(self._evc, self._cslice)

    # -- lookups
    def ticker_cik_at(self, ticker: str, date, allow_carried: bool = True) -> Optional[int]:
        """CIK that `ticker` denoted on `date` (as-of evidence; see label_dates). If only carried
        evidence exists near the date it is used when allow_carried (the ticker may then be the
        company's later ticker copied back -- callers check names / prices)."""
        t = norm(ticker)
        d = pd.Timestamp(date).normalize()
        c = int(self.label_dates(t, pd.DatetimeIndex([d]))[0])
        if c:
            return c
        e = self.by_ticker.get(t)
        if e is not None and len(e) and allow_carried:
            dist = (e["date"] - d).abs()
            i = int(dist.values.argmin())
            if dist.iloc[i].days <= EDGE_DAYS:
                return int(e["cik"].iloc[i])
        return None

    def evidence_conf_at(self, ticker: str, date) -> Optional[int]:
        """0 if as-of evidence within the window supports ticker_cik_at, 1 if only carried."""
        e = self.by_ticker.get(norm(ticker))
        if e is None or not len(e):
            return None
        d = pd.Timestamp(date).normalize()
        ea = e[e["conf"] == 0]
        if len(ea) and (ea["date"] - d).abs().min().days <= MAX_EVIDENCE_GAP_DAYS:
            return 0
        return 1

    def label_dates(self, ticker: str, dates: pd.DatetimeIndex) -> np.ndarray:
        """Vectorized as-of identity of `ticker` on many dates (0 = unknown). Only 'as_of'
        evidence labels dates: carried_back / carried_forward tickers are a CIK's later/earlier
        ticker copied onto old/new filings and are NOT evidence of who traded under the ticker
        (e.g. Clear Channel 'CCU' carried forward to 2018 while CCU was Cervecerias Unidas)."""
        t = norm(ticker)
        d = pd.DatetimeIndex(dates).values
        out = np.zeros(len(d), dtype=np.int64)
        e = self.by_ticker.get(t)
        if e is None or not len(e):
            c = self.sec_now.get(t)
            first = self.first_asof.get(c) if c is not None else None
            if c is not None and first is not None:
                out[d >= first.to_datetime64()] = c
            elif c is None:
                sib = self.class_sibling(t)
                if sib is not None:
                    # share class without own evidence and no current SEC owner (VIAB 2006-2019 =
                    # Viacom class B, evidenced as 'VIA'; DISCK = Discovery class C, 'DISCA'):
                    # the class line denotes the sibling's issuer on the dates the sibling does
                    return self.label_dates(sib, dates)
            return out
        ec = e[e["conf"] == 0]
        if not len(ec):
            return out
        ed = ec["date"].values.astype("datetime64[D]")
        ck = ec["cik"].values
        d = d.astype("datetime64[D]")
        pos = np.searchsorted(ed, d, side="right")       # ed[pos-1] <= d < ed[pos]
        has_prev = pos > 0
        has_next = pos < len(ed)
        pi = np.clip(pos - 1, 0, len(ed) - 1)
        ni = np.clip(pos, 0, len(ed) - 1)
        dprev = np.where(has_prev, (d - ed[pi]).astype("timedelta64[D]"), np.timedelta64(10**6, "D"))
        dnext = np.where(has_next, (ed[ni] - d).astype("timedelta64[D]"), np.timedelta64(10**6, "D"))
        span = np.timedelta64(MAX_EVIDENCE_GAP_DAYS, "D")
        edge = np.timedelta64(EDGE_DAYS, "D")
        # bracketed by evidence of the same CIK no more than 548 days apart ...
        inside = has_prev & has_next & (ck[pi] == ck[ni]) & ((dprev + dnext) <= span)
        out = np.where(inside, ck[pi], out)
        # ... or within EDGE_DAYS of the nearest evidence (beyond a company's first/last filing the
        # ticker is not attributed: Smith International 'SII' stops 2010-08, Sprott reuses it)
        near_prev = has_prev & (dprev <= edge)
        near_next = has_next & (dnext <= edge)
        pick = np.where(near_prev & (~near_next | (dprev <= dnext)), ck[pi], np.where(near_next, ck[ni], 0))
        return np.where(out == 0, pick, out)

    def class_sibling(self, ticker: str) -> Optional[str]:
        """Evidenced sibling share class of a class ticker with NO evidence of its own and no current
        SEC owner: 'VIAB' -> 'VIA', 'DISCK' -> 'DISCA', 'CMCSK' -> 'CMCSA', 'LBTYK' -> 'LBTYA'.
        Only class letters B / C / K (A is the usual primary line, and other suffixes are not class
        markers: 'CCEP' is NOT 'CCE'); the sibling must have as-of evidence. Dates are attributed by the
        sibling's own date labels (VIA = Viacom 2006-2019, Via Renewables 2021-2024), so the class line
        is only ever attributed to the issuer the sibling denoted on that date. None otherwise."""
        t = norm(ticker)
        if len(t) < 4 or t[-1] not in "BCK" or "-" in t:
            return None
        if self.by_ticker.get(t) is not None or t in self.sec_now:
            return None
        for sib in (t[:-1] + "A", t[:-1]):
            if sib == t:
                continue
            e = self.by_ticker.get(sib)
            if e is None or not len(e):
                continue
            if len(e[e["conf"] == 0]):
                return sib
        return None

    def pre_evidence_cik(self, ticker: str, date, renamed_before=None) -> Optional[int]:
        """Candidate issuer of `ticker` on a date BEFORE its first as-of evidence (ZipLime's as-of
        ticker resolution starts in 2006): the CIK of the ticker's earliest as-of evidence, unless
        that CIK was evidenced under another ticker before (renamed INTO the ticker -> at `date` it
        probably traded under the other one). A candidate only: the Portal accepts it solely when
        the S&P 500 member's name for that membership interval (or Wikipedia's CIK for a current
        member) confirms it, and serves it as T2."""
        t = norm(ticker)
        e = self.by_ticker.get(t)
        if e is None or not len(e):
            return None
        ea = e[e["conf"] == 0]
        if not len(ea):
            return None
        first = ea["date"].iloc[0]
        d = pd.Timestamp(date).normalize()
        if d >= first:
            return None
        c = int(ea["cik"].iloc[0])
        ec = self.by_cik.get(c)
        if ec is not None and len(ec):
            before = ec[(ec["conf"] == 0) & (ec["date"] < first) & (ec["ticker"] != t)]
            # evidence under another ticker blocks, unless it all predates `renamed_before` (the
            # start of the S&P membership interval under this ticker: BLL -> BALL 2022-05)
            if renamed_before is not None:
                before = before[before["date"] > pd.Timestamp(renamed_before) + pd.Timedelta(days=30)]
            if len(before):
                return None
        return c

    def is_successor(self, pred: int, succ: int, ticker: str) -> bool:
        """Issuer succession on the SAME traded line (holding-company reorganization, merger into
        the listed entity, re-domiciliation: XOM 34088 -> 2115436, MRK 64978 -> 310158, ORCL
        777676 -> 1341439, DIS 1001039 -> 1744489, MDT 64670 -> 1613103): the ticker's as-of
        evidence passes from `pred` to `succ` with no third issuer afterwards and no gap > 548
        days. Callers must also check names and price continuity (build_series)."""
        if pred == succ:
            return True
        e = self.by_ticker.get(norm(ticker))
        if e is None or not len(e):
            return False
        ea = e[e["conf"] == 0]
        ep, es = ea[ea["cik"] == pred], ea[ea["cik"] == succ]
        if not len(ep):
            return False
        if not len(es):
            # very recent hand-over (new holdco not yet in the 8-K corpus, e.g. XOM 2025): the
            # successor must be the SEC current owner and the predecessor evidenced until recently
            last_any = ea["date"].max()
            return (self.sec_now.get(norm(ticker)) == succ and ep["date"].max() == last_any
                    and (pd.Timestamp.now() - last_any).days <= MAX_EVIDENCE_GAP_DAYS)
        last_p = ep["date"].max()
        after = ea[ea["date"] > last_p]
        if len(after) and set(after["cik"].astype(int)) != {int(succ)}:
            return False
        first_s = es["date"].min()
        if es["date"].max() < last_p:
            return False
        return (first_s - last_p).days <= MAX_EVIDENCE_GAP_DAYS

    def cik_now(self, ticker: str) -> Optional[int]:
        t = norm(ticker)
        if t in self.sec_now:
            return self.sec_now[t]
        e = self.by_ticker.get(t)
        if e is not None and len(e):
            last = e[e["conf"] == 0]
            if len(last) and (pd.Timestamp.now() - last["date"].iloc[-1]).days < 400:
                return int(last["cik"].iloc[-1])
        return None

    def zl_tickers_of(self, cik: int) -> list[str]:
        e = self.by_cik.get(cik)
        if e is None:
            return []
        e = e[e["conf"] == 0]
        return list(dict.fromkeys(e.sort_values("date")["ticker"]))

    def sec_tickers_of(self, cik: int) -> list[str]:
        return [t for t, c in self.sec_now.items() if c == cik]

    def lineage(self, cik: int, ticker: str) -> tuple[list[str], list[str]]:
        """(tickers verified by CIK evidence, extra tickers acceptable only by price agreement).
        A share-class ticker without ZipLime evidence (FOXA, BRK-B, GOOG...) is its own lineage:
        other classes of the same registrant are different securities."""
        t = norm(ticker)
        zl = self.zl_tickers_of(cik)
        if t in zl or not zl:
            main = list(dict.fromkeys(zl + [t]))
        else:
            main = [t]
        extra = [x for x in self.sec_tickers_of(cik) if x not in main]
        return main, extra


class _SliceView:
    def __init__(self, df, slices):
        self.df, self.slices = df, slices

    def get(self, key, default=None):
        sl = self.slices.get(key)
        if sl is None:
            return default
        return self.df.iloc[sl[0]:sl[1]]


# ---------------------------------------------------------------------------- series builder
def _agree(a: pd.Series, b: pd.Series, min_overlap: int = 20) -> Optional[bool]:
    """Price agreement over >= min_overlap common sessions: median |log ratio| <= 1.5 % and the
    99th percentile <= 5 % (isolated vendor glitches tolerated). None if not enough overlap."""
    common = a.index.intersection(b.index)
    if len(common) < min_overlap:
        return None
    r = np.abs(np.log(a.loc[common].to_numpy() / b.loc[common].to_numpy()))
    r = r[np.isfinite(r)]
    if len(r) < min_overlap:
        return None
    return bool(np.median(r) <= 0.015 and np.quantile(r, 0.99) <= 0.05)


_MKT: dict = {}


def _market(portal) -> Optional[pd.Series]:
    if "spy" not in _MKT:
        b = portal._read("bars_1d", "yahoo", "SPY")
        _MKT["spy"] = pd.Series(b["close"].to_numpy(), index=b.index) if len(b) else None
    return _MKT["spy"]


def _glitches(close: pd.Series, portal=None) -> pd.Series:
    mkt = _market(portal) if portal is not None else None
    if mkt is not None:
        mkt = mkt.reindex(close.index)
    return qa.bad_ticks(close, k=6.0, min_move=0.08, market=mkt)


def _dolt_actions(portal, ticker: str) -> pd.DataFrame:
    a = portal._dolthub_actions(ticker)
    return a if a is not None else pd.DataFrame()


def _off_tape(portal, ticker: str, dates: pd.DatetimeIndex, min_run: int = 5) -> np.ndarray:
    """True for dates inside Alpaca's fetched range for `ticker` on which the SIP tape has no trade
    under that symbol, in runs of >= `min_run` consecutive sessions (isolated gaps are ignored).
    All False when the ticker was not fetched from Alpaca (no tape evidence)."""
    out = np.zeros(len(dates), dtype=bool)
    if not len(dates) or not portal.store.exists("bars_1d", "alpaca_eod", ticker):
        return out
    meta = portal.store.meta("bars_1d", "alpaca_eod", ticker) or {}
    a = portal._read("bars_1d", "alpaca_eod", ticker)
    if not len(a):
        return out
    have = set(frames.to_naive_date(a.index))
    lo = pd.Timestamp("2016-01-04")
    hi = pd.Timestamp(str(meta.get("fetched_at", ""))[:10] or frames.to_naive_date(a.index)[-1]) - pd.Timedelta(days=1)
    sch = frames.schedule().index
    sch = sch[(sch >= lo) & (sch <= hi)]
    miss = ~sch.isin(list(have))
    run = np.zeros(len(sch), dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        if i1 - i0 >= min_run:
            run[i0:i1] = True
    gone = set(sch[run])
    return np.array([d in gone for d in pd.DatetimeIndex(dates)], dtype=bool)


def _close_series(b: pd.DataFrame) -> pd.Series:
    return pd.Series(b["close"].to_numpy(), index=frames.to_naive_date(b.index))


def _junction_ok(run: pd.DataFrame, follower: Optional[pd.DataFrame], max_gap: int = 10, tol: float = 0.10) -> bool:
    """Renamed-history continuity: `run` (identity-unknown rows of ticker t) ends within `max_gap`
    sessions before `follower` (the CIK-verified rows of the same ticker t, or the next accepted
    segment) begins, and the close-to-close move across the junction is < `tol`."""
    if follower is None or not len(follower) or not len(run):
        return False
    sch = frames.schedule().index
    d0 = frames.to_naive_date(run.index)[-1]
    fd = frames.to_naive_date(follower.index)
    after = fd > d0
    if not after.any():
        return False
    d1 = fd[after][0]
    gap = int(((sch > d0) & (sch < d1)).sum())
    c0 = float(run["close"].iloc[-1])
    c1 = float(follower["close"][after].iloc[0])
    return gap <= max_gap and abs(np.log(c1 / c0)) < tol and frames.to_naive_date(run.index)[0] < d1


def _succession_ok(portal, pred: int, succ: int, ticker: str, yb: pd.DataFrame, idx) -> tuple[bool, str]:
    """Names of the two registrants must match (current or former SEC names) and the Yahoo raw
    close must be continuous across the hand-over window (no unexplained day move > 25 %)."""
    # an acquirer that ADOPTS the target's ticker (ACE -> CB 2016, SBC -> T 2005) carries its own
    # price line: Yahoo's pre-hand-over history is then the acquirer's, not the predecessor's.
    # Only a successor with no earlier ticker of its own (new holding company) continues the line.
    ec = idx.by_cik.get(succ)
    ea_all = idx.by_ticker.get(norm(ticker))
    ea_all = ea_all[ea_all["conf"] == 0]
    first_s_any = ea_all[ea_all["cik"] == succ]["date"].min()
    if ec is not None and len(ec):
        own = ec[(ec["conf"] == 0) & (ec["ticker"] != norm(ticker)) & (ec["date"] < first_s_any)]
        if len(own):
            return False, f"successor traded as {sorted(set(own['ticker']))[:3]} before (acquirer adopted the ticker)"
    if portal.store.exists("bars_1d", "dolthub", ticker):
        db = portal._read("bars_1d", "dolthub", ticker)
        lab = idx.label_dates(ticker, frames.to_naive_date(db.index))
        part = db[lab == pred]
        v = _agree(_close_series(part), _close_series(yb)) if len(part) else None
        if v is False:
            return False, "Yahoo line disagrees with the predecessor's as-traded DoltHub prices"
    np_, ns = cik_names(pred), cik_names(succ)
    if not np_ or not ns or not any(names_match(a, ns) for a in np_):
        return False, f"names differ {np_[:1]} vs {ns[:1]}"
    ea = idx.by_ticker.get(norm(ticker))
    ea = ea[ea["conf"] == 0]
    last_p = ea[ea["cik"] == pred]["date"].max()
    later = ea[(ea["cik"] == succ) & (ea["date"] >= last_p)]["date"]
    first_s = later.min() if len(later) else last_p
    lo, hi = min(last_p, first_s) - pd.Timedelta(days=15), max(last_p, first_s) + pd.Timedelta(days=15)
    c = _close_series(yb)
    w = c[(c.index >= lo) & (c.index <= hi)]
    if len(w) < 5:
        return False, "no Yahoo prices across the hand-over"
    r = np.log(w).diff().abs()
    acts = portal._read("actions", "yahoo", ticker)
    if len(acts):
        sp = pd.to_datetime(acts.loc[acts["type"].isin(["SPLIT", "SPINOFF"]), "ex_date"]).dt.normalize()
        r = r[~r.index.isin(sp)]
    if float(r.max()) > 0.25:
        return False, f"price discontinuity {float(r.max()):.0%} at hand-over"
    return True, f"names match, continuous price across {last_p.date()}..{first_s.date()}"


def build_series(portal, idx: IdentityIndex, cik: int, ticker: Optional[str] = None,
                 lineage_hint: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """-> (raw daily bars with columns 'src' / 'src_ticker', actions frame, log list).
    lineage_hint: a ticker of the CIK whose lineage `ticker` joins (membership continuity)."""
    st = portal.store
    lg: list = []
    main, extra = idx.lineage(cik, ticker or (idx.zl_tickers_of(cik) or [""])[-1])
    if lineage_hint:
        main = list(dict.fromkeys(idx.lineage(cik, lineage_hint)[0] + [norm(ticker or "")]))
    lg.append(f"lineage {main} (+ agreement-only {extra})")
    segs = []          # [rank, name, bars, ticker]
    pending = []       # candidates needing agreement / continuity
    for y in main + extra:
        if not st.exists("bars_1d", "yahoo", y):
            continue
        b = portal._read("bars_1d", "yahoo", y)
        if not len(b):
            continue
        owner = idx.cik_now(y)
        if b["close"].median() <= 1.0:
            lg.append(f"reject yahoo {y}: implausible price level {b['close'].median():.2f}")
            continue
        # QA feeds the identity decision (red-team #10): clustered spike-and-revert days mean two
        # securities are interleaved -> reject; isolated flags / OHLC violations are logged
        gl = _glitches(b["close"], portal)
        ng = int(gl.sum())
        if ng:
            # a cluster counts only TRUE glitches: the price returns to the same print the next
            # session (|r_t + r_t+1| < 0.5 %; bad prints, tick-quantized thin 1970s-90s data) and
            # the 20-session window shows no sustained move (|net| < 15 %). Genuine volatile
            # regimes (CVNA 2021-12 .. 2022 crash: alternating -10 % / +9 % days while falling
            # 98 %) are never cut.
            lr = np.log(b["close"]).diff()
            exact = gl & ((lr + lr.shift(-1)).abs() < 0.005)
            net = np.log(b["close"]).diff(20).abs()
            roll = exact.astype(int).rolling(20, min_periods=1).sum()
            roll = roll.where(net.fillna(0) < 0.15, 0)
            if int(roll.max()) >= 3:
                # clustered flags: two securities interleaved, or (usual for Yahoo) thin, tick-
                # quantized 1970s-90s history. Never serve the clustered stretch: cut the series
                # after the last cluster; if the cluster is in the last 2 years reject the ticker.
                cut = roll.index[roll.to_numpy() >= 3][-1]
                if cut > b.index[-1] - pd.Timedelta(days=730):
                    lg.append(f"reject yahoo {y}: QA collision ({ng} spike-and-revert days, clustered, recent)")
                    continue
                keep = b.index > cut
                lg.append(f"trim yahoo {y}: dropped {int((~keep).sum())} sessions up to {cut.date()} (clustered "
                          f"spike-and-revert flags: interleaved or thin early data)")
                b, gl = b[keep], gl[keep]
                ng = int(gl.sum())
        nv = int(qa.ohlc_violations(b).sum())
        if ng or nv:
            lg.append(f"qa yahoo {y}: {ng} isolated spike-and-revert days, {nv} OHLC violations (kept, flagged)")
        if owner is not None and owner != cik and y in main and idx.is_successor(cik, owner, y):
            ok, why = _succession_ok(portal, cik, owner, y, b, idx)
            if ok:
                lg.append(f"accept yahoo {y}: current owner CIK {owner} succeeded CIK {cik} on the same "
                          f"line ({why})")
                segs.append([0, f"yahoo:{y}", b, y])
                continue
            lg.append(f"skip yahoo {y}: owner CIK {owner} not a verified successor of {cik} ({why})")
            continue
        if owner == cik and y in main:
            segs.append([0, f"yahoo:{y}", b, y])
        elif owner == cik:
            pending.append([0, f"yahoo~:{y}", b, y])
        else:
            lg.append(f"skip yahoo {y}: current owner CIK {owner} != {cik}")
    # as-traded BY-TICKER sources (rows filed under the symbol that traded on each date): DoltHub
    # (2011 ->, but nothing for symbols gone before 2018) and Alpaca SIP daily bars (2016 ->, incl.
    # since-delisted symbols; providers/alpaca_eod.py). Rows on dates the ticker is evidenced to
    # denote this CIK are segments; rows of unknown identity need price agreement / continuity.
    for t, src in [(t_, s_) for s_ in AS_TRADED_SOURCES for t_ in main]:
        if not st.exists("bars_1d", src, t):
            continue
        b = portal._read("bars_1d", src, t)
        if not len(b):
            continue
        lab = idx.label_dates(t, frames.to_naive_date(b.index))
        if src != "alpaca_eod":
            gone = _off_tape(portal, t, frames.to_naive_date(b.index))
            if gone.any():
                # the consolidated tape shows NO trade under t on these dates (DoltHub files a renamed
                # company's history under its later ticker: DowDuPont 2017-09 .. 2019-05 under 'DD'
                # while old DuPont's 'DD' had stopped trading) -> identity of those rows unknown
                lg.append(f"{src} {t}: {int(gone.sum())} sessions not on the SIP tape under {t} -> identity unknown")
                lab = np.where(gone, 0, lab)
        nother = int(((lab != cik) & (lab != 0)).sum())
        if nother:
            lg.append(f"{src} {t}: {nother} sessions belong to other CIKs -> excluded")
        r_lab, r_unk = SOURCE_RANK[src], SOURCE_RANK[src + "?"]
        for name, part, rank in ((f"{src}:{t}", b[lab == cik], r_lab), (f"{src}?:{t}", b[lab == 0], r_unk)):
            if not len(part):
                continue
            gl = _glitches(part["close"], portal)
            ng = int(gl.sum())
            clustered = int(gl.astype(int).rolling(20, min_periods=1).sum().max()) if ng else 0
            if ng > max(3, int(0.01 * len(part))) or clustered >= 3:
                lg.append(f"reject {name}: collision test failed ({ng} spike-and-revert days of {len(part)})")
                continue
            if ng:
                bad = frames.to_naive_date(part.index[gl.to_numpy()])
                lg.append(f"{name}: dropped {ng} glitch sessions {[str(x.date()) for x in bad][:5]}")
                part = part[~gl.to_numpy()]
            (segs if rank == r_lab else pending).append([rank, name, part, t])
    # Tiingo lines are keyed by the ticker a security LAST traded under and carry its whole
    # history (providers/tiingo.py): usable for this CIK only if the line's owner (issuer the ticker
    # denoted at the line's end) is this CIK, on dates inside the CIK's evidence span; rows on dates
    # the ticker denoted another issuer are dropped, rows before the ticker denoted anyone (renamed
    # history) and every row of an ambiguous ticker (several Tiingo lines) need price agreement.
    first_seen = idx.first_seen.get(int(cik))
    for t in main:
        if not st.exists("bars_1d", "tiingo", t):
            continue
        b = portal._read("bars_1d", "tiingo", t)
        if not len(b):
            continue
        meta = st.meta("bars_1d", "tiingo", t) or {}
        d = frames.to_naive_date(b.index)
        line_end = pd.Timestamp(meta["line_end"]) if meta.get("line_end") else d[-1]   # stored rows may be truncated
        own = int(idx.label_dates(t, pd.DatetimeIndex([line_end]))[0])
        if not own and (pd.Timestamp.now().normalize() - line_end).days <= 10:
            own = idx.cik_now(t) or 0
        elif not own:
            # dead line whose issuer stopped filing > 120 days before its last trade (CVH, MHS, AYE
            # acquired): the ticker's last as-of issuer, if its evidence ends <= 548 days before the
            # line's end and no other issuer was evidenced on the ticker since
            ev = idx.by_ticker.get(norm(t))
            ea = ev[(ev["conf"] == 0) & (ev["date"] <= line_end + pd.Timedelta(days=EDGE_DAYS))] if ev is not None else None
            if ea is not None and len(ea):
                c_last = int(ea["cik"].iloc[-1])
                last_d = ea["date"].iloc[-1]
                later = ev[(ev["conf"] == 0) & (ev["date"] > last_d) & (ev["cik"] != c_last)]
                if (line_end - last_d).days <= MAX_EVIDENCE_GAP_DAYS and \
                        not len(later[later["date"] <= line_end + pd.Timedelta(days=EDGE_DAYS)]):
                    own = c_last
                    lg.append(f"tiingo {t}: dead line ending {line_end.date()} owned by CIK {c_last} (last evidence "
                              f"{last_d.date()}, no later issuer)")
        if own != cik:
            lg.append(f"skip tiingo {t}: line owner CIK {own or 'unknown'} != {cik}")
            continue
        lab = idx.label_dates(t, d)
        ok = (lab == cik) | (lab == 0)
        if first_seen is not None:
            ok &= np.asarray(d >= first_seen - pd.Timedelta(days=EDGE_DAYS))
        nother = int((lab != cik).sum() - (lab == 0).sum())
        if nother:
            lg.append(f"tiingo {t}: {nother} sessions belong to other CIKs -> excluded")
        amb = int(meta.get("n_lines", 1) or 1) > 1
        r_lab = SOURCE_RANK["tiingo?"] if amb else SOURCE_RANK["tiingo"]
        if amb:
            lg.append(f"tiingo {t}: {meta.get('n_lines')} Tiingo lines share the ticker -> agreement required")
        for name, part, rank in ((f"tiingo{'?' if amb else ''}:{t}", b[ok & (lab == cik)], r_lab),
                                 (f"tiingo?:{t}", b[ok & (lab == 0)], SOURCE_RANK["tiingo?"])):
            if not len(part):
                continue
            gl = _glitches(part["close"], portal)
            ng = int(gl.sum())
            clustered = int(gl.astype(int).rolling(20, min_periods=1).sum().max()) if ng else 0
            if ng > max(3, int(0.01 * len(part))) or clustered >= 3:
                lg.append(f"reject {name}: collision test failed ({ng} spike-and-revert days of {len(part)})")
                continue
            if ng:
                lg.append(f"{name}: dropped {ng} glitch sessions")
                part = part[~gl.to_numpy()]
            (segs if rank == SOURCE_RANK["tiingo"] else pending).append([rank, name, part, t])
    # --- agreement / renamed-history continuity gate
    accepted = sorted(segs, key=lambda x: x[0])
    for sgm in sorted(pending, key=lambda x: x[0]):
        verdicts = [_agree(_close_series(sgm[2]), _close_series(a[2])) for a in accepted]
        if any(v is False for v in verdicts):
            lg.append(f"reject {sgm[1]}: disagrees with a verified segment")
            continue
        if any(v for v in verdicts):
            accepted.append(sgm)
            lg.append(f"accept {sgm[1]} by price agreement")
            continue
        # renamed history (DoltHub files a renamed company's past under its later ticker, e.g. CBS
        # 2011-2017 under VIAC): the run must precede the CIK's own use of that ticker, the CIK must
        # be evidenced under another ticker during the run, and the next verified segment must
        # continue the price within 10 %
        cands = [a[2] for a in accepted if len(a[2]) and a[2].index[0] > sgm[2].index[-1]]
        nxt = min(cands, key=lambda z: z.index[0]) if cands else None
        rd = frames.to_naive_date(sgm[2].index)
        own_t = idx.by_ticker.get(sgm[3])
        later_use = own_t is not None and len(own_t[(own_t["cik"] == cik) & (own_t["conf"] == 0)
                                                    & (own_t["date"] > rd[-1])])
        ce = idx.by_cik.get(cik)
        other_ev = ce is not None and len(ce[(ce["conf"] == 0) & (ce["ticker"] != sgm[3]) &
                                            (ce["date"] >= rd[0]) & (ce["date"] <= rd[-1])])
        if "?:" in sgm[1] and sgm[1].split("?:")[0] in AS_TRADED_SOURCES + ("tiingo",) and later_use and other_ev and _junction_ok(sgm[2], nxt):
            # DoltHub (like Yahoo) files a renamed company's history under its later ticker
            accepted.append(sgm)
            lg.append(f"accept {sgm[1]} as renamed history (CIK lineage + price continuity at the junction)")
            continue
        lg.append(f"reject {sgm[1]}: identity unknown and no agreement with a verified segment")
    keep = []
    for sgm in sorted(accepted, key=lambda x: x[0]):
        bad = [k[1] for k in keep if _agree(_close_series(sgm[2]), _close_series(k[2])) is False]
        if bad:
            lg.append(f"reject {sgm[1]}: disagrees with {bad[0]} on overlapping sessions")
        else:
            keep.append(sgm)
    if not keep:
        return frames.empty_bars(), pd.DataFrame(), lg + ["FAIL CLOSED: no verified segment"]
    rows = []
    for rank, name, b, t in keep:
        x = b[["open", "high", "low", "close", "volume", "session", "known_at"]].copy()
        x["src"] = name.split(":")[0].rstrip("?~")
        x["src_ticker"] = t
        x["rank"] = rank
        rows.append(x)
    m = pd.concat(rows).reset_index().sort_values(["start", "rank"])
    m = m.drop_duplicates("start", keep="first").set_index("start").sort_index()
    acts = _segment_actions(portal, m, keep)
    m, acts = _validate(m, acts, lg)
    return m.drop(columns=["rank"]), acts, lg


def _segment_actions(portal, m: pd.DataFrame, keep) -> pd.DataFrame:
    parts = []
    d = frames.to_naive_date(m.index)
    seen = set()
    for rank, name, b, t in keep:
        src = name.split(":")[0].rstrip("?~")
        if (src, t) in seen:                       # labelled + unlabelled chunks of one source/ticker:
            continue                               # its actions once (ETFC 2010 1:10 was applied twice)
        seen.add((src, t))
        # by-ticker as-traded segments (DoltHub, Alpaca) take DoltHub's dividend / split tables,
        # which are keyed by the ticker as traded and include since-delisted symbols
        if src == "tiingo":                        # Tiingo's own divCash / splitFactor (as traded)
            from .providers.tiingo import actions as _tiingo_actions
            a = _tiingo_actions(portal, t)
        else:
            a = portal._yahoo_actions(t) if src == "yahoo" else _dolt_actions(portal, t)
        if a is None or not len(a):
            continue
        a = a.copy()
        # an action belongs to this segment when the ex-date bar is served from it
        sel = m[(m["src_ticker"] == t) & (m["src"] == src)]
        days = set(frames.to_naive_date(sel.index))
        ex = pd.to_datetime(a["ex_date"]).dt.normalize()
        a = a[ex.isin(days)]
        if len(a):
            parts.append(a.assign(src=src, src_ticker=t))
    # DoltHub segments carry no spin-offs: add known ones (value-based) on their days
    from .actions_classify import known_spinoff_rows
    have = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["type", "ex_date"])
    seen = set()
    for rank, name, b, t in keep:
        src = name.split(":")[0].rstrip("?~")
        if src not in AS_TRADED_SOURCES or (src, t) in seen:
            continue
        seen.add((src, t))
        sel = m[(m["src_ticker"] == t) & (m["src"] == src)]
        extra = known_spinoff_rows(t, have, sel, portal._child_close)
        if len(extra):
            parts.append(extra.assign(src="known_spinoff", src_ticker=t))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _validate(m: pd.DataFrame, acts: pd.DataFrame, lg: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = frames.to_naive_date(m.index)
    close = pd.Series(m["close"].to_numpy(), index=d)
    open_ = pd.Series(m["open"].to_numpy(), index=d)
    # expected overnight factor from actions on each date
    expect = pd.Series(1.0, index=d)
    if len(acts):
        for r in acts.itertuples():
            ex = pd.Timestamp(r.ex_date).normalize()
            if ex not in expect.index:
                continue
            if r.type == "SPLIT" and r.ratio and r.ratio > 0:
                expect.loc[ex] *= 1.0 / r.ratio
            elif r.type in ("CASH_DIVIDEND", "SPECIAL_DIVIDEND", "SPINOFF") and r.amount > 0:
                i = expect.index.get_loc(ex)
                if i > 0:
                    expect.loc[ex] *= max(1e-6, 1 - r.amount / close.iloc[i - 1])
    # validate splits against the raw price ratio
    if len(acts):
        drop = []
        for i, r in acts.iterrows():
            if r["type"] != "SPLIT":
                continue
            ex = pd.Timestamp(r["ex_date"]).normalize()
            if ex not in close.index:
                continue
            j = close.index.get_loc(ex)
            if j == 0:
                continue
            g = open_.iloc[j] / close.iloc[j - 1]
            # same-day distributions (spin-offs encoded with a reverse split: EXPE 2011-12-21 1:2 +
            # TRIP, ITT 2011, DD 2019) move the price too: compare against both hypotheses
            q = 1.0
            same = acts[(pd.to_datetime(acts["ex_date"]).dt.normalize() == ex) &
                        acts["type"].isin(["SPINOFF", "SPECIAL_DIVIDEND", "CASH_DIVIDEND"])]
            for amt in same["amount"].fillna(0):
                if amt > 0:
                    q *= max(1e-6, 1 - amt / close.iloc[j - 1])
            e_split, e_none = abs(np.log(g * r["ratio"] / q)), abs(np.log(g / q))
            # (a) clear miss; (b) the prices fit "no split" far better than the split: a fractional
            # split that only re-encodes a same-day spin-off already carried as SPINOFF (Yahoo GE
            # 2024-04-02 1.253 + GEV spin-off -> shares were multiplied twice), or a split with no
            # price gap at all (SSP 2025-12-08 1.33)
            if np.isfinite(g) and ((e_split > 0.35 and e_none < 0.15) or (e_none < 0.08 and e_split > 2 * e_none + 0.1)):
                drop.append(i)
                expect.loc[ex] = 1.0
                lg.append(f"drop split {ex.date()} ratio {r['ratio']}: no price jump (gap {g:.3f})")
        acts = acts.drop(index=drop)
    ret = np.log(close / close.shift(1)) - np.log(expect)
    src = (m["src"] + ":" + m["src_ticker"]).to_numpy()
    rank = m["rank"].to_numpy()
    mad = ret.abs().rolling(60, min_periods=20).median().shift(1).bfill()
    big = (ret.abs() > np.maximum(np.log(1.2), 6 * mad)).to_numpy().copy()
    sw = np.r_[False, src[1:] != src[:-1]]
    dropmask = np.zeros(len(m), dtype=bool)
    # infer unrecorded splits inside DoltHub chunks (as-traded data, split table only from 2014)
    inferred = []
    for j in np.where(big & ~sw)[0]:
        if m["src"].iloc[j] not in AS_TRADED_SOURCES:
            continue
        g = open_.iloc[j] / close.iloc[j - 1]
        k = np.argmin(np.abs(np.log(g) - np.log(1 / INFER_SPLITS)))
        vol = m["volume"].to_numpy()
        vb = np.nanmedian(vol[max(0, j - 10):j]) if j > 0 else np.nan
        va = np.nanmedian(vol[j:j + 10])
        vr = va / vb if vb and np.isfinite(vb) and vb > 0 else np.nan
        r_k = float(INFER_SPLITS[k])
        vol_ok = np.isfinite(vr) and (0.4 * r_k <= vr <= 2.5 * r_k)
        if abs(np.log(g) - np.log(1 / r_k)) < 0.04 and vol_ok:
            ratio = r_k
            inferred.append({"symbol": m["src_ticker"].iloc[j], "type": "SPLIT", "ex_date": d[j],
                             "known_at": m["known_at"].iloc[j - 1], "ratio": ratio, "amount": 0.0,
                             "fidelity": int(Fidelity.T2_PROXY), "src": "inferred", "src_ticker": m["src_ticker"].iloc[j]})
            big[j] = False
            lg.append(f"infer split {d[j].date()} ratio {ratio:g} from DoltHub price jump {g:.3f}")
    chunk_id = np.cumsum(sw)
    # identity break INSIDE a by-ticker Alpaca / Tiingo chunk: an unexplained x3 (or /3) close-to-close
    # move (no split / distribution, no inferable split) is a different security under the same
    # ticker+CIK labels (AmSurg -> Envision 'EVHC' 2016-12-02 x3.1, Valaris new equity 'VAL'
    # 2021-05-03 x72) -> the rest of that chunk is dropped (fail closed)
    srcs = m["src"].to_numpy()
    for j in np.where(big & ~sw)[0]:
        if srcs[j] in ("alpaca_eod", "tiingo") and abs(ret.iloc[j]) > np.log(3.0) and not dropmask[j]:
            tail = (chunk_id == chunk_id[j]) & (np.arange(len(m)) >= j)
            dropmask |= tail
            big[j] = False
            lg.append(f"identity break in {src[j]} at {d[j].date()} (move {np.exp(ret.iloc[j]):.2f}x unexplained): "
                      f"dropped {int(tail.sum())} later sessions of that chunk")
    # splice points: unexplained jump -> drop the lower-priority chunk
    sch_idx = frames.schedule().index
    for j in np.where(big & sw)[0]:
        prev_rank, cur_rank = rank[j - 1], rank[j]
        # a junction across a long data gap is not a splice: allow the move to scale with the gap
        gap = int(((sch_idx > d[j - 1]) & (sch_idx < d[j])).sum())
        if gap > 10 and abs(ret.iloc[j]) < np.log(1.2) * np.sqrt(gap + 1):
            continue
        cid = chunk_id[j] if cur_rank >= prev_rank else chunk_id[j - 1]
        other = chunk_id[j - 1] if cid == chunk_id[j] else chunk_id[j]
        n_c, n_o = int((chunk_id == cid).sum()), int((chunk_id == other).sum())
        if n_c >= 250 and n_c >= 20 * n_o:
            cid = other          # never let a stub (e.g. a 1-bar post-delisting quote) evict years of data
        dropmask |= chunk_id == cid
        lg.append(f"splice reject at {d[j].date()}: {src[j - 1]} -> {src[j]} move {ret.iloc[j]:+.3f} unexplained")
    # single-session fills from a lower-priority source must be within 10 % of both neighbours
    for j in range(1, len(m) - 1):
        if sw[j] and src[j + 1] != src[j] and rank[j] > min(rank[j - 1], rank[j + 1]):
            c = close.iloc[j]
            if abs(c / close.iloc[j - 1] - 1) > 0.10 or abs(close.iloc[j + 1] / c - 1) > 0.10:
                dropmask[j] = True
                lg.append(f"drop single-day fill {d[j].date()} from {src[j]}")
    if dropmask.any():
        lg.append(f"dropped {int(dropmask.sum())} sessions at splice validation")
        m = m[~dropmask]
    if inferred:
        acts = pd.concat([acts, pd.DataFrame(inferred)], ignore_index=True) if len(acts) else pd.DataFrame(inferred)
    return m, acts


# ---------------------------------------------------------------------------- names
_STOP = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "the", "group",
         "holdings", "holding", "and", "of", "nv", "sa", "ag", "lp", "llc", "trust", "international", "intl",
         "new", "de", "com", "cos", "companies", "class", "a", "b"}


def name_tokens(name: str) -> set:
    import re
    toks = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " ")).split()
    return {t for t in toks if t not in _STOP and len(t) > 1}


def _squeezed(name: str) -> str:
    import re
    toks = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " ")).split()
    return "".join(t for t in toks if t not in _STOP)


def names_match(hint: str, names: list[str]) -> bool:
    h = name_tokens(hint)
    if not h:
        return True
    hs = _squeezed(hint)
    for n in names:
        t = name_tokens(n)
        if not t:
            continue
        if h & t and (len(h & t) / min(len(h), len(t)) >= 0.5):
            return True
        ts = _squeezed(n)          # 'ExxonMobil Holdings' vs 'Exxon Mobil Corp', 'Coca-Cola' vs 'COCA COLA'
        if len(hs) >= 4 and len(ts) >= 4 and (hs in ts or ts in hs):
            return True
    return False


_NAMES: dict = {}
_DATED_NAMES: dict = {}


def cik_name_at(cik: int, date) -> Optional[str]:
    """SEC registrant name in force on `date` (submissions `formerNames` from/to; dates before the
    earliest recorded name get the earliest one). None if unavailable."""
    import json
    from .http import FetchError, client
    cik = int(cik)
    if cik not in _DATED_NAMES:
        rows = []
        try:
            # cache-only: Portal use never fetches (precached by `fetch --set core`, task
            # sec_submissions); a missing name = unknown = fail closed
            body = client().get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", allow_404=True, ttl=None,
                                cache_only=True)
            if body:
                d = json.loads(body)
                for f in d.get("formerNames", []) or []:
                    try:
                        rows.append((pd.Timestamp(f["from"][:10]), pd.Timestamp(f["to"][:10]), f.get("name", "")))
                    except (KeyError, TypeError, ValueError):
                        continue
                rows.append((max([r[1] for r in rows], default=pd.Timestamp("1900-01-01")),
                             pd.Timestamp("2262-01-01"), d.get("name") or ""))
        except FetchError:
            rows = []
        _DATED_NAMES[cik] = sorted(rows)
    rows = _DATED_NAMES[cik]
    if not rows:
        return None
    d = pd.Timestamp(date).normalize()
    for a, b, n in rows:
        if a <= d <= b:
            return n
    return rows[0][2] if d < rows[0][0] else rows[-1][2]


def cik_names(cik: int) -> list[str]:
    """Current + former SEC registrant names (data.sec.gov submissions; cached on disk)."""
    if cik in _NAMES:
        return _NAMES[cik]
    import json
    from .http import FetchError, client
    names: list[str] = []
    try:
        body = client().get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", allow_404=True,
                            ttl=None, cache_only=True)
        if body:
            d = json.loads(body)
            names = [d.get("name") or ""] + [f.get("name", "") for f in d.get("formerNames", [])]
    except FetchError:
        names = []
    _NAMES[cik] = [n for n in names if n]
    return _NAMES[cik]

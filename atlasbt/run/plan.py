"""`plan`: resolve every strategy's data requirements against the Portal and derive its window.

Rules (documented in docs/RUNNING.md §2):

* Requirements are read from a strategy INSTANCE built with the sheet (baseline) params, because some
  families derive ``universe``/``requires`` from params (options: the ``underlying`` param).
* A critical requirement is satisfied when ``portal.has`` is True and ``portal.fidelity_of`` is not NA and
  not worse than ``Requirement.min_fidelity`` -- the same test the engine applies at run start.
  Additional facts the engine cannot see from ``has`` are checked here and reported as NOT_TESTABLE with
  the reason: intraday OPTION quotes (none exist in the free data), quote requirements whose cached quote
  history covers fewer sessions than the minimum window, ``inav``-type kinds without a source.
* Window start = the first session at which every critical input exists:
  - bars of explicitly named symbols: the latest first bar among them; bars declared for the whole
    universe (``symbols=("*",)``) are checked like bars without symbols (next two rules);
  - bars of the strategy universe (no explicit symbols): all members for small universes (<= 6 names);
    otherwise the first date by which >= 80% of the members have bars (the
    strategies trade late listers once they have bars);
  - PIT universe 'SP500': the first month start at which PIT price coverage >= min_coverage (default 0.95,
    paired with the runner's CoverageFilteredPortal; 1.0 = the engine's strict every-member rule, which is
    never met under the fail-closed identity layer);
  - events: first event known in the store; fundamentals: first XBRL filing (2009-07);
  - real option chains (min_fidelity <= T2): first DoltHub snapshot of the underlying.
  When the binding constraint is a symbol's first bar (a listing date), a warm-up of
  ``WARMUP_SESSIONS`` (252) sessions is added for daily+ strategies, because no pre-listing history exists
  for indicators. Intraday strategies get no warm-up (their indicators read daily/intraday history before
  the window through the PIT portal).
  - intraday bars / quotes are scanned from 2016-01-01 (Alpaca SIP history, DATA.md §4.15);
  - PIT universe (SP500, SP500_TOPn_DV) under an INTRADAY strategy: per-session coverage of THAT
    session's members (bars of every no-symbol bars frequency ON the session and the engine's other
    member checks) >= min_coverage at every sampled session of the window, plus the engine's
    first-10-session PRE/POST probe of the kept members (``PlanContext.pit_window``);
  - quotes: first quote session common to the symbols (static) or of the UNION of a PIT universe's
    members (pre-open windows exist only on member sessions); Alpaca quote sessions are the recorded
    windows of ``reference/alpaca/quote_windows`` (the Portal serves nothing outside them).
* Window end = the last cached session common to the critical bar inputs (and the last quote session).
* Minimum window: 252 sessions for daily+ strategies, 10 sessions for intraday ones.
* The memo (results/plan_cache.json) is dropped on a new identity build / membership / events, and its
  intraday entries on any change of the intraday bar / quote / option-quote stores.
"""
from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .common import DEFAULT_RESULTS, is_intraday, read_json, write_json

WARMUP_SESSIONS = 252
MIN_SESSIONS_DAILY = 252
MIN_SESSIONS_INTRADAY = 10
SMALL_UNIVERSE = 6
UNIVERSE_SHARE = 0.80
FUNDAMENTALS_START = pd.Timestamp("2009-07-22")
#: intraday bar / quote history scanned by the planner: Alpaca SIP starts 2016-01-04 (DATA.md §4.15); the
#: Yahoo-era ``now - 800 days`` start cut every Alpaca-backed intraday window to the last ~2 years
INTRADAY_HISTORY_START = pd.Timestamp("2016-01-01", tz="UTC")
#: PIT intraday universes: coverage is sampled every N sessions and refined by bisection at the edges
PIT_STEP_SESSIONS = 10
#: the engine probes PRE/POST bars and quotes of every kept member over the run's first N sessions
#: (Backtester._probe_sessions n_probe)
PROBE_SESSIONS = 10
ET = "America/New_York"

#: rough cost model (seconds) per config: SETUP + sessions * (BASE[freq] + PER_SYMBOL[freq] * n_symbols)
SETUP_SEC = 15.0
BASE_SEC = {"1d": 0.035, "1mo": 0.03, "1w": 0.03, "5m": 0.25, "1m": 0.8}
PER_SYMBOL_SEC = {"1d": 0.006, "1mo": 0.004, "1w": 0.004, "5m": 0.04, "1m": 0.15}
OPTION_SESSION_SEC = 0.12


def _today_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _data_version(portal) -> str:
    try:
        from atlasbt.data.identity import BUILD_VERSION
    except Exception:            # pragma: no cover
        BUILD_VERSION = "?"
    mem = ""
    try:
        meta = portal.store.meta("reference", "sp500", "membership") or {}
        mem = str(meta.get("fetched_at", ""))
    except Exception:
        pass
    ev = ""
    try:
        root = Path(portal.store.root) / "events"
        ev = str(max((f.stat().st_mtime_ns for f in root.rglob("*.parquet")), default=0))
    except Exception:
        pass
    return f"identity={BUILD_VERSION};membership={mem};events={ev}"


def _intraday_version(portal) -> str:
    """Stamp of the intraday stores (every ``bars_<intraday>`` kind, ``quotes``, ``reference/alpaca`` with
    the recorded quote windows, the option quote / option bar stores -- Databento OPRA answers
    ``has(member, 'options', '1m')``, one of the engine's member checks): parquet file count and newest
    mtime per kind. A fetch that adds or rewrites intraday bars / quotes / option quotes changes it, which
    drops the memoised intraday facts (``bars|SYM|1m``, ``quotes|SYM``, ``pit|window|...`` computed before
    the fetch, possibly as None / NOT_TESTABLE) but keeps the daily ones."""
    try:
        root = Path(portal.store.root)
    except Exception:
        return ""
    if not root.is_dir():
        return ""
    dirs = sorted(d for d in root.glob("bars_*") if d.is_dir() and d.name != "bars_1d")
    dirs += [root / "quotes", root / "reference" / "alpaca"]
    dirs += sorted(d for d in root.glob("option_bars_*") if d.is_dir()) + [root / "option_quotes"]
    parts = []
    for d in dirs:
        if not d.is_dir():
            continue
        n, newest = 0, 0
        for f in d.rglob("*.parquet"):
            try:
                newest = max(newest, f.stat().st_mtime_ns)
            except OSError:
                continue
            n += 1
        parts.append(f"{d.relative_to(root)}={n}:{newest}")
    return ";".join(parts)


def _is_intraday_key(key: str) -> bool:
    """Memo keys whose value depends on intraday bars or quotes."""
    parts = str(key).split("|")
    if parts[0] in ("quotes", "pit"):
        return True
    return parts[0] in ("bars", "spans") and len(parts) >= 3 and parts[2] != "1d"


def _norm_freq(f, default: str) -> str:
    f = f or default
    return "1d" if f in (None, "1mo", "1w", "event") else f


def _in_spans(spans, a: str, b: str) -> bool:
    """True if a session span [x, y] (ISO dates) intersects [a, b]."""
    return any(x <= b and y >= a for x, y in spans or ())


def _bisect_first(lo: int, hi: int, ok) -> int:
    """First index in (lo, hi] with ok() True, given ok(lo) False and ok(hi) True."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _bisect_last(lo: int, hi: int, ok) -> int:
    """Last index in [lo, hi) with ok() True, given ok(lo) True and ok(hi) False."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _is_pit_universe(name: Optional[str]) -> bool:
    """Universes whose membership changes by session (S&P 500 PIT and its top-N dollar-volume subsets)."""
    import re

    n = (name or "").upper()
    return n in ("SP500", "S&P500", "SPX_MEMBERS") or bool(re.fullmatch(r"SP500_TOP\d+_DV", n))


class PlanContext:
    """Memoized availability facts (persisted in results/plan_cache.json)."""

    def __init__(self, portal, results_dir: Path = DEFAULT_RESULTS, min_coverage: float = 1.0):
        self.portal = portal
        self.min_coverage = float(min_coverage)
        self.path = Path(results_dir) / "plan_cache.json"
        self.cache: dict = read_json(self.path, {}) or {}
        self.dirty = False
        # memoized availability facts are only valid for the data they were computed on: the identity
        # layer's build version and the S&P membership table (survivorship patches move the SP500
        # coverage start) -> a changed data version drops the memo
        dv = _data_version(portal)
        if self.cache.get("_data_version") != dv:
            self.cache = {"_data_version": dv}
            self.dirty = True
        # intraday facts are also only valid for the intraday stores they were computed on: a fetch of
        # Alpaca bars / quotes after a plan must not leave `bars|SYM|1m` / `quotes|SYM` memoised as None
        iv = _intraday_version(portal)
        if self.cache.get("_intraday_version") != iv:
            self.cache = {k: v for k, v in self.cache.items() if not _is_intraday_key(k)}
            self.cache["_intraday_version"] = iv
            self.dirty = True
        self._qwin: Optional[dict] = None
        from atlasbt.core.clock import nyse

        self.cal = nyse()

    def save(self):
        if self.dirty:
            write_json(self.path, self.cache)
            self.dirty = False

    def _memo(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
            self.dirty = True
        return self.cache[key]

    # ---------------------------------------------------------------- bars
    def bar_range(self, sym: str, freq: str) -> Optional[list]:
        """[first_session, last_session] (ISO dates) of bars, or None."""
        freq = "1d" if freq in (None, "1mo", "1w", "event") else freq

        def f():
            p = self.portal
            now = _today_utc()
            start = pd.Timestamp("1960-01-01", tz="UTC") if freq == "1d" else INTRADAY_HISTORY_START
            try:
                if hasattr(p, "bars_coverage"):        # f39aa6b+: computed BY bars() (identity-verified)
                    c = p.bars_coverage(sym, start, now, now, freq=freq)
                    if not c.get("available"):
                        return None
                    i0 = pd.Timestamp(c["first"]).tz_convert("America/New_York")
                    i1 = pd.Timestamp(c["last"]).tz_convert("America/New_York")
                    return [str(i0.date()), str(i1.date())]
                if not p.has(sym, "bars", freq):
                    return None
                df = p.bars(sym, freq, start, now, as_of=now, adjusted=False)
            except Exception:
                return None
            if df is None or not len(df):
                return None
            idx = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
            return [str(idx[0].date()), str(idx[-1].date())]

        return self._memo(f"bars|{sym}|{freq}", f)

    def _alpaca_quote_days(self) -> dict:
        """{SYMBOL: set of ISO session dates} of the recorded Alpaca quote windows with at least one NBBO
        update (``reference/alpaca/quote_windows``, one row per (symbol, window)). The Portal serves Alpaca
        quotes only inside these windows, so they -- not the quotes parquet -- define the quote sessions."""
        if self._qwin is None:
            tab: dict = {}
            try:
                st = self.portal.store
                w = st.read("reference", "alpaca", "quote_windows") if st.exists("reference", "alpaca",
                                                                                "quote_windows") else None
            except Exception:
                w = None
            if w is not None and len(w) and {"symbol", "w0"} <= set(w.columns):
                if "n" in w.columns:
                    w = w[pd.to_numeric(w["n"], errors="coerce").fillna(0) > 0]
                t = pd.DatetimeIndex(pd.to_datetime(w["w0"], utc=True)).tz_convert(ET)
                days = pd.Series(t.strftime("%Y-%m-%d"), index=w.index)
                for s_, g in days.groupby(w["symbol"].astype(str).str.upper()):
                    tab[s_] = set(g)
            self._qwin = tab
        return self._qwin

    def quote_range(self, sym: str) -> Optional[list]:
        """Sorted ISO sessions on which the Portal serves quotes for ``sym``: Alpaca from the recorded
        quote windows, the other sources from their stored quotes."""
        def f():
            store = self.portal.store
            days = set(self._alpaca_quote_days().get(str(sym).upper(), ()))
            for src in ("databento", "iex", "dukascopy"):
                try:
                    if store.exists("quotes", src, sym):
                        q = store.read("quotes", src, sym)
                        idx = pd.DatetimeIndex(q.index if not isinstance(q.index, pd.RangeIndex) else q["ts"])
                        if idx.tz is None:
                            idx = idx.tz_localize("UTC")
                        days |= {str(d.date()) for d in idx.tz_convert(ET).normalize().unique()}
                except Exception:
                    continue
            return sorted(days) if days else None

        return self._memo(f"quotes|{sym}", f)

    def bar_spans(self, sym: str, freq: str) -> list:
        """Maximal runs of consecutive NYSE sessions with ``freq`` bars, as [[first, last], ...] (ISO).
        Alpaca intraday bars exist only around a stock's fetched sessions (top-60 sessions -30/+5), so
        "has a bar in [a, b]" -- the engine's windowed has() -- is "a span intersects [a, b]". Daily
        series are contiguous: one span = bar_range."""
        freq = _norm_freq(freq, "1d")
        if freq == "1d":
            br = self.bar_range(sym, "1d")
            return [br] if br else []

        def f():
            p = self.portal
            now = _today_utc()
            try:
                df = p.bars(sym, freq, INTRADAY_HISTORY_START, now, as_of=now, adjusted=False)
            except Exception:
                df = None
            finally:
                ev = getattr(p, "_evict_intraday", None)
                if callable(ev):
                    try:
                        ev(sym)
                    except Exception:
                        pass
            if df is None or not len(df):
                return []
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx
            days = idx.tz_convert(ET).tz_localize(None).normalize().unique().sort_values()
            sess = self.cal.sessions(days[0], days[-1])
            pos = sess.get_indexer(days)
            pos = pos[pos >= 0]
            if not len(pos):
                return []
            br = np.flatnonzero(np.diff(pos) > 1)
            a, b = np.r_[0, br + 1], np.r_[br, len(pos) - 1]
            return [[str(sess[pos[i]].date()), str(sess[pos[j]].date())] for i, j in zip(a, b)]

        return self._memo(f"spans|{sym}|{freq}", f)

    def _pit_sessions(self):
        end = _today_utc().tz_localize(None).normalize()
        sess = self.cal.sessions(INTRADAY_HISTORY_START.tz_localize(None), end)
        idx = list(range(0, len(sess), PIT_STEP_SESSIONS))
        if idx and idx[-1] != len(sess) - 1:
            idx.append(len(sess) - 1)
        return sess, idx

    def pit_members(self, name: str) -> list:
        """Union of a PIT universe's members over the sampled intraday-era sessions."""
        def f():
            sess, idx = self._pit_sessions()
            out: set = set()
            for i in idx:
                out |= set(self.universe_at(name, sess[i]))
            return sorted(out)

        return self._memo(f"pit|members|{name}|{PIT_STEP_SESSIONS}", f)

    def pit_window(self, name: str, freqs: tuple, kinds: tuple = (), probe_freqs: tuple = (),
                   probe_quotes: bool = False) -> dict:
        """Intraday window of a PIT universe by the engine's coverage rule, applied per session to THAT
        session's members (like sp500_full_coverage_start for daily): a member counts at session d when
        it has bars of every ``freqs`` frequency ON d and the portal has every non-bars ``kinds``
        requirement (the engine's member verification). Coverage is sampled every PIT_STEP_SESSIONS
        sessions; the longest run of samples with coverage >= min_coverage (ties: the latest) is the
        window, its edges refined by bisection. Coverage of bars ON d is a lower bound of the engine's
        (bars anywhere in the run window), so the engine accepts every sampled session of the window.
        With PRE/POST requirements (``probe_freqs`` bars, ``probe_quotes``) the start also satisfies the
        engine's probe: every member the engine keeps at the first session has those bars / quotes within
        the first PROBE_SESSIONS sessions. Returns {start, end, why, cov_min} or {reason}."""
        th = self.min_coverage
        kinds_s = ";".join(f"{k}:{q or ''}" for k, q in kinds)
        key = (f"pit|window|{name}|{','.join(freqs)}|{kinds_s}|probe={','.join(probe_freqs)}"
               f"{'+quotes' if probe_quotes else ''}|{th:.3f}|{PIT_STEP_SESSIONS}")

        def f():
            sess, idx = self._pit_sessions()
            if not idx:
                return {"reason": f"{name}: no sessions in the intraday era"}
            iso = [str(d.date()) for d in sess]
            has_memo: dict = {}

            def kinds_ok(m) -> bool:
                for k, q in kinds:
                    if (m, k, q) not in has_memo:
                        try:
                            has_memo[(m, k, q)] = bool(self.portal.has(m, k, q))
                        except Exception:
                            has_memo[(m, k, q)] = False
                    if not has_memo[(m, k, q)]:
                        return False
                return True

            stat_memo: dict = {}

            def stat(i):
                if i not in stat_memo:
                    mem = self.universe_at(name, sess[i])
                    miss = [m for m in mem if not (all(_in_spans(self.bar_spans(m, q), iso[i], iso[i]) for q in freqs)
                                                   and kinds_ok(m))]
                    stat_memo[i] = (len(mem), miss)
                return stat_memo[i]

            def cov(i):
                n, miss = stat(i)
                return (n - len(miss)) / n if n else float("nan")

            def good(i):
                return stat(i)[0] > 0 and cov(i) >= th

            flags = [good(i) for i in idx]
            best, j = None, 0
            while j < len(idx):
                if not flags[j]:
                    j += 1
                    continue
                k = j
                while k + 1 < len(idx) and flags[k + 1]:
                    k += 1
                if best is None or k - j >= best[1] - best[0]:
                    best = (j, k)
                j = k + 1
            if best is None:
                vals = [(cov(i), i) for i in idx if stat(i)[0]]
                if not vals:
                    return {"reason": f"{name} has no members in the intraday era"}
                c, i = max(vals)
                n, miss = stat(i)
                return {"reason": (f"{name}: per-session {'+'.join(freqs)} coverage never reaches "
                                   f"{th:.0%} (best {c:.0%} on {iso[i]}: {len(miss)}/{n} members without bars that "
                                   f"session, e.g. {', '.join(miss[:4])}); the engine refuses below "
                                   f"min_universe_coverage")}
            j, k = best
            i0 = _bisect_first(idx[j - 1], idx[j], good) if j > 0 else idx[j]
            i1 = _bisect_last(idx[k], idx[k + 1], good) if k + 1 < len(idx) else idx[k]
            why = f"{name} per-session {'+'.join(freqs)} coverage >= {th:.0%}"
            if probe_freqs or probe_quotes:
                def probe_ok(i) -> bool:
                    d0, d1 = iso[i], iso[i1]
                    pe = iso[min(i + PROBE_SESSIONS - 1, len(iso) - 1)]
                    for m in self.universe_at(name, sess[i]):
                        if not (all(_in_spans(self.bar_spans(m, q), d0, d1) for q in freqs) and kinds_ok(m)):
                            continue                  # not kept by the engine: never probed
                        if any(not _in_spans(self.bar_spans(m, q), d0, pe) for q in probe_freqs):
                            return False
                        if probe_quotes and not any(d0 <= d <= pe for d in (self.quote_range(m) or ())):
                            return False
                    return True

                if not probe_ok(i0):
                    prev, hit = i0, None
                    for i in [x for x in idx if i0 < x < i1] + [i1]:
                        if probe_ok(i):
                            hit = i
                            break
                        prev = i
                    if hit is None:
                        return {"reason": (f"{name}: members kept at the first session lack PRE/POST data in "
                                           f"the first {PROBE_SESSIONS} sessions of every candidate start "
                                           f"(engine probe)")}
                    i0 = _bisect_first(prev, hit, probe_ok)
                    why += f" + engine PRE/POST probe of kept members ({PROBE_SESSIONS} sessions)"
            run = sorted({i0, i1, *[i for i in idx if i0 <= i <= i1]})

            def engine_cov(i):
                # the engine keeps a member with bars anywhere in the run window; below 100 % it flags the
                # run T2_PROXY ("universe coverage ... members without verified data excluded")
                mem = self.universe_at(name, sess[i])
                kept = [m for m in mem if all(_in_spans(self.bar_spans(m, q), iso[i0], iso[i1]) for q in freqs)
                        and kinds_ok(m)]
                return len(kept) / len(mem) if mem else 1.0

            return {"start": iso[i0], "end": iso[i1], "why": why,
                    "cov_min": float(np.nanmin([cov(i) for i in run])),
                    "engine_cov_min": float(min(engine_cov(i) for i in run))}

        return self._memo(key, f)

    def event_first(self, kind: str) -> Optional[str]:
        def f():
            p = self.portal
            srcs = getattr(p, "EVENT_SOURCES", {}).get(kind.upper(), [])
            best = None
            for k, src, name in srcs:
                try:
                    if not p.store.exists(k, src, name):
                        continue
                    df = p.store.read(k, src, name)
                    col = "known_at" if "known_at" in df.columns else None
                    if col is None:
                        for c in ("date", "ts", "effective_at"):
                            if c in df.columns:
                                col = c
                                break
                    s = pd.to_datetime(df[col], utc=True) if col else pd.DatetimeIndex(df.index)
                    v = pd.Timestamp(s.min())
                    if v.tzinfo is not None:
                        v = v.tz_convert("America/New_York").tz_localize(None)
                    best = v if best is None or v < best else best
                except Exception:
                    continue
            return str(best.date()) if best is not None else None

        return self._memo(f"events|{kind.upper()}", f)

    def option_first(self, sym: str) -> Optional[str]:
        def f():
            st = self.portal.store
            if not st.exists("option_chain", "dolthub", sym):
                return None
            for yr in range(2019, pd.Timestamp.now().year + 1):
                try:
                    d = st.read("option_chain", "dolthub", sym, start=pd.Timestamp(f"{yr}-01-01", tz="UTC"),
                                end=pd.Timestamp(f"{yr + 1}-01-01", tz="UTC"), columns=["ts"], time_col="ts")
                except Exception:
                    continue
                if len(d):
                    ts = d["ts"] if "ts" in d.columns else pd.Series(d.index)
                    return str(pd.Timestamp(ts.min()).tz_convert("America/New_York").date())
            return None

        return self._memo(f"options|{sym}", f)

    def sp500_full_coverage_start(self) -> Optional[str]:
        """First month start (1996+) from which PIT S&P 500 price coverage >= ``min_coverage`` (1.0 = every
        member, the engine's strict rule; < 1 with the runner's CoverageFilteredPortal). Binary search on
        month starts; coverage is monotone in practice -- DATA.md §8 table."""
        th = self.min_coverage

        def cov(d):
            a = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21)
            try:                                   # f39aa6b+: exactly the engine's / has(window) definition
                allm = self.portal.universe("SP500", a)
                kept = self.portal.universe("SP500", a, require_bars=True)
                return len(kept) / len(allm) if allm else 0.0
            except TypeError:
                r = self.portal.universe_survivorship_bias("SP500", a)
                return float(r.get("coverage", 0.0))

        def f():
            # yearly scan for the first full-coverage January, then month-level binary search in the
            # preceding year (the latest months can miss a renamed member, e.g. FI in 2026-10, which
            # does not matter: the engine checks members at the FIRST session only)
            years = list(range(1997, _today_utc().year + 1))
            first_full = next((y for y in years if cov(pd.Timestamp(f"{y}-01-01")) >= th), None)
            if first_full is None:
                return None
            months = pd.date_range(f"{first_full - 1}-01-01", f"{first_full}-01-01", freq="MS")
            lo, hi = 0, len(months) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cov(months[mid]) >= th:
                    hi = mid
                else:
                    lo = mid + 1
            # guard against non-monotone patches: walk forward until two consecutive full months
            i = lo
            while i + 1 < len(months) and cov(months[i + 1]) < th:
                i += 1
            return str(self.cal.sessions(months[i], months[i] + pd.Timedelta(days=10))[0].date())

        return self._memo(f"sp500|coverage_start|{th:.3f}", f)

    #: long-lived issuers used to probe when a fundamentals field becomes available through the portal
    FIELD_PROBE = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "IBM", "GE")

    def field_start(self, field: str) -> Optional[str]:
        """First month end at which portal.fundamentals(field) already had a value ONE month earlier
        (a level AND its change/revision are computable): estimates 2017-10-31 -> 2017-11-30; XBRL
        revenue_q 2009-08 -> 2009-09-30. None if the portal never serves the field for the probe set."""
        def known(d) -> bool:
            a = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21)
            try:
                r = self.portal.fundamentals(list(self.FIELD_PROBE), [field], a)
            except Exception:
                return False
            return bool(len(r)) and bool(pd.to_numeric(r.get("value"), errors="coerce").notna().any())

        def f():
            now = _today_utc().tz_localize(None)
            years = list(range(2000, now.year + 1))
            y0 = next((y for y in years if known(pd.Timestamp(f"{y}-12-31"))), None)
            if y0 is None:
                return None
            mes = pd.date_range(f"{y0}-01-31", f"{y0}-12-31", freq="ME")
            first = next(m for m in mes if known(m))
            return str((first + pd.offsets.MonthEnd(1)).date())

        return self._memo(f"fundamentals|start|{field}", f)

    def sp500_coverage_note(self) -> str:
        def f():
            notes = []
            for d in ("2012-01-03", "2020-01-02", "2024-01-02"):
                r = self.portal.universe_survivorship_bias("SP500", pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21))
                notes.append(f"{d}: {r.get('with_data')}/{r.get('members')} missing {','.join(list(r.get('missing', []))[:4])}")
            return "; ".join(notes)

        return self._memo("sp500|coverage_note", f)

    def universe_at(self, name: str, date) -> list:
        def f():
            return list(self.portal.universe(name, pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=14)))

        return self._memo(f"universe|{name}|{pd.Timestamp(date).date()}", f)

    def sessions_between(self, a, b) -> int:
        return len(self.cal.sessions(pd.Timestamp(a), pd.Timestamp(b)))

    def add_sessions(self, d, n: int) -> pd.Timestamp:
        s = self.cal.sessions(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=int(n * 1.6) + 30))
        return s[min(n, len(s) - 1)]


def _fid_name(f) -> str:
    try:
        return Fidelity(int(f)).name
    except Exception:
        return str(f)


def plan_strategy(sid: str, pc: PlanContext) -> dict:
    """Plan one sheet strategy or one proxy variant (``<SID>@<tag>``, run/variants.py). A variant is
    instantiated with its proxy params, so its requirements, bar frequency and window are the proxy's;
    its row carries ``variant_of`` and an expected fidelity of at least T2_PROXY."""
    from atlasbt.strategies import registry as R

    from . import variants as V

    row = {"id": sid, "variant_of": "", "family": "", "module": "", "bar_freq": "", "sizing": "", "testable": "N",
           "reason": "", "start": "", "end": "", "n_sessions": 0, "expected_fidelity": "", "fidelity_notes": "",
           "n_grid": 0, "universe": "", "n_symbols": 0, "binding_constraint": "",
           "est_runtime_min_baseline": np.nan, "est_runtime_min_grid": np.nan}
    var = V.get(sid) if V.is_variant(sid) else None
    if V.is_variant(sid) and var is None:
        row["reason"] = f"unknown proxy variant (registered: {', '.join(V.variant_ids())})"
        return row
    if var is not None:
        row["variant_of"] = var.base
    try:
        cls = V.strategy_class(sid)
    except KeyError:
        errs = R.load_all()
        row["reason"] = f"strategy module not loaded: {errs or 'not registered'}"
        return row
    row["module"] = cls.__module__.split(".")[-1]
    row["family"] = _family(V.base_of(sid))
    try:
        inst = V.instantiate(sid) if var is not None else cls()
    except Exception as e:
        row["reason"] = f"strategy instantiation failed: {type(e).__name__}: {e}"
        return row
    freq = str(inst.bar_freq)
    row.update(bar_freq=freq, sizing=str(inst.sizing or ""), n_grid=len(V.expand_grid(sid)))
    if var is not None:
        row["fidelity_notes"] = f"PROXY VARIANT of {var.base} {var.params}: {var.reason}"
    out = _plan_instance(sid, inst, freq, pc, row)
    if var is not None and out.get("testable") == "Y":
        fid = out.get("expected_fidelity") or "T0_QUOTE"
        try:
            if Fidelity[fid] < Fidelity.T2_PROXY:
                out["expected_fidelity"] = Fidelity.T2_PROXY.name
        except KeyError:
            pass
    return out


def _plan_instance(sid: str, inst, freq: str, pc: PlanContext, row: dict) -> dict:
    var_note = row.get("fidelity_notes", "")
    intraday = is_intraday(freq)
    universe = [s.upper() for s in (getattr(inst, "universe", None) or [])]
    uname = getattr(inst, "universe_name", None)
    row["universe"] = uname or ",".join(universe[:8]) + ("..." if len(universe) > 8 else "")
    reqs = list(getattr(inst, "requires", None) or [])
    end_cap = pd.Timestamp(_today_utc().tz_localize(None).normalize())

    from atlasbt.engine.strategy import PROFILE_SESSION_RULES

    rules = PROFILE_SESSION_RULES.get(inst.sizing or "", {})
    if not intraday and (rules.get("flat_by") or getattr(inst, "flat_by", None)):
        row["reason"] = (f"{inst.sizing} session rules (flat by {rules.get('flat_by')}) need an intraday base "
                         f"frequency; daily bars cannot exit before the close (engine rule)")
        return row

    p = pc.portal
    starts: list[tuple[pd.Timestamp, str, bool]] = []     # (date, why, is_listing)
    ends: list[pd.Timestamp] = []
    worst = Fidelity.T0_QUOTE
    notes: list[str] = []
    members_resolved = None

    def resolve_members():
        nonlocal members_resolved
        if members_resolved is not None:
            return members_resolved
        if uname and uname.upper() == "SP500":
            s0 = pc.sp500_full_coverage_start()
            if s0 is None:
                cov = pc.sp500_coverage_note()
                members_resolved = ([], "S&P 500 PIT universe never fully covered by price data (" + cov + "); the "
                                        "engine requires portal.has() for EVERY member at the first session")
            else:
                members_resolved = (pc.universe_at("SP500", s0), f"SP500 PIT universe fully covered from {s0}")
                starts.append((pd.Timestamp(s0), f"SP500 PIT price coverage >= {pc.min_coverage:.0%}", False))
        elif uname:
            last = pc.bar_range("SPY", "1d")
            d = last[1] if last else str(end_cap.date())
            members_resolved = (pc.universe_at(uname, d), f"{uname} resolved at {d}")
        else:
            members_resolved = (universe, "static universe")
        return members_resolved

    if uname and uname.upper().startswith("SP500") and uname.upper() == "SP500":
        syms0, why0 = resolve_members()            # PIT coverage start applies whatever the requirements say
        if not syms0:
            row["reason"] = why0
            return row
    # PIT universe under an intraday strategy (SP500_TOP50_DV, S&P 500 5m): the engine verifies the members
    # of EVERY session it queries the universe (base-frequency bars in the run window + every critical
    # requirement without symbols) against min_universe_coverage, and Alpaca bars exist only around each
    # stock's own member sessions -> the window comes from per-session coverage (PlanContext.pit_window),
    # computed once for all bars requirements without explicit symbols
    pit_intraday = intraday and _is_pit_universe(uname)
    pit_done = False
    if pit_intraday:
        nosym = [r for r in reqs if r.critical and not r.symbols]       # the engine's member checks

        def ext(r) -> bool:                    # PRE/POST data the engine probes over the first sessions
            return bool(set(getattr(r, "sessions", ()) or ()) & {"PRE", "POST"})

        pit_freqs = tuple(dict.fromkeys([_norm_freq(freq, freq)] + [_norm_freq(r.freq, freq) for r in nosym
                                                                     if r.kind == "bars"]))
        pit_kinds = tuple(dict.fromkeys((r.kind, r.freq) for r in nosym if r.kind != "bars"))
        pit_probe = tuple(dict.fromkeys(_norm_freq(r.freq, freq) for r in nosym if r.kind == "bars" and ext(r)))
        pit_probe_q = any(r.kind == "quotes" and ext(r) for r in nosym)
    for r in reqs:
        if not r.critical:
            # non-critical inputs never block a run, but when available they TIER it exactly like the
            # engine's fidelity ledger does (S042 DIVIDEND_DECL events are T2 -> the run is T2)
            fid_nc, what = _noncritical_fidelity(p, r)
            if fid_nc is None:
                notes.append(f"non-critical {what} unavailable (run proceeds without it)")
            else:
                worst = max(worst, fid_nc)
                if fid_nc > Fidelity.T1_BAR:
                    notes.append(f"non-critical {what} used at {fid_nc.name}")
            continue
        kind = r.kind
        rfreq = r.freq
        explicit = r.symbols is not None and len(r.symbols) > 0
        star = explicit and all(str(s) == "*" for s in r.symbols)
        if kind == "bars" and star:
            # bars of the WHOLE universe ("*"): verified exactly like a bars requirement without
            # symbols -- per-member availability, the PIT coverage rule for SP500 (min_coverage) and the
            # 80 %-listed rule for larger static universes. Without any universe it is a portal-wide
            # check (portal.has('*', 'bars', freq), the engine's semantics for "*").
            explicit = False
            if not uname and not universe:
                f = rfreq or freq
                f = "1d" if f in ("1mo", "1w", "event") else f
                try:
                    ok = bool(p.has("*", "bars", f))
                except Exception:
                    ok = False
                if not ok:
                    row["reason"] = f"required bars {f} for the whole universe ('*') unavailable (no universe, portal-wide check failed)"
                    return row
                continue
        if explicit:
            syms = [s.upper() for s in r.symbols]
        else:
            syms, why = resolve_members()
            if not syms:
                row["reason"] = f"{why}"
                return row
        if kind == "events":
            k = (rfreq or "").upper()
            if not p.has("*", "events", k):
                row["reason"] = f"required events {k} unavailable (no source integrated)"
                return row
            f0 = pc.event_first(k)
            if f0:
                starts.append((pd.Timestamp(f0), f"events {k}", False))
            worst = max(worst, Fidelity(int(p.fidelity_of("*", "events", k))))
            continue
        if kind == "fundamentals":
            if not p.has("*", "fundamentals", rfreq):
                row["reason"] = f"required fundamentals {rfreq or ''} unavailable"
                return row
            # per-field availability start and tier come from the portal (XBRL fields 2009 T1; analyst
            # consensus eps_consensus_ntm 2017-11 T2), never from constants
            f0 = pc.field_start(rfreq) if rfreq else None
            starts.append((pd.Timestamp(f0) if f0 else FUNDAMENTALS_START,
                           f"fundamentals {rfreq or ''} first usable month end".strip() if f0 else
                           "fundamentals (XBRL era)", False))
            try:
                ff = Fidelity(int(p.fidelity_of("*", "fundamentals", rfreq)))
            except Exception:
                ff = Fidelity.T1_BAR
            if ff == Fidelity.NA or ff > r.min_fidelity:
                row["reason"] = f"required fundamentals {rfreq or ''} unavailable (fidelity {ff.name})"
                return row
            worst = max(worst, ff)
            continue
        if kind == "options":
            # availability and tier from the portal (OPTION_SOURCE_FIDELITY: Databento/Polygon/Alpaca T0,
            # cboe_eod T1, DoltHub/hod T2, synthetic fallback T3 (EOD only), intraday without a source NA)
            for s in syms:
                try:
                    has = bool(p.has(s, "options", rfreq))
                    fid = Fidelity(int(p.fidelity_of(s, "options", rfreq)))
                except Exception:
                    has, fid = False, Fidelity.NA
                if fid != Fidelity.NA and fid <= r.min_fidelity and not has and s != "*" and fid == Fidelity.T3_MODELED:
                    has = True              # synthetic fallback: no stored chain, but the portal serves T3 quotes
                if not has or fid == Fidelity.NA or fid > r.min_fidelity:
                    row["reason"] = (f"required options {rfreq or 'EOD'} for {s} unavailable (fidelity {fid.name}"
                                     f"{', min ' + r.min_fidelity.name if fid != Fidelity.NA else ''})")
                    return row
                worst = max(worst, fid)
                if s != "*" and fid <= Fidelity.T2_PROXY and r.min_fidelity <= Fidelity.T2_PROXY:
                    f0 = pc.option_first(s)
                    if f0:
                        starts.append((pd.Timestamp(f0), f"real option chains {s}", False))
            continue
        if kind == "quotes":
            need = MIN_SESSIONS_INTRADAY if intraday else MIN_SESSIONS_DAILY
            if not explicit and _is_pit_universe(uname):
                # PIT universe: a stock needs quotes only on the sessions it is a member (Alpaca pre-open
                # windows are recorded exactly for the day's top-N) -> UNION of the members' quote sessions
                pool = pc.pit_members(uname) or syms
                days_all, used = set(), []
                for s in pool:
                    q = pc.quote_range(s)
                    if q:
                        days_all |= set(q)
                        used.append(s)
                if not days_all:
                    row["reason"] = f"required quotes for the {uname} members unavailable"
                    return row
                what = f"quote sessions (union of {len(used)} {uname} members)"
            else:
                days_all, used = None, list(syms)
                for s in syms:
                    q = pc.quote_range(s)
                    if not q:
                        row["reason"] = f"required quotes for {s} unavailable"
                        return row
                    days_all = set(q) if days_all is None else days_all & set(q)
                days_all = days_all or set()
                what = "quote sessions common to " + ",".join(used[:4]) + ("..." if len(used) > 4 else "")
            n = len(days_all)
            if n < need:
                row["reason"] = (f"required quotes exist on only {n} common session(s) "
                                 f"({', '.join(sorted(days_all)[:4])}); minimum window {need} sessions")
                return row
            days = sorted(days_all)
            starts.append((pd.Timestamp(days[0]), f"first of {n} {what}", False))
            ends.append(pd.Timestamp(days[-1]))
            for s in used[:50]:
                try:
                    fq = Fidelity(int(p.fidelity_of(s, "quotes", rfreq)))
                except Exception:
                    continue
                if fq != Fidelity.NA:
                    worst = max(worst, fq)
            continue
        if kind == "bars" and pit_intraday and not explicit:
            f = _norm_freq(rfreq, freq)
            if not pit_done:
                pit_done = True
                w = pc.pit_window(uname, pit_freqs, pit_kinds, pit_probe, pit_probe_q)
                if "reason" in w:
                    row["reason"] = w["reason"]
                    return row
                starts.append((pd.Timestamp(w["start"]), w["why"], False))
                ends.append(pd.Timestamp(w["end"]))
                if w.get("engine_cov_min", 1.0) < 1.0:
                    # the engine flags a run whose universe coverage drops below 100 % as T2_PROXY
                    worst = max(worst, Fidelity.T2_PROXY)
                    notes.append(f"{uname} coverage {w['engine_cov_min']:.1%} at worst (members without "
                                 f"{'+'.join(pit_freqs)} bars in the window are dropped and the engine flags "
                                 f"the run T2_PROXY)")
            for s in syms:      # tier from the members' sources (the engine drops a worse-than-minimum member)
                try:
                    fid = Fidelity(int(p.fidelity_of(s, "bars", f))) if p.has(s, "bars", f) else Fidelity.NA
                except Exception:
                    fid = Fidelity.NA
                if fid != Fidelity.NA and fid <= r.min_fidelity:
                    worst = max(worst, fid)
            continue
        if kind == "bars":
            f = rfreq or freq
            f = "1d" if f in ("1mo", "1w", "event") else f
            rng = {}
            # the runtime filter only drops members WITHOUT DAILY data (survivorship gaps); intraday gaps
            # (e.g. top-50 names without Yahoo 1m) stay NOT_TESTABLE: dropping them would change the universe
            pit_filter = f == "1d" and bool(uname) and not explicit and pc.min_coverage < 1.0 and (
                uname.upper().startswith("SP500") or uname.upper() in ("MEGACAP",))
            dropped = []
            for s in syms:
                if s == "*":
                    continue
                if pit_filter and pc.bar_range(s, f) is None:
                    dropped.append(s)          # CoverageFilteredPortal drops it at run time too
                    continue
                if not p.has(s, "bars", f):
                    row["reason"] = f"required bars {f} for {s} unavailable"
                    return row
                fid = Fidelity(int(p.fidelity_of(s, "bars", f)))
                if fid == Fidelity.NA or fid > r.min_fidelity:
                    row["reason"] = f"required bars {f} for {s} unavailable (fidelity {fid.name})"
                    return row
                worst = max(worst, fid)
                br = pc.bar_range(s, f)
                if br is None:
                    row["reason"] = f"required bars {f} for {s}: no rows returned"
                    return row
                rng[s] = (pd.Timestamp(br[0]), pd.Timestamp(br[1]))
            if dropped:
                from .common import display_symbol

                notes.append(f"{len(dropped)} of {len(syms)} {uname} members at the start have no {f} data and are "
                             f"dropped by the coverage filter ({', '.join(display_symbol(x) for x in dropped[:6])}"
                             f"{'...' if len(dropped) > 6 else ''})")
            if not rng:
                continue
            firsts = sorted(v[0] for v in rng.values())
            if uname and uname.upper() == "SP500" and not explicit:
                ends.append(pd.Timestamp(np.median([v[1].value for v in rng.values()])).normalize())
                continue
            if explicit or len(rng) <= SMALL_UNIVERSE:
                d0 = firsts[-1]
                who = max(rng, key=lambda k: rng[k][0])
                starts.append((d0, f"first {f} bar of {who}", True))
                ends.append(min(v[1] for v in rng.values()))
            else:
                k = int(math.ceil(UNIVERSE_SHARE * len(firsts))) - 1
                starts.append((firsts[k], f"{int(UNIVERSE_SHARE * 100)}% of {len(firsts)} universe members listed "
                                          f"({f})", True))
                ends.append(sorted(v[1] for v in rng.values())[len(firsts) // 2])
            continue
        # unknown kind (e.g. 'inav', 'borrow')
        try:
            ok = bool(p.has(syms[0] if syms else "*", kind, rfreq))
        except Exception:
            ok = False
        if not ok:
            row["reason"] = f"required {kind} {rfreq or ''} unavailable (no source in the cache)"
            return row

    if str(getattr(inst, "asset", "")).upper().startswith("OPT"):
        for s in universe:
            if s in ("SPX", "SPXW", "XSP", "NDX", "RUT", "DJX") or not p.has(s, "options"):
                worst = max(worst, Fidelity.T3_MODELED)
                notes.append(f"{s} options: synthetic chain only (T3_MODELED)")
            else:
                notes.append(f"{s} options: real DoltHub EOD chains from {pc.option_first(s)} (T2), synthetic T3 "
                             f"before")
    if not starts:
        # no critical bars requirement (unusual): fall back to the universe's daily bars
        syms, why_m = resolve_members()
        if not syms:
            row["reason"] = why_m
            return row
        for s in syms[:20]:
            br = pc.bar_range(s, "1d")
            if br:
                starts.append((pd.Timestamp(br[0]), f"first 1d bar of {s}", True))
                ends.append(pd.Timestamp(br[1]))
    if not starts:
        row["reason"] = "no data window could be determined"
        return row
    d0, why, listing = max(starts, key=lambda t: t[0])
    if listing and not intraday:
        d0 = pc.add_sessions(d0, WARMUP_SESSIONS)
        why += f" + {WARMUP_SESSIONS}-session warm-up"
    d1 = min(ends) if ends else pd.Timestamp(pc.bar_range("SPY", "1d")[1])
    sess = pc.cal.sessions(d0, d1)
    if len(sess):
        d0, d1 = sess[0], sess[-1]
    n = len(sess)
    need = MIN_SESSIONS_INTRADAY if intraday else MIN_SESSIONS_DAILY
    syms_n = len(resolve_members()[0]) if (uname or universe) else 1
    row.update(start=str(pd.Timestamp(d0).date()), end=str(pd.Timestamp(d1).date()), n_sessions=n,
               binding_constraint=why, n_symbols=syms_n,
               expected_fidelity=_fid_name(worst),
               fidelity_notes="; ".join(dict.fromkeys(filter(None, [var_note] + notes))))
    if uname and uname.upper().startswith("SP500"):
        row["fidelity_notes"] = "; ".join(filter(None, [row["fidelity_notes"],
                                                        "PIT S&P 500 universe; sectors not PIT (DATA.md §8)"]))
    if n < need:
        row["reason"] = f"window {row['start']}..{row['end']} has {n} sessions < minimum {need}"
        return row
    # universes the strategy asks the portal for at RUN time (S042: ETF_CORE + 'SP500_TOP50_DV'): the
    # engine filters each one through its coverage rule (members with base-frequency bars in the run
    # window, BacktestConfig.min_universe_coverage) and refuses the run below the threshold
    for un in _runtime_universes(inst, uname):
        bad = _runtime_coverage(pc, un, freq, pd.Timestamp(d0), pd.Timestamp(d1))
        if bad:
            row["reason"] = bad
            return row
    row["testable"] = "Y"
    est = estimate_runtime_sec(freq, n, syms_n, is_option=str(inst.asset).upper().startswith("OPT"))
    row["est_runtime_min_baseline"] = round(est / 60, 1)
    row["est_runtime_min_grid"] = round(est * row["n_grid"] / 60, 1)
    return row


def _noncritical_fidelity(p, r) -> tuple[Optional[Fidelity], str]:
    """(fidelity, label) of a non-critical requirement when the portal has it, else (None, label)."""
    kind = r.kind
    if kind == "events":
        k = (r.freq or "").upper()
        what, sym, fq = f"events {k}", "*", k
    else:
        syms = [s for s in (r.symbols or ()) if s] or ["*"]
        sym, fq = syms[0], r.freq
        what = f"{kind} {r.freq or ''}".strip() + ("" if sym == "*" else f" for {sym}")
    try:
        if not bool(p.has(sym, kind, fq)):
            return None, what
        f = Fidelity(int(p.fidelity_of(sym, kind, fq)))
    except Exception:
        return None, what
    return (None if f == Fidelity.NA else f), what


_RT_UNIVERSE_PATTERNS = (r"""_universe\(\s*ctx\s*,\s*["']([A-Z0-9_]+)["']\s*\)""",
                         r"""\.universe\(\s*["']([A-Z0-9_]+)["']""")


def _runtime_universes(inst, uname: Optional[str]) -> list[str]:
    """Portal universes a strategy queries at run time besides its declared ``universe_name``: an explicit
    ``runtime_universes`` attribute, else the literal names in its class source (``_universe(ctx, "X")``,
    ``portal().universe("X")``) -- every such call goes through the engine's coverage filter."""
    names = getattr(inst, "runtime_universes", None)
    if names is None:
        import inspect
        import re

        found: list[str] = []
        for cls in type(inst).__mro__:
            if not cls.__module__.startswith("atlasbt.strategies") or cls.__name__.startswith("_"):
                continue                       # shared bases (e.g. _TimedBase) are generic helpers
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            for pat in _RT_UNIVERSE_PATTERNS:
                found += re.findall(pat, src)
        names = found
    out = []
    for n in names:
        n = str(n).upper()
        if n and n != (uname or "").upper() and n not in out:
            out.append(n)
    return out


def _runtime_coverage(pc: "PlanContext", name: str, freq: str, d0: pd.Timestamp, d1: pd.Timestamp) -> str:
    """'' when a run-time universe meets the engine's coverage rule at the window's first and last
    session (members with ``freq`` bars inside [d0, d1] / members >= min_coverage), else the reason."""
    f = "1d" if freq in ("1mo", "1w", "event") else freq
    th = pc.min_coverage
    for d in (d0, d1):
        try:
            members = pc.universe_at(name, d)
        except Exception as e:
            return f"run-time universe {name} unavailable ({type(e).__name__}: {e})"
        if not members:
            continue
        a, b = str(pd.Timestamp(d0).date()), str(pd.Timestamp(d1).date())
        # the engine's windowed has(): a bar anywhere in [d0, d1] (intraday spans: gaps between a stock's
        # fetched periods count as missing)
        missing = [s_ for s_ in members if not _in_spans(pc.bar_spans(s_, f), a, b)]
        cov = 1 - len(missing) / len(members)
        if cov < th:
            from .common import display_symbol

            return (f"run-time universe {name}: {f} bar coverage {cov:.0%} on {pd.Timestamp(d).date()} "
                    f"({len(missing)}/{len(members)} members without {f} bars in {d0.date()}..{d1.date()}, e.g. "
                    f"{', '.join(display_symbol(x) for x in missing[:4])}) < min_universe_coverage {th:.0%} "
                    f"(the engine refuses the run)")
    return ""


def estimate_runtime_sec(freq: str, n_sessions: int, n_symbols: int, is_option: bool = False) -> float:
    f = freq if freq in BASE_SEC else ("1d" if not is_intraday(freq) else "5m")
    per = BASE_SEC[f] + PER_SYMBOL_SEC[f] * max(n_symbols, 1)
    if is_option:
        per += OPTION_SESSION_SEC
    return SETUP_SEC + n_sessions * per


_FAMILY = None


def _family(sid: str) -> str:
    global _FAMILY
    if _FAMILY is None:
        import json

        from .common import REPO

        try:
            _FAMILY = {d["ID"]: d.get("Family", "") for d in json.loads((REPO / "docs" / "strategies.json").read_text())}
        except Exception:
            _FAMILY = {}
    return _FAMILY.get(sid, "")


def all_sheet_ids() -> list[str]:
    import json

    from .common import REPO

    return [d["ID"] for d in json.loads((REPO / "docs" / "strategies.json").read_text())]


def make_plan(ids=None, results_dir: Path = DEFAULT_RESULTS, portal=None, log=print,
              min_coverage: float = 0.95, variants: bool = True) -> pd.DataFrame:
    """min_coverage: PIT stock-universe price coverage required at the window start (1.0 = engine-strict;
    the default 0.95 matches the runner's --universe-coverage filter mode). ``ids=None`` plans all 172
    sheet ids plus (``variants=True``) every registered proxy variant; explicit ids may name variants."""
    from . import variants as V
    from .fastportal import FastPortal

    portal = portal or FastPortal()
    pc = PlanContext(portal, results_dir, min_coverage)
    ids = ids or (all_sheet_ids() + (V.variant_ids() if variants else []))
    rows = []
    for sid in ids:
        try:
            r = plan_strategy(sid, pc)
        except Exception as e:
            r = {"id": sid, "testable": "N", "reason": f"plan error: {type(e).__name__}: {e}",
                 "plan_traceback": traceback.format_exc()[-2000:]}
        rows.append(r)
        log(f"plan {sid}: {r.get('testable')} {r.get('start', '')}..{r.get('end', '')} {r.get('reason', '')[:120]}")
        pc.save()
    df = pd.DataFrame(rows)
    df["testable_reason"] = [("Y" if t == "Y" else f"N: {why}") for t, why in zip(df["testable"], df["reason"])]
    pc.save()
    return df


def write_plan(df: pd.DataFrame, results_dir: Path = DEFAULT_RESULTS) -> Path:
    out = Path(results_dir) / "plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "variant_of", "family", "module", "bar_freq", "sizing", "testable", "reason", "start", "end", "n_sessions",
            "expected_fidelity", "fidelity_notes", "n_grid", "est_runtime_min_baseline", "est_runtime_min_grid",
            "universe", "n_symbols", "binding_constraint"]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    # merge into an existing plan (plan for a subset must not drop other rows)
    if out.exists():
        old = pd.read_csv(out, dtype=str)
        old = old[~old["id"].isin(df["id"])]
        df = pd.concat([old, df[cols].astype(str)], ignore_index=True).sort_values("id")
    df[cols].to_csv(out, index=False)
    return out


def load_plan(results_dir: Path = DEFAULT_RESULTS) -> pd.DataFrame:
    p = Path(results_dir) / "plan.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str).fillna("")


def annotate_plan(results_dir: Path = DEFAULT_RESULTS, allow_stock: bool = False) -> pd.DataFrame:
    """Add/refresh the `hold` (integrator data holds) and `activity` (from baseline runs) columns."""
    from .holds import held, is_stale

    plan = load_plan(results_dir)
    if not len(plan):
        return plan
    holds, acts, stale = [], [], []
    for sid in plan["id"]:
        try:
            holds.append(held(sid, allow_stock))
        except Exception:
            holds.append("")
        st = read_json(Path(results_dir) / "runs" / sid / "baseline" / "status.json", {}) or {}
        a = st.get("activity") or ""
        if st.get("status") and st.get("status") != "OK":
            a = st["status"]
        acts.append(a)
        stale.append("STALE" if is_stale(results_dir, sid) else "")
    plan["hold"] = holds
    plan["activity"] = acts
    plan["stale"] = stale
    plan.to_csv(Path(results_dir) / "plan.csv", index=False)
    return plan

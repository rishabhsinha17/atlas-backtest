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
  - bars of explicitly named symbols: the latest first bar among them;
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
* Window end = the last cached session common to the critical bar inputs.
* Minimum window: 252 sessions for daily+ strategies, 10 sessions for intraday ones.
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

#: rough cost model (seconds) per config: SETUP + sessions * (BASE[freq] + PER_SYMBOL[freq] * n_symbols)
SETUP_SEC = 15.0
BASE_SEC = {"1d": 0.035, "1mo": 0.03, "1w": 0.03, "5m": 0.25, "1m": 0.8}
PER_SYMBOL_SEC = {"1d": 0.006, "1mo": 0.004, "1w": 0.004, "5m": 0.04, "1m": 0.15}
OPTION_SESSION_SEC = 0.12


def _today_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class PlanContext:
    """Memoized availability facts (persisted in results/plan_cache.json)."""

    def __init__(self, portal, results_dir: Path = DEFAULT_RESULTS, min_coverage: float = 1.0):
        self.portal = portal
        self.min_coverage = float(min_coverage)
        self.path = Path(results_dir) / "plan_cache.json"
        self.cache: dict = read_json(self.path, {}) or {}
        self.dirty = False
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
            start = pd.Timestamp("1960-01-01", tz="UTC") if freq == "1d" else now - pd.Timedelta(days=800)
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

    def quote_range(self, sym: str) -> Optional[list]:
        def f():
            store = self.portal.store
            days = set()
            for src in ("alpaca", "databento", "iex", "dukascopy"):
                try:
                    if store.exists("quotes", src, sym):
                        q = store.read("quotes", src, sym)
                        idx = pd.DatetimeIndex(q.index if not isinstance(q.index, pd.RangeIndex) else q["ts"])
                        if idx.tz is None:
                            idx = idx.tz_localize("UTC")
                        days |= {str(d.date()) for d in idx.tz_convert("America/New_York").normalize().unique()}
                except Exception:
                    continue
            return sorted(days) if days else None

        return self._memo(f"quotes|{sym}", f)

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
    from atlasbt.strategies import registry as R

    row = {"id": sid, "family": "", "module": "", "bar_freq": "", "sizing": "", "testable": "N", "reason": "",
           "start": "", "end": "", "n_sessions": 0, "expected_fidelity": "", "fidelity_notes": "",
           "n_grid": 0, "universe": "", "n_symbols": 0, "binding_constraint": "",
           "est_runtime_min_baseline": np.nan, "est_runtime_min_grid": np.nan}
    try:
        cls = R.get(sid)
    except KeyError:
        errs = R.load_all()
        row["reason"] = f"strategy module not loaded: {errs or 'not registered'}"
        return row
    row["module"] = cls.__module__.split(".")[-1]
    row["family"] = _family(sid)
    try:
        inst = cls()
    except Exception as e:
        row["reason"] = f"strategy instantiation failed: {type(e).__name__}: {e}"
        return row
    freq = str(inst.bar_freq)
    row.update(bar_freq=freq, sizing=str(inst.sizing or ""), n_grid=len(R.expand_grid(cls.grid)))
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
    for r in reqs:
        if not r.critical:
            continue
        kind = r.kind
        rfreq = r.freq
        explicit = r.symbols is not None and len(r.symbols) > 0
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
            starts.append((FUNDAMENTALS_START, "fundamentals (XBRL era)", False))
            worst = max(worst, Fidelity.T1_BAR)
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
            days_all = None
            for s in syms:
                q = pc.quote_range(s)
                if not q:
                    row["reason"] = f"required quotes for {s} unavailable"
                    return row
                days_all = set(q) if days_all is None else days_all & set(q)
            n = len(days_all or [])
            need = MIN_SESSIONS_INTRADAY if intraday else MIN_SESSIONS_DAILY
            row["reason"] = (f"required quotes exist on only {n} common session(s) "
                             f"({', '.join(sorted(days_all or [])[:4])}); minimum window {need} sessions")
            return row
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
               expected_fidelity=_fid_name(worst), fidelity_notes="; ".join(dict.fromkeys(notes)))
    if uname and uname.upper().startswith("SP500"):
        row["fidelity_notes"] = "; ".join(filter(None, [row["fidelity_notes"],
                                                        "PIT S&P 500 universe; sectors not PIT (DATA.md §8)"]))
    if n < need:
        row["reason"] = f"window {row['start']}..{row['end']} has {n} sessions < minimum {need}"
        return row
    row["testable"] = "Y"
    est = estimate_runtime_sec(freq, n, syms_n, is_option=str(inst.asset).upper().startswith("OPT"))
    row["est_runtime_min_baseline"] = round(est / 60, 1)
    row["est_runtime_min_grid"] = round(est * row["n_grid"] / 60, 1)
    return row


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
              min_coverage: float = 0.95) -> pd.DataFrame:
    """min_coverage: PIT stock-universe price coverage required at the window start (1.0 = engine-strict;
    the default 0.95 matches the runner's --universe-coverage filter mode)."""
    from .fastportal import FastPortal

    portal = portal or FastPortal()
    pc = PlanContext(portal, results_dir, min_coverage)
    ids = ids or all_sheet_ids()
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
    cols = ["id", "family", "module", "bar_freq", "sizing", "testable", "reason", "start", "end", "n_sessions",
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

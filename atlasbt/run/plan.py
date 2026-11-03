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
        missing = []
        for s_ in members:
            br = pc.bar_range(s_, f)
            if br is None or pd.Timestamp(br[1]) < d0 or pd.Timestamp(br[0]) > d1:
                missing.append(s_)
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

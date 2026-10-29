"""Memoizing Portal used by the atlas runner (performance only, same semantics).

``atlasbt.data.portal.Portal.corporate_actions`` rebuilds CorporateAction objects
from a DataFrame on every call (~100 ms for 10 symbols). The engine calls it at
every session start and several strategies call it every bar, which made a
21-year daily ETF backtest take ~40 minutes. ``FastPortal`` asks the parent
ONCE per symbol for every action (as_of = far future, full date range) and then
applies exactly the parent's filters (``start <= ex_date <= end`` on naive ET
dates, ``known_at <= as_of``) on numpy arrays. Equivalence with the parent is
tested in ``tests/run/test_fastportal.py``. Nothing else is overridden, so the
PIT contract (``known_at <= as_of``) is still enforced by the parent filters and
re-checked by the engine's PITPortal.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.data.portal import Portal, _naive_date, _session_date, _utc

_FAR_PAST = pd.Timestamp("1900-01-01")
_FAR_FUTURE = pd.Timestamp("2200-01-01", tz="UTC")


class FastPortal(Portal):
    name = "Portal"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ca_memo: dict[str, tuple[list, np.ndarray, np.ndarray]] = {}

    def _all_actions(self, sym: str, rdate=None):
        """All actions of the security ``sym`` denotes on ``rdate`` (identity-aware Portal: a reused
        ticker denotes different securities on different dates, so the memo is keyed by the identity
        frame the parent resolves for that date)."""
        ident = None
        if hasattr(self, "_identity_frame"):
            ident = self._identity_frame(sym, rdate)
            key = (sym.upper(), ident.attrs.get("security_id"), id(ident))
        else:  # pragma: no cover - pre-identity Portal
            key = (sym.upper(),)
        hit = self._ca_memo.get(key)
        if hit is None:
            kw = {"_ident": ident} if ident is not None else {}
            acts = Portal.corporate_actions(self, [sym], _FAR_PAST, pd.Timestamp("2199-12-31"), _FAR_FUTURE, **kw)
            ex = np.array([_naive_date(a.ex_date).value for a in acts], dtype=np.int64)
            kn = np.array([_utc(a.known_at).value for a in acts], dtype=np.int64)
            hit = (acts, ex, kn)
            self._ca_memo[key] = hit
        return hit

    def corporate_actions(self, symbols: Optional[Iterable[str]], start, end, as_of, _ident=None):
        if symbols is None or _ident is not None:
            kw = {"_ident": _ident} if _ident is not None else {}
            return Portal.corporate_actions(self, symbols, start, end, as_of, **kw)
        s0d, e0d = _naive_date(start), _naive_date(end)
        s0, e0 = s0d.value, e0d.value
        a_ns = _utc(as_of).value
        rdate = min(e0d, _session_date(_utc(as_of)))      # same identity date as the parent
        out = []
        for sym in symbols:
            acts, ex, kn = self._all_actions(sym, rdate)
            if not len(acts):
                continue
            m = (ex >= s0) & (ex <= e0) & (kn <= a_ns)
            if m.any():
                out.extend(acts[i] for i in np.flatnonzero(m))
        return sorted(out, key=lambda x: (x.ex_date, x.symbol))

    def clear_cache(self):
        super().clear_cache()
        self._ca_memo.clear()


class CoverageFilteredPortal(FastPortal):
    """FastPortal whose stock universes (SP500, SP500_TOP<N>_DV, MEGACAP) are filtered to members that have
    daily price data (``portal.has(sym, 'bars', '1d')``, memoized per symbol).

    Why: the engine NOT_TESTABLEs a run when ANY member of a ``universe_name`` lacks data at the first
    session, and under the fail-closed identity layer some PIT S&P 500 member always does (2008: 94 of 497,
    2018: 8, 2024: 1). This is opt-in (``--universe-coverage filter``), disclosed in every result: each call is
    logged (members, kept, dropped) and summarised into ``coverage_summary()``. Dropping members WITHOUT data
    is survivorship-biased towards survivors; the coverage numbers quantify it.
    """

    STOCK_UNIVERSES = ("SP500", "S&P500", "SPX_MEMBERS", "MEGACAP")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._has_memo: dict[str, bool] = {}
        self.coverage_log: dict = {}

    def _has_bars(self, s: str) -> bool:
        v = self._has_memo.get(s)
        if v is None:
            try:
                v = bool(self.has(s, "bars", "1d"))
                if v:          # has() can be True while bars() returns nothing (e.g. ABC, 2026-10-29)
                    now = pd.Timestamp.now(tz="UTC")
                    b = self.bars(s, "1d", pd.Timestamp("1960-01-01", tz="UTC"), now, as_of=now)
                    v = b is not None and len(b) > 0
            except Exception:
                v = False
            self._has_memo[s] = v
        return v

    def universe(self, name: str, as_of):
        base = super().universe(name, as_of)
        n = name.upper()
        if not (n in self.STOCK_UNIVERSES or n.startswith("SP500")):
            return base
        kept = [s for s in base if self._has_bars(s)]
        d = str(_session_date(_utc(as_of)).date())
        if (n, d) not in self.coverage_log:
            self.coverage_log[(n, d)] = (len(base), len(kept), [s for s in base if s not in kept])
        return kept

    def coverage_series(self) -> pd.DataFrame:
        rows = [{"date": d, "universe": n, "members": m, "kept": k, "coverage": (k / m if m else np.nan)}
                for (n, d), (m, k, _) in sorted(self.coverage_log.items(), key=lambda kv: kv[0][1])]
        return pd.DataFrame(rows)

    def coverage_summary(self) -> dict:
        if not self.coverage_log:
            return {}
        rows = sorted(self.coverage_log.items(), key=lambda kv: kv[0][1])
        cov = [k / m for (_, _), (m, k, _) in rows if m]
        dropped = {}
        for (_, _), (_, _, dr) in rows:
            for s in dr:
                dropped[s] = dropped.get(s, 0) + 1
        first, last = rows[0], rows[-1]
        return {"mode": "filter (members without daily data dropped)", "n_calls": len(rows),
                "coverage_min": float(min(cov)), "coverage_mean": float(np.mean(cov)),
                "first": {"date": first[0][1], "members": first[1][0], "kept": first[1][1]},
                "last": {"date": last[0][1], "members": last[1][0], "kept": last[1][1]},
                "most_dropped": sorted(dropped.items(), key=lambda kv: -kv[1])[:15]}


# ---------------------------------------------------------------------------------------------------
# Per-session memo of PIT stock universes (performance only, identical results).
# Portal._universe('SP500', as_of) is a function of the SESSION DATE of as_of only (membership interval
# test + _member_key(ticker, d)), but rebuilding ~500 security keys costs ~4 s per call; Portal.borrow()
# calls it on EVERY call (S&P membership decides the default fee) and the engine calls borrow() for every
# short position at every session end -> ~500 s per simulated year for a 1-symbol strategy that shorts
# (f9702e1). Static universes depend on as_of itself (listing known_at) and are NOT memoized; neither are
# require_bars=True calls (bars known within the day).
_SESSION_KEYED = ("SP500", "S&P500", "SPX_MEMBERS")


def _universe_memo(self, name: str, as_of, require_bars: bool = False):
    n = str(name).upper()
    if require_bars or n not in _SESSION_KEYED:
        return Portal.universe(self, name, as_of, require_bars) if require_bars else Portal.universe(self, name, as_of)
    memo = self.__dict__.setdefault("_univ_memo", {})
    key = (n, _session_date(_utc(as_of)))
    hit = memo.get(key)
    if hit is None:
        hit = list(Portal.universe(self, name, as_of))
        memo[key] = hit
    return list(hit)


FastPortal.universe = _universe_memo

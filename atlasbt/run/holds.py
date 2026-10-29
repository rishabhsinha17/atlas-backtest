"""Run holds: strategies the integrator has put on hold until a data fix lands.

Current hold (integrator, 2026-10-29): single-stock histories have identity errors (reused tickers,
fake splice returns, unadjusted splits in alias histories). Strategies trading single stocks
(PIT S&P 500 universes, SP500_TOP50_DV, MEGACAP lists, any static list containing a non-ETF
symbol, options on single names) must not get long/final runs until the integrator lifts it.
Lift it with ``--allow-stock-universes`` (or env ATLASBT_ALLOW_STOCK=1). Results produced for a
held strategy are marked STALE (file ``results/runs/<SID>/STALE``) and reported as such by
``atlas``; ``--force`` reruns clear the marker.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

STOCK_HOLD_REASON = ("HOLD: single-stock identity resolved at as_of=range_end in the engine path (reused tickers "
                     "show the wrong company, e.g. BBT 2019 at Berkshire Hills prices); DATA re-design with "
                     "identity-qualified keys pending (integrator 2026-10-29) - stock-universe results are invalid")

_EXTRA_ETFS = {"VAW", "VCR", "VDC", "VDE", "VFH", "VGT", "VHT", "VIS", "VPU", "VOX", "VNQ", "IVV", "VOO", "IJH",
               "IJR", "MDY", "VTWO", "VIOO", "QQQM", "RSP", "KRE", "XBI", "XOP", "SMH", "VTI", "BND", "TIP"}
_INDEX = {"SPX", "SPXW", "XSP", "NDX", "RUT", "DJX", "VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "VIX1D", "SKEW",
          "VVIX", "VXN", "RVX", "VXD", "OEX", "XEO", "PUT", "BXM"}
STOCK_UNIVERSE_NAMES = ("SP500", "S&P500", "SPX_MEMBERS", "MEGACAP")


@lru_cache(maxsize=1)
def etf_set() -> frozenset:
    s = set(_EXTRA_ETFS) | _INDEX
    try:
        from atlasbt.data import universes as U

        s |= set(U.CORE_ETFS)
        for k, v in U.STATIC_UNIVERSES.items():
            if k.upper() not in ("MEGACAP",):
                s |= set(v)
    except Exception:
        pass
    try:
        from atlasbt.data.store import default_store

        d = default_store().read("reference", "nasdaqtrader", "DIRECTORY")
        if len(d) and "etf" in d.columns:
            s |= set(d.loc[d["etf"].astype(str).str.upper().isin(["Y", "TRUE", "1"]), "symbol"].astype(str))
    except Exception:
        pass
    return frozenset(x.upper() for x in s)


def stock_symbols(sid: str) -> list[str]:
    """Non-ETF, non-index symbols (or stock universe names) the strategy trades at baseline params."""
    from atlasbt.strategies import registry as R

    cls = R.get(sid)
    try:
        inst = cls()
    except Exception:
        inst = cls
    out = []
    un = (getattr(inst, "universe_name", None) or "").upper()
    if un and (un.startswith("SP500") or un in STOCK_UNIVERSE_NAMES):
        out.append(f"universe:{un}")
    etfs = etf_set()
    for s in getattr(inst, "universe", None) or []:
        if s.upper() not in etfs:
            out.append(s.upper())
    for r in getattr(inst, "requires", None) or []:
        if r.kind == "bars" and r.symbols:
            for s in r.symbols:
                if s != "*" and s.upper() not in etfs and r.critical:
                    out.append(s.upper())
    if not out and str(getattr(inst, "asset", "")).upper() == "STK":
        out.append("asset:STK (names chosen by the strategy at run time)")
    if getattr(inst, "params", {}).get("use_megacap"):
        out.append("param:use_megacap")
    return sorted(set(out))


def stock_hold_active(allow: bool = False) -> bool:
    """RE-ARMED 2026-10-29 07:10 (integrator): the engine path resolves security identity at
    as_of = range_end, so long runs see the wrong company for reused tickers (BBT 2019 filled at
    Berkshire Hills prices); DATA is moving to identity-qualified security keys. Lift with
    --allow-stock-universes / ATLASBT_ALLOW_STOCK=1 (or ATLASBT_STOCK_HOLD=0) once the integrator says so."""
    if allow or os.environ.get("ATLASBT_ALLOW_STOCK", "") == "1":
        return False
    # LIFTED for the final atlas (integrator 2026-10-29: identity redesign f39aa6b + 50c0d9a, engine eea82fa)
    return os.environ.get("ATLASBT_STOCK_HOLD", "0") == "1"


def held(sid: str, allow: bool = False) -> str:
    """Reason string if the strategy is on hold, else ''."""
    if not stock_hold_active(allow):
        return ""
    try:
        s = stock_symbols(sid)
    except Exception:
        return ""
    return f"{STOCK_HOLD_REASON} [stock exposure: {', '.join(s[:6])}{'...' if len(s) > 6 else ''}]" if s else ""


def mark_stale(results_dir: Path, sid: str, reason: str) -> bool:
    d = Path(results_dir) / "runs" / sid
    if d.exists():
        (d / "STALE").write_text(reason + "\n")
        return True
    return False


def is_stale(results_dir: Path, sid: str) -> str:
    f = Path(results_dir) / "runs" / sid / "STALE"
    return f.read_text().strip() if f.exists() else ""

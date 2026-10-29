"""Activity safeguard: a backtest that (almost) never trades must never be reported as "OK, 0% return".

``classify_activity(result, ...)`` labels every baseline / grid run:

* ``ZERO_ACTIVITY`` -- status OK but no entry at all (no fills, or no position ever opened);
* ``LOW_ACTIVITY``  -- entries below a family-specific plausibility floor:
    - intraday bar strategies (1m/5m...): < 1 entry per month of window (at least 1);
    - rebalance-based strategies (sizing PORTFOLIO / CROSS_SECTION): fewer than 2 distinct trading
      sessions (bought once, never rebalanced) or < 1 trading session per 2 years;
    - every other daily+ strategy (swing, pairs, options, events): < 5 entries per 10 years;
* ``OK`` otherwise; ``n/a`` for runs that are not status OK (NOT_TESTABLE / FAILED keep their status).

Entries = position-level round trips opened (``result.trades`` rows, including still-open ones); for
rebalance-based strategies the count is the number of distinct sessions with fills. Option combos
count once per package (a trade row per leg is collapsed by entry timestamp).

A strategy may declare why low activity is expected with a class attribute ``activity_note``
(non-empty string), e.g. an event strategy whose event class is rare; the label stays but is marked
``declared``. ``diagnose`` collects what a human needs to see why: rejected-order reasons, engine
warnings, fidelity reasons, NOT_TESTABLE text and the log path.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .common import is_intraday

REBALANCE_SIZING = {"PORTFOLIO", "CROSS_SECTION"}
LABELS = ("OK", "LOW_ACTIVITY", "ZERO_ACTIVITY", "n/a")
#: "empty leading window": first fill this many years AND this share of the window after the start
LEAD_GAP_YEARS = 2.0
LEAD_GAP_FRAC = 0.20


def leading_gap(result) -> dict:
    """Years between the window start and the first trading fill (e.g. the 2026-10-29 OCC year-pivot bug
    made every pre-2000 option silently unfillable: OK runs with nine empty leading years)."""
    eq = getattr(result, "equity", None)
    f = getattr(result, "fills", None)
    if eq is None or len(eq) < 2 or f is None or not len(f) or "ts" not in f.columns:
        return {}
    t0, t1 = pd.Timestamp(eq.index[0]), pd.Timestamp(eq.index[-1])
    first = pd.to_datetime(f["ts"], utc=True).min()
    yrs = max((first - t0).days / 365.25, 0.0)
    span = max((t1 - t0).days / 365.25, 1e-9)
    return {"first_fill": str(first.date()), "leading_gap_years": round(yrs, 2),
            "leading_gap_frac": round(yrs / span, 3),
            "empty_leading_window": bool(yrs >= LEAD_GAP_YEARS and yrs / span >= LEAD_GAP_FRAC)}


def _years(result) -> float:
    eq = getattr(result, "equity", None)
    if eq is None or len(eq) < 2:
        return 0.0
    return max((pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25, 1 / 365.25)


def _session_days(ts) -> pd.Index:
    t = pd.to_datetime(pd.Series(ts), utc=True)
    return pd.Index(t.dt.tz_convert("America/New_York").dt.normalize().unique())


def count_entries(result, sizing: str = "") -> dict:
    fills = getattr(result, "fills", None)
    trades = getattr(result, "trades", None)
    n_fills = int(len(fills)) if fills is not None else 0
    fill_days = 0
    if n_fills and "ts" in fills.columns:
        liq = fills["liquidity"].astype(str) if "liquidity" in fills.columns else pd.Series("", index=fills.index)
        trade_fills = fills[~liq.isin(["CORP_ACTION", "EXPIRE", "SETTLEMENT", "EXERCISE", "ASSIGNMENT"])]
        fill_days = len(_session_days(trade_fills["ts"])) if len(trade_fills) else 0
    n_trades = 0
    if trades is not None and len(trades):
        if "entry_ts" in trades.columns:
            n_trades = int(pd.to_datetime(trades["entry_ts"], utc=True).nunique())
        else:
            n_trades = int(len(trades))
    rebalance = (sizing or "").upper() in REBALANCE_SIZING
    n_entries = fill_days if rebalance else n_trades
    if not rebalance and n_entries == 0 and n_fills > 0:
        n_entries = fill_days          # fills but no trade rows (should not happen): be lenient
    return {"n_fills": n_fills, "n_trade_entries": n_trades, "n_fill_sessions": fill_days,
            "n_entries": int(n_entries), "entry_unit": "fill sessions" if rebalance else "trade entries"}


def activity_floor(freq: str, sizing: str, years: float) -> tuple[int, str]:
    if is_intraday(freq):
        months = years * 12
        return max(1, int(round(months))), ">= 1 entry per month (intraday)"
    if (sizing or "").upper() in REBALANCE_SIZING:
        return max(2, int(round(years / 2))), ">= 2 trading sessions and >= 1 per 2 years (rebalance-based)"
    return max(1, int(round(0.5 * years))), ">= 5 entries per 10 years (daily+)"


def classify_activity(result, freq: str, sizing: str, declared_note: str = "") -> dict:
    status = getattr(result.status, "value", str(result.status))
    out = {"activity": "n/a", "declared_note": declared_note or ""}
    if status != "OK":
        return out
    c = count_entries(result, sizing)
    yrs = _years(result)
    floor, rule = activity_floor(freq, sizing, yrs)
    out.update(c)
    out.update(leading_gap(result))
    out.update({"years": round(yrs, 2), "floor": floor, "rule": rule,
                "entries_per_year": round(c["n_entries"] / yrs, 3) if yrs > 0 else None})
    if c["n_fills"] == 0 or c["n_entries"] == 0:
        out["activity"] = "ZERO_ACTIVITY"
    elif c["n_entries"] < floor:
        out["activity"] = "LOW_ACTIVITY"
    else:
        out["activity"] = "OK"
    out["declared"] = bool(declared_note) and out["activity"] != "OK"
    return out


#: Strategy-side interface (agreed for the options families; any strategy may use it):
#:   ctx.log(msg, activity="<category>")   categories: selection_failure, combo_cancelled, order_rejected,
#:                                         no_signal, data_missing, skip (free text allowed)
#:   ctx.flag_fidelity(Fidelity.T1_BAR, "activity <SID>: <generic reason>")  (aggregated reasons)
#: Logs without an `activity` key are categorised by keywords.
_LOG_RULES = (("selection_failure", ("selection", "no listed", "selectionerror", "no series")),
              ("combo_cancelled", ("cancel",)), ("order_rejected", ("reject",)),
              ("data_missing", ("failed", "unavailable", "missing", "no chain", "no data")),
              ("skip", ("skip", "no trade")))


def _log_category(entry) -> str:
    if isinstance(entry, dict):
        cat = entry.get("activity")
        if cat:
            return str(cat)
        msg = str(entry.get("msg", "")).lower()
    else:
        msg = str(entry).lower()
    for cat, keys in _LOG_RULES:
        if any(k in msg for k in keys):
            return cat
    return "other"


def order_activity(diag: dict) -> dict:
    """Order outcome counts from the (unpruned) engine diagnostics: by type/status and cancel/reject reasons."""
    o = diag.get("orders")
    out = {}
    if isinstance(o, pd.DataFrame) and len(o):
        t = o["type"].astype(str) if "type" in o.columns else pd.Series("?", index=o.index)
        st = o["status"].astype(str) if "status" in o.columns else pd.Series("?", index=o.index)
        out["by_type_status"] = {f"{a}/{b}": int(n) for (a, b), n in pd.crosstab(t, st).stack().items() if n}
        combo = t.str.upper().str.contains("COMBO")
        out["combo_submitted"] = int(combo.sum())
        out["combo_filled"] = int((combo & st.str.upper().str.contains("FILLED") & ~st.str.upper().str.contains("PARTIAL")).sum())
        out["combo_cancelled"] = int((combo & st.str.upper().str.contains("CANCEL")).sum())
        out["combo_rejected"] = int((combo & st.str.upper().str.contains("REJECT")).sum())
        if "reason" in o.columns:
            bad = st.str.upper().str.contains("CANCEL|REJECT|EXPIRED")
            rs = o.loc[bad, "reason"].astype(str).str.replace(r"\d+(\.\d+)?", "#", regex=True).str[:100]
            out["cancel_reject_reasons"] = Counter(rs).most_common(5)
    return out


def log_activity(diag: dict) -> dict:
    logs = diag.get("logs") or []
    c = Counter(_log_category(e) for e in logs)
    ex = {}
    for e in logs:
        cat = _log_category(e)
        if cat != "other" and cat not in ex:
            ex[cat] = str(e.get("msg") if isinstance(e, dict) else e)[:160]
    return {"n_logs": len(logs), "by_category": dict(c), "examples": ex}


def diagnose(result, log_path: str = "") -> dict:
    """Why a run might not trade: order outcomes (incl. cancelled/rejected combos), categorised strategy
    logs (selection failures ...), rejected orders, warnings, fidelity reasons. Call it on the UNPRUNED
    result (before pipeline._prune_diagnostics)."""
    diag = getattr(result, "diagnostics", None) or {}
    rej = diag.get("rejected_orders") or []
    reasons = Counter()
    for r in rej:
        why = r.get("reason") if isinstance(r, dict) else getattr(r, "reject_reason", str(r))
        reasons[str(why)[:140]] += 1
    warn = [str(w)[:200] for w in (diag.get("warnings") or [])[:8]]
    fid = getattr(result, "fidelity", None) or {}
    freasons = []
    for r in (fid.get("reasons") or [])[:8]:
        if isinstance(r, dict):
            freasons.append(f"T{r.get('tier')}: {str(r.get('reason'))[:120]} (x{r.get('count')})")
        else:
            freasons.append(str(r)[:140])
    orders = diag.get("orders")
    return {"n_orders": len(orders) if isinstance(orders, pd.DataFrame) else diag.get("n_orders"),
            "orders": order_activity(diag), "strategy_logs": log_activity(diag),
            "n_rejected": diag.get("n_rejected_orders", len(rej)),
            "top_reject_reasons": reasons.most_common(5), "warnings": warn, "fidelity_reasons": freasons,
            "not_testable_reason": getattr(result, "not_testable_reason", ""),
            "sim": diag.get("sim"), "log": log_path}


def declared_note_for(sid: str) -> str:
    try:
        from atlasbt.strategies import registry as R

        return str(getattr(R.get(sid), "activity_note", "") or "")
    except Exception:
        return ""

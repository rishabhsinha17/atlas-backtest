"""Round-trip trade tracking, fidelity scorecard and result assembly."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import Fidelity, Fill


@dataclass
class _OpenTrade:
    symbol: str
    side: int
    entry_ts: pd.Timestamp
    mult: float
    entry_order_id: int
    initial_stop: Optional[float]
    tag: str = ""
    qty: float = 0.0              # open quantity (abs)
    max_qty: float = 0.0
    entry_qty: float = 0.0
    entry_value: float = 0.0
    exit_qty: float = 0.0
    exit_value: float = 0.0
    fees: float = 0.0
    hi: float = -math.inf
    lo: float = math.inf
    n_fills: int = 0
    worst_fid: int = 0
    exit_ts: Optional[pd.Timestamp] = None
    exit_tag: str = ""
    carry: float = 0.0            # attributed borrow fees / dividends (signed cash, + = inflow)
    bars: int = 0
    trade_id: int = 0
    planned_entry: Optional[float] = None

    @property
    def avg_entry(self) -> float:
        return self.entry_value / self.entry_qty if self.entry_qty else float("nan")


class TradeTracker:
    """Position-level round trips: a trade opens when a symbol's position
    leaves zero and closes when it returns to zero; a flip closes one trade
    and opens another with the remainder."""

    def __init__(self):
        self.open: dict[str, _OpenTrade] = {}
        self.closed: list[dict] = []
        self._next_id = 1

    def on_split(self, symbol: str, ratio: float, cil_qty: float = 0.0, cil_price: float = float("nan"),
                 ts=None) -> None:
        """Rescale the open trade on a split (quantities x ratio, prices / ratio;
        values unchanged) and book any cash-in-lieu fraction as a partial exit."""
        tr = self.open.get(symbol)
        if tr is None or not (ratio > 0) or ratio == 1.0:
            return
        r = ratio
        tr.qty *= r
        tr.max_qty *= r
        tr.entry_qty *= r
        tr.exit_qty *= r
        if tr.initial_stop is not None:
            tr.initial_stop /= r
        if tr.planned_entry is not None:
            tr.planned_entry /= r
        tr.hi /= r
        tr.lo /= r
        if cil_qty and cil_price == cil_price:
            close = min(abs(cil_qty), tr.qty)
            tr.exit_qty += close
            tr.exit_value += close * cil_price
            tr.qty -= close
            if tr.qty <= 1e-9:
                tr.exit_ts = ts
                tr.exit_tag = "cash_in_lieu"
                self._close(symbol)

    def rekey(self, old: str, new: str) -> None:
        """Symbol change / option contract adjustment: same trade, new key."""
        tr = self.open.pop(old, None)
        if tr is None:
            return
        tr.symbol = new
        other = self.open.get(new)
        if other is None:
            self.open[new] = tr
            return
        other.qty += tr.qty
        other.max_qty = max(other.max_qty, other.qty)
        other.entry_qty += tr.entry_qty
        other.entry_value += tr.entry_value
        other.exit_qty += tr.exit_qty
        other.exit_value += tr.exit_value
        other.fees += tr.fees
        other.carry += tr.carry

    def on_spinoff(self, parent: str, child: str, child_qty: float, basis: float, frac: float, ts,
                   mult: float = 1.0) -> None:
        """Move ``frac`` of the parent's entry value to a child trade opened at
        the allocated basis."""
        tr = self.open.get(parent)
        if tr is not None and frac:
            tr.entry_value -= frac * tr.avg_entry * tr.qty
        if not child_qty:
            return
        ct = self.open.get(child)
        if ct is None:
            ct = _OpenTrade(child, 1 if child_qty > 0 else -1, ts, mult, -1, None, "spinoff")
            ct.trade_id = self._next_id
            self._next_id += 1
            self.open[child] = ct
        q = abs(child_qty)
        ct.qty += q
        ct.max_qty = max(ct.max_qty, ct.qty)
        ct.entry_qty += q
        ct.entry_value += abs(basis) / (mult or 1.0)
        px = abs(basis) / q / (mult or 1.0)
        ct.hi = max(ct.hi, px)
        ct.lo = min(ct.lo, px)

    def attribute(self, symbol: str, amount: float) -> None:
        """Attribute a symbol-specific cash flow (borrow fee, dividend,
        payment in lieu, withholding) to the open trade on that symbol."""
        tr = self.open.get(symbol)
        if tr is not None:
            tr.carry += amount

    def on_fill(self, fill: Fill, pos_before: float, initial_stop: Optional[float], tag: str = "",
                planned_entry: Optional[float] = None) -> None:
        sym = fill.instrument.symbol
        q = fill.quantity
        px = fill.price
        mult = fill.instrument.multiplier
        fees = fill.costs.explicit
        remaining = q
        if pos_before != 0 and (pos_before > 0) != (q > 0):
            tr = self.open.get(sym)
            closing = min(abs(q), abs(pos_before))
            if tr is not None:
                share = closing / abs(q)
                tr.exit_qty += closing
                tr.exit_value += closing * px
                tr.fees += fees * share
                tr.qty -= closing
                tr.n_fills += 1
                tr.worst_fid = max(tr.worst_fid, int(fill.fidelity))
                tr.hi = max(tr.hi, px)
                tr.lo = min(tr.lo, px)
                tr.exit_tag = tag
                if tr.qty <= 1e-9:
                    tr.exit_ts = fill.ts
                    self._close(sym)
                fees *= (1 - share)
            remaining = q + math.copysign(closing, pos_before)
        if abs(remaining) > 1e-9:
            tr = self.open.get(sym)
            if tr is None:
                tr = _OpenTrade(sym, 1 if remaining > 0 else -1, fill.ts, mult, fill.order_id, initial_stop, tag)
                tr.trade_id = self._next_id
                self._next_id += 1
                self.open[sym] = tr
            tr.qty += abs(remaining)
            tr.max_qty = max(tr.max_qty, tr.qty)
            tr.entry_qty += abs(remaining)
            tr.entry_value += abs(remaining) * px
            tr.fees += fees
            tr.n_fills += 1
            tr.worst_fid = max(tr.worst_fid, int(fill.fidelity))
            tr.hi = max(tr.hi, px)
            tr.lo = min(tr.lo, px)
            if tr.initial_stop is None and initial_stop is not None:
                tr.initial_stop = initial_stop
            if tr.planned_entry is None and planned_entry is not None:
                tr.planned_entry = planned_entry

    def on_bar(self, symbol: str, high: float, low: float, count_bar: bool = True) -> None:
        tr = self.open.get(symbol)
        if tr is None:
            return
        if count_bar:
            tr.bars += 1
        if high == high:
            tr.hi = max(tr.hi, high)
        if low == low:
            tr.lo = min(tr.lo, low)

    def _close(self, sym: str) -> None:
        tr = self.open.pop(sym)
        self.closed.append(self._row(tr, closed=True))

    def _row(self, tr: _OpenTrade, closed: bool, mark: float = float("nan")) -> dict:
        ent = tr.avg_entry
        ex = tr.exit_value / tr.exit_qty if tr.exit_qty else float("nan")
        gross = tr.side * (tr.exit_value - tr.exit_qty * ent) * tr.mult if tr.exit_qty else 0.0
        if not closed and mark == mark:
            gross += tr.side * (mark - ent) * tr.qty * tr.mult
        net = gross - tr.fees + tr.carry
        risk_ps = abs(ent - tr.initial_stop) if tr.initial_stop is not None else float("nan")
        risk_usd = risk_ps * tr.entry_qty * tr.mult if risk_ps == risk_ps else float("nan")
        if tr.side > 0:
            mae_ps = max(ent - tr.lo, 0.0)
            mfe_ps = max(tr.hi - ent, 0.0)
        else:
            mae_ps = max(tr.hi - ent, 0.0)
            mfe_ps = max(ent - tr.lo, 0.0)
        size = tr.max_qty * tr.mult
        planned = (abs(tr.planned_entry - tr.initial_stop) * tr.entry_qty * tr.mult
                   if tr.planned_entry is not None and tr.initial_stop is not None else float("nan"))
        return {
            "trade_id": tr.trade_id, "symbol": tr.symbol, "side": "LONG" if tr.side > 0 else "SHORT",
            "entry_ts": tr.entry_ts, "exit_ts": tr.exit_ts, "quantity": tr.side * tr.entry_qty,
            "entry_price": ent, "exit_price": ex,
            "pnl": net, "pnl_gross": gross, "costs": gross - net,
            "r_multiple": net / risk_usd if risk_usd and risk_usd == risk_usd else float("nan"),
            "initial_risk": risk_usd,
            "mae": -mae_ps * size, "mfe": mfe_ps * size, "bars_held": tr.bars, "multiplier": tr.mult,
            "exit_reason": (tr.exit_tag or "exit") if closed else "open",
            # extras
            "max_qty": tr.max_qty, "fees": tr.fees, "carry": tr.carry, "initial_stop": tr.initial_stop,
            "return_pct": net / (ent * tr.entry_qty * tr.mult) if ent and tr.entry_qty else float("nan"),
            "mae_per_unit": mae_ps, "mfe_per_unit": mfe_ps,
            "mae_pct": mae_ps / ent if ent else float("nan"), "mfe_pct": mfe_ps / ent if ent else float("nan"),
            "mae_r": mae_ps / risk_ps if risk_ps == risk_ps and risk_ps > 0 else float("nan"),
            "mfe_r": mfe_ps / risk_ps if risk_ps == risk_ps and risk_ps > 0 else float("nan"),
            "holding_days": ((tr.exit_ts - tr.entry_ts).total_seconds() / 86400.0) if tr.exit_ts is not None else float("nan"),
            "n_fills": tr.n_fills, "worst_fidelity": Fidelity(tr.worst_fid).name,
            "entry_tag": tr.tag, "closed": closed,
            "planned_risk": planned,
            "planned_r_multiple": net / planned if planned == planned and planned > 0 else float("nan"),
        }

    def frame(self, marks: Optional[dict] = None) -> pd.DataFrame:
        from ..core.schemas import TRADE_COLUMNS
        rows = list(self.closed)
        for sym, tr in sorted(self.open.items()):
            rows.append(self._row(tr, closed=False, mark=(marks or {}).get(sym, float("nan"))))
        extras = ["max_qty", "fees", "carry", "initial_stop", "return_pct", "mae_per_unit", "mfe_per_unit",
                  "mae_pct", "mfe_pct", "mae_r", "mfe_r", "holding_days", "n_fills", "worst_fidelity",
                  "entry_tag", "closed", "planned_risk", "planned_r_multiple"]
        return pd.DataFrame(rows, columns=list(TRADE_COLUMNS) + extras)


_GRADE = {Fidelity.T0_QUOTE: "A", Fidelity.T1_BAR: "B", Fidelity.T2_PROXY: "C",
          Fidelity.T3_MODELED: "D", Fidelity.NA: "E"}


class FidelityLedger:
    """Collects per-fill tiers/notional and (tier, reason) counts.

    Scorecard shape (core.schemas): ``{"tiers": {TIER: {"fills", "notional"}},
    "reasons": [{"tier", "reason", "count"}], "grade": "A".."E" | "NA"}``.
    Grade = the worst tier that carries >= ``grade_share`` of traded notional
    or is raised by a strategy/engine flag or a required data input
    (T0 -> A, T1 -> B, T2 -> C, T3 -> D, NA -> E); "NA" when nothing traded
    and nothing was flagged."""

    grade_share = 0.05

    def __init__(self):
        self.fills_by_tier: Counter = Counter()
        self.notional_by_tier: defaultdict = defaultdict(float)
        self.reasons: Counter = Counter()          # (tier name, reason) -> count
        self.flags: list[dict] = []

    def on_fill(self, fill: Fill, reasons: tuple[str, ...]) -> None:
        t = Fidelity(fill.fidelity).name
        self.fills_by_tier[t] += 1
        self.notional_by_tier[t] += abs(fill.quantity * fill.price * fill.instrument.multiplier)
        for r in reasons:
            if r:
                self.reasons[(t, r)] += 1

    def note(self, tier: Fidelity, reason: str) -> None:
        self.reasons[(Fidelity(tier).name, reason)] += 1

    def flag(self, ts, tier: Fidelity, reason: str) -> None:
        self.flags.append({"ts": ts, "tier": Fidelity(tier).name, "reason": reason})
        self.note(tier, reason)

    def scorecard(self, requirement_fidelity: dict) -> dict:
        total_notional = float(sum(self.notional_by_tier.values()))
        tiers = {}
        for t in Fidelity:
            n = self.fills_by_tier.get(t.name, 0)
            if n:
                v = float(self.notional_by_tier.get(t.name, 0.0))
                tiers[t.name] = {"fills": int(n), "notional": v,
                                 "notional_share": v / total_notional if total_notional else float("nan")}
        cands = [Fidelity[t] for t, d in tiers.items()
                 if total_notional == 0 or d["notional"] / total_notional >= self.grade_share]
        cands += [Fidelity[f["tier"]] for f in self.flags]
        cands += [Fidelity(v) for v in requirement_fidelity.values() if v is not None and v != int(Fidelity.NA)]
        grade = _GRADE[max(cands)] if cands else "NA"
        reasons = [{"tier": t, "reason": r, "count": int(c)}
                   for (t, r), c in sorted(self.reasons.items(), key=lambda kv: (-kv[1], kv[0]))]
        return {
            "tiers": tiers,
            "reasons": reasons,
            "grade": grade,
            "fills": int(sum(self.fills_by_tier.values())),
            "notional": total_notional,
            "strategy_flags": self.flags[:1000],
            "requirements": {k: (Fidelity(v).name if v is not None else None) for k, v in requirement_fidelity.items()},
        }


def fills_frame(rows: list[dict]) -> pd.DataFrame:
    from ..core.schemas import FILL_COLUMNS
    df = pd.DataFrame(rows, columns=list(FILL_COLUMNS) + ["fidelity_name"])
    if df.empty:
        df["ts"] = pd.Series(pd.DatetimeIndex([], tz="UTC"), dtype="datetime64[ns, UTC]")
    return df

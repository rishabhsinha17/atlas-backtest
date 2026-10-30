"""Corporate-action cross-checks: Yahoo actions vs an independent source and vs implied price gaps.

The engine applies the corporate actions the portal serves (Yahoo for Yahoo segments, see
docs/DATA.md §8.1). This module validates them, per symbol, against

1. **DoltHub** (``reference/dolthub/SPLIT`` from 2014-03-28 and ``reference/dolthub/DIVIDEND``),
   an independent vendor; events match when the ex-dates are within ``date_tol_days`` calendar
   days and the split ratio agrees to 1 % (dividend amount to ``amount_rel_tol``, or a match on
   a split-adjusted basis, which is flagged separately as ``amount_split_basis``);
2. **implied price gaps** on DoltHub's as-traded daily bars (independent of Yahoo's actions:
   Yahoo's raw bars are reconstructed by un-adjusting with Yahoo's own split factors, so a gap
   check on them is circular and is not used): after an r-for-1 split the ex-date open should be
   about previous close / r (``|log(gap * r)| < gap_tol``).

Identity guard (tickers are reused, DATA.md §8.1): an event is only compared when Yahoo's raw
close and DoltHub's close on the last common session before the ex-date agree within
``identity_tol`` (same security); otherwise the event is ``identity_unconfirmed`` and excluded
from the rate. Spin-offs encoded by Yahoo as fractional splits are classified first
(``actions_classify``) and are not split-compared.

Statuses: ``agree``; ``ratio_mismatch``; ``date_mismatch`` (same ratio, ex-dates 4-31 days apart;
the note says which date the price gap supports); ``missing_in_dolthub`` / ``missing_in_yahoo``;
``price_contradicts`` (a Yahoo split no source or price supports); ``spinoff_served_as_split``
(a known spin-off still served as a SPLIT row: shares multiplied AND value distributed);
``dolthub_extra_row`` (DoltHub repeats a split on its announcement/record date -- not counted);
``served_as_spinoff`` (DoltHub split = Yahoo spin-off on the same date -- not counted);
``identity_unconfirmed`` (excluded); ``amount_mismatch`` / ``amount_split_basis`` (dividends).

Conventions (DATA.md): nothing is corrected here; every disagreement is FLAGGED with a severity:
``error`` = a split that the independent price gap contradicts (or a ratio mismatch) -- consumers
should fail closed on that symbol/date; ``warn`` = present in one source only, or dividend amount
mismatch. ``flags(report)`` returns the flagged rows; the report is persisted to
``cache/qa/ca_crosscheck_<date>.csv`` (+ ``_summary.json``) by the CLI::

    python -m atlasbt.data.ca_crosscheck [SYMBOL ...] [--limit N] [--no-write]
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import frames
from .store import ParquetStore, default_store

SPLIT_START = pd.Timestamp("2014-03-28")     # first ex-date in DoltHub's split table

STATUS_MISMATCH = {"ratio_mismatch", "missing_in_dolthub", "missing_in_yahoo", "amount_mismatch",
                   "price_contradicts", "spinoff_served_as_split", "date_mismatch"}


def _naive_dates(x) -> np.ndarray:
    t = pd.to_datetime(pd.Series(list(x)))
    if t.dt.tz is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t.dt.normalize().to_numpy()


def _close_series(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if bars is None or not len(bars):
        e = pd.Series(dtype=float)
        return e, e
    d = frames.to_naive_date(bars.index)
    c = pd.Series(bars["close"].to_numpy(float), index=d)
    o = pd.Series(bars["open"].to_numpy(float), index=d)
    c = c[~c.index.duplicated(keep="last")]
    o = o[~o.index.duplicated(keep="last")]
    return o, c


class CrossChecker:
    def __init__(self, store: Optional[ParquetStore] = None, date_tol_days: int = 3,
                 amount_rel_tol: float = 0.01, gap_tol: float = 0.25, identity_tol: float = 0.05):
        self.store = store or default_store()
        self.date_tol = pd.Timedelta(days=date_tol_days)
        self.amount_rel_tol = amount_rel_tol
        self.gap_tol = gap_tol
        self.identity_tol = identity_tol
        self._ds = self._dd = None

    # ------------------------------------------------------------------ independent source
    def _dolt(self):
        if self._ds is None:
            sp = self.store.read("reference", "dolthub", "SPLIT")
            dv = self.store.read("reference", "dolthub", "DIVIDEND")
            sp = sp.reset_index() if "act_symbol" not in sp.columns else sp
            dv = dv.reset_index() if "act_symbol" not in dv.columns else dv
            if len(sp):
                sp = sp.assign(ex=pd.to_datetime(sp["ex_date"]).dt.normalize(),
                               ratio=sp["to_factor"].astype(float) / sp["for_factor"].astype(float))
            if len(dv):
                dv = dv.assign(ex=pd.to_datetime(dv["ex_date"]).dt.normalize(), amount=dv["amount"].astype(float))
            self._ds = {k: g for k, g in sp.groupby("act_symbol")} if len(sp) else {}
            self._dd = {k: g for k, g in dv.groupby("act_symbol")} if len(dv) else {}
            self.split_end = sp["ex"].max() if len(sp) else SPLIT_START
            self.div_end = dv["ex"].max() if len(dv) else pd.Timestamp("1970-01-01")
        return self._ds, self._dd

    def symbols(self) -> list[str]:
        return self.store.symbols("actions", "yahoo")

    # ------------------------------------------------------------------ per symbol
    def check_symbol(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        ds, dd = self._dolt()
        ya = self.store.read("actions", "yahoo", sym)
        if ya is None or not len(ya):
            return pd.DataFrame()
        ya = ya.reset_index() if "type" not in ya.columns else ya
        ybars = self.store.read("bars_1d", "yahoo", sym)
        dbars = self.store.read("bars_1d", "dolthub", sym)
        try:
            from .actions_classify import classify
            ya = classify(sym, ya, ybars)
        except Exception:       # classification is best effort here (raw actions still compared)
            pass
        ya = ya.assign(ex=_naive_dates(ya["ex_date"]))
        _, yc = _close_series(ybars)
        do, dc = _close_series(dbars)
        rows: list[dict] = []
        d_first = dc.index.min() if len(dc) else None
        d_last = dc.index.max() if len(dc) else None

        def identity(ex) -> Optional[bool]:
            """Same security in both sources on the last common session before ex."""
            if d_first is None:
                return None
            common = yc.index[(yc.index < ex)].intersection(dc.index[dc.index < ex])
            if not len(common):
                return None
            d = common.max()
            if (ex - d).days > 10:
                return None
            a, b = float(yc.loc[d]), float(dc.loc[d])
            if not (a > 0 and b > 0):
                return None
            return abs(math.log(a / b)) <= self.identity_tol

        def price_gap(ex) -> float:
            if d_first is None or ex not in do.index:
                return float("nan")
            prev = dc.index[dc.index < ex]
            if not len(prev) or (ex - prev.max()).days > 5:
                return float("nan")
            pc = float(dc.loc[prev.max()])
            return float(do.loc[ex]) / pc if pc > 0 else float("nan")

        def in_dolt_window(ex, start) -> bool:
            return d_first is not None and ex >= max(start, d_first) and ex <= d_last

        # ---------------------------------------------------------------- splits
        try:
            from .actions_classify import _SPIN
            # composite events (reverse split + spin-off on one ex-date: EXPE/ITT/DD) legitimately
            # carry both rows; only pure spin-offs left as a SPLIT are double-encoded
            spin_keys = {(k[0], str(k[1])) for k, v in _SPIN.items() if not v.get("split")}
        except Exception:
            spin_keys = set()
        ysp = ya[ya["type"] == "SPLIT"]
        # a known spin-off still present as a SPLIT row is double-encoded wherever it falls
        for _, r in ysp.iterrows():
            ex = r["ex"]
            if (sym, ex.date().isoformat()) in spin_keys and not in_dolt_window(ex, SPLIT_START):
                rows.append(self._row(sym, "SPLIT", ex, float(r["ratio"]), float("nan"),
                                      "spinoff_served_as_split", None, float("nan"), None))
        dsp = ds.get(sym)
        if dsp is None:
            dsp = pd.DataFrame({"ex": pd.Series(dtype="datetime64[ns]"), "ratio": pd.Series(dtype=float)})
        used = set()
        for _, r in ysp.iterrows():
            ex, ry = r["ex"], float(r["ratio"])
            if not in_dolt_window(ex, SPLIT_START) or ex > self.split_end:
                continue
            idn = identity(ex)
            g = price_gap(ex)
            cand = dsp[(dsp["ex"] - ex).abs() <= self.date_tol]
            status, rd, note = "missing_in_dolthub", float("nan"), ""
            if len(cand):
                k = (np.log(cand["ratio"].to_numpy()) - math.log(ry))
                dist = np.abs((cand["ex"] - ex).dt.days.to_numpy())
                j = int(np.lexsort((dist, np.round(np.abs(k), 2)))[0])     # same ratio first, then nearest date
                rd = float(cand["ratio"].iloc[j])
                used.add(cand.index[j])
                status = "agree" if abs(k[j]) < 0.01 else "ratio_mismatch"
            else:
                # same ratio on another date within a month: one source dates it off the ex-date
                # (DoltHub sometimes carries the announcement or pay date)
                near = dsp[((dsp["ex"] - ex).abs() <= pd.Timedelta(days=31)) & ~dsp.index.isin(list(used))
                           & (np.abs(np.log(dsp["ratio"].astype(float)) - math.log(ry)) < 0.01)]
                if len(near):
                    j = int(np.argmin((near["ex"] - ex).abs().to_numpy()))
                    rd = float(near["ratio"].iloc[j])
                    used.add(near.index[j])
                    status = "date_mismatch"
                    gd = price_gap(near["ex"].iloc[j])
                    note = (f"dolthub ex_date {near['ex'].iloc[j].date()}; price supports "
                            f"{'yahoo' if self._confirms(g, ry) else 'dolthub' if self._confirms(gd, rd) else 'neither'}")
            confirms = self._confirms(g, ry)
            if status != "agree" and confirms is False:
                status = "price_contradicts" if status in ("missing_in_dolthub", "date_mismatch") else status
            if (sym, ex.date().isoformat()) in spin_keys:
                status = "spinoff_served_as_split"
            rows.append(self._row(sym, "SPLIT", ex, ry, rd, status, idn, g, confirms, note))
        for i, r in dsp.iterrows():
            if i in used:
                continue
            ex = r["ex"]
            if not in_dolt_window(ex, SPLIT_START):
                continue
            yfirst = yc.index.min() if len(yc) else None
            if yfirst is None or ex < yfirst or ex > yc.index.max():
                continue
            g = price_gap(ex)
            confirms = self._confirms(g, float(r["ratio"]))
            twin = ysp[((ysp["ex"] - ex).abs() <= pd.Timedelta(days=31))
                       & (np.abs(np.log(ysp["ratio"].astype(float)) - math.log(float(r["ratio"]))) < 0.01)]
            # DoltHub often carries extra rows for one split (announcement / record / pay date)
            spin = ya[(ya["type"] == "SPINOFF") & ((ya["ex"] - ex).abs() <= self.date_tol)]
            st = "dolthub_extra_row" if len(twin) else ("served_as_spinoff" if len(spin) else "missing_in_yahoo")
            rows.append(self._row(sym, "SPLIT", ex, float("nan"), float(r["ratio"]), st,
                                  identity(ex), g, confirms))

        # ---------------------------------------------------------------- dividends
        ydv = ya[ya["type"].isin(["CASH_DIVIDEND", "SPECIAL_DIVIDEND"])]
        ddv = dd.get(sym)
        if ddv is not None and len(ddv) and len(ydv):
            lo = max(ydv["ex"].min(), ddv["ex"].min(), d_first if d_first is not None else ddv["ex"].min())
            hi = min(ydv["ex"].max(), ddv["ex"].max(), d_last if d_last is not None else ddv["ex"].max())
            ydv = ydv[(ydv["ex"] >= lo) & (ydv["ex"] <= hi)]
            ddv = ddv[(ddv["ex"] >= lo) & (ddv["ex"] <= hi)]
            splits_all = ya[ya["type"] == "SPLIT"][["ex", "ratio"]]
            used = set()
            for _, r in ydv.iterrows():
                ex, ay = r["ex"], float(r["amount"])
                cand = ddv[((ddv["ex"] - ex).abs() <= self.date_tol) & ~ddv.index.isin(list(used))]
                status, ad = "missing_in_dolthub", float("nan")
                if len(cand):
                    rel = np.abs(cand["amount"].to_numpy() / ay - 1.0) if ay else np.full(len(cand), np.inf)
                    j = int(np.argmin(rel))
                    ad = float(cand["amount"].iloc[j])
                    used.add(cand.index[j])
                    if rel[j] <= self.amount_rel_tol or abs(ad - ay) <= 0.001:
                        status = "agree"
                    else:
                        # one source split-adjusted for (some of) the later splits?
                        later = splits_all[splits_all["ex"] > ex].sort_values("ex")["ratio"].astype(float)
                        facs = np.cumprod(later.to_numpy()) if len(later) else np.array([])
                        hit = any(abs(ad * f / ay - 1) <= self.amount_rel_tol or abs(ay * f / ad - 1) <= self.amount_rel_tol
                                  for f in facs)
                        status = "amount_split_basis" if hit else "amount_mismatch"
                rows.append(self._row(sym, "DIVIDEND", ex, ay, ad, status, identity(ex), float("nan"), None))
            for i, r in ddv.iterrows():
                if i in used:
                    continue
                rows.append(self._row(sym, "DIVIDEND", r["ex"], float("nan"), float(r["amount"]),
                                      "missing_in_yahoo", identity(r["ex"]), float("nan"), None))
        return pd.DataFrame(rows)

    def _confirms(self, g: float, r: float) -> Optional[bool]:
        """Does the independent overnight gap support split ratio r? None = inconclusive (no
        price, or |log r| < 0.2: small factors are within normal overnight moves)."""
        if not (g == g and g > 0 and r == r and r > 0) or abs(math.log(r)) < 0.2:
            return None
        with_split, without = abs(math.log(g * r)), abs(math.log(g))
        if with_split < without and with_split < self.gap_tol:
            return True
        if without <= with_split:
            return False                        # the price did not move as the split implies
        return None                             # moved towards the split but by an unusual amount

    @staticmethod
    def _row(sym, kind, ex, yv, dv, status, idn, gap, confirms, note: str = "") -> dict:
        if idn is False and status != "spinoff_served_as_split":
            status_eff = "identity_unconfirmed"
        else:
            status_eff = status
        sev = ""
        if status_eff == "spinoff_served_as_split":
            sev = "error"                       # share count multiplied AND spin value paid
        elif kind == "SPLIT" and status_eff == "missing_in_yahoo" and confirms is True:
            sev = "error"                       # a real split (price gap) absent from the served actions
        elif kind == "SPLIT" and status_eff not in ("identity_unconfirmed", "agree") and yv == yv and (
                confirms is False or (status_eff == "ratio_mismatch" and confirms is None)):
            sev = "error"                       # independent prices contradict Yahoo's (served) split
        elif status_eff in STATUS_MISMATCH or status_eff == "amount_split_basis":
            sev = "warn"
        return {"symbol": sym, "kind": kind, "ex_date": pd.Timestamp(ex).date().isoformat(),
                "yahoo": yv, "dolthub": dv, "status": status_eff, "raw_status": status,
                "identity_confirmed": idn, "dolthub_price_gap": gap, "price_confirms": confirms,
                "severity": sev, "note": note}

    # ------------------------------------------------------------------ report
    def run(self, symbols: Optional[Iterable[str]] = None, progress: bool = False) -> pd.DataFrame:
        syms = list(symbols) if symbols is not None else self.symbols()
        out = []
        for i, s in enumerate(syms):
            try:
                df = self.check_symbol(s)
            except Exception as e:      # report, never abort the batch
                df = pd.DataFrame([{"symbol": s.upper(), "kind": "ERROR", "status": "error",
                                    "severity": "error", "note": f"{type(e).__name__}: {e}"}])
            if len(df):
                out.append(df)
            if progress and (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(syms)}", flush=True)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["symbol", "kind", "ex_date", "yahoo", "dolthub", "status", "severity"])


NOT_COUNTED = {"identity_unconfirmed", "dolthub_extra_row", "served_as_spinoff"}


def summarize(report: pd.DataFrame) -> dict:
    """Mismatch rate per kind over events compared in both sources' window with the identity
    confirmed (identity_unconfirmed / identity unknown are reported, not counted)."""
    out = {"symbols": int(report["symbol"].nunique()) if len(report) else 0}
    for kind in ("SPLIT", "DIVIDEND"):
        r = report[report["kind"] == kind] if len(report) else report
        comp = r[(~r["status"].isin(NOT_COUNTED)) & (r["identity_confirmed"] == True)]  # noqa: E712
        counts = comp["status"].value_counts().to_dict()
        for st in ("dolthub_extra_row", "served_as_spinoff"):
            if len(r) and (r["status"] == st).any():
                counts_nc = int((r["status"] == st).sum())
                out.setdefault("not_counted", {})[f"{kind}:{st}"] = counts_nc
        n = int(len(comp))
        mism = int(comp["status"].isin(STATUS_MISMATCH).sum())
        d = {"events_compared": n, "agree": int(counts.get("agree", 0)), "mismatches": mism,
             "mismatch_rate": (mism / n) if n else float("nan"), "by_status": {k: int(v) for k, v in counts.items()},
             "identity_unconfirmed": int((r["status"] == "identity_unconfirmed").sum()) if len(r) else 0,
             "identity_unknown": int(r["identity_confirmed"].isna().sum()) if len(r) else 0}
        if kind == "SPLIT" and len(comp):
            pc = comp["price_confirms"].dropna()
            d["price_gap_checked"] = int(len(pc))
            d["price_gap_contradicts"] = int((~pc.astype(bool)).sum())
            d["errors"] = int((comp["severity"] == "error").sum())
        if kind == "SPLIT" and len(r):
            dbl = r[r["status"] == "spinoff_served_as_split"]
            d["spinoff_served_as_split"] = sorted(f"{a} {b}" for a, b in zip(dbl["symbol"], dbl["ex_date"]))
        if kind == "DIVIDEND":
            d["amount_split_basis"] = int(counts.get("amount_split_basis", 0))
        out[kind] = d
    return out


def flags(report: pd.DataFrame, severity: Iterable[str] = ("error", "warn")) -> pd.DataFrame:
    """Rows a consumer must act on: 'error' -> fail closed on that symbol/ex-date, 'warn' -> flag."""
    sev = set(severity)
    return report[report["severity"].isin(sev)] if len(report) else report


def write_report(report: pd.DataFrame, summary: dict, out_dir=None) -> tuple[str, str]:
    from .config import CACHE_ROOT
    d = (CACHE_ROOT / "qa") if out_dir is None else out_dir
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    p1 = d / f"ca_crosscheck_{stamp}.csv"
    p2 = d / f"ca_crosscheck_{stamp}_summary.json"
    report.to_csv(p1, index=False)
    p2.write_text(json.dumps(summary, indent=1, default=str))
    return str(p1), str(p2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    cc = CrossChecker()
    syms = [s.upper() for s in a.symbols] or cc.symbols()
    if a.limit:
        syms = syms[: a.limit]
    rep = cc.run(syms, progress=True)
    summ = summarize(rep)
    print(json.dumps(summ, indent=1, default=str))
    if not a.no_write:
        print("written:", *write_report(rep, summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

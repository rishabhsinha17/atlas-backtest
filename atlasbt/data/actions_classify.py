"""Reclassify Yahoo corporate actions (red-team #9).

Yahoo encodes spin-offs (and some mergers) as fractional "stock splits" whose ratio is the
price-adjustment factor P_before / (P_before - spun value): HPQ 2015-11-02 2.202 (HPE),
ABT 2013-01-02 2.0842 (ABBV), DRI 2015-11-10 1.1187 (FCPT), EBAY 2015-07-20 2.376 (PYPL) ...
Served as SPLIT they multiply the share count in the ledger. Rules:
  * known spin-off table (KNOWN_SPINOFFS)                       -> SPINOFF (new_symbol, ratio)
  * ratio a common split (k:1, 1:k, 3:2, 5:4, ...)              -> SPLIT
  * ratio a round stock dividend (1.02 .. 1.25 in round steps)  -> SPLIT (stock dividend)
  * other ratio > 1 (heuristic)                                 -> SPINOFF, new_symbol unknown
  * other ratio < 1 (reverse split combined with a distribution, e.g. MSI 2011 0.2474) -> kept as
    SPLIT with the Yahoo factor (value-preserving for NAV) but fidelity T3 and logged
  * cash dividend > 10 % of the previous raw close              -> SPECIAL_DIVIDEND (excluded from
    yield estimates)
SPINOFF rows carry amount = value distributed per parent share = P_prev * (1 - 1/factor)
(the value Yahoo removed), ratio = child shares per parent share when known (else NaN),
`adj_factor` = Yahoo's factor. Adjustments treat SPINOFF like a distribution of `amount`.
Every reclassification is logged in the `classify_note` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from . import frames

# parent ticker, ex_date -> spin-off terms (public deal terms; child shares per PARENT share held
# at the previous close, i.e. before any same-day reverse split). 'split' = a reverse split
# effected on the same ex-date (EXPE/ITT/DD composites). Both the Yahoo ticker and the as-traded
# DoltHub ticker of the parent are listed where they differ.
_SPIN = {
    ("HPQ", "2015-11-02"): {"children": [("HPE", 1.0)]},
    ("ABT", "2013-01-02"): {"children": [("ABBV", 1.0)]},
    ("ABT", "2004-05-03"): {"children": [("HSP", 0.1)]},
    ("DRI", "2015-11-10"): {"children": [("FCPT", 1 / 3)]},
    ("MO", "2007-03-30"): {"children": [("KFT", 0.692024)]},
    ("MO", "2008-03-31"): {"children": [("PM", 1.0)]},
    ("MDLZ", "2012-10-02"): {"children": [("KRFT", 1 / 3)]},
    ("CAH", "2009-09-01"): {"children": [("CFN", 0.5)]},
    ("DHR", "2016-07-05"): {"children": [("FTV", 0.5)]},
    ("DHR", "2023-10-02"): {"children": [("VLTO", 1 / 3)]},
    ("FIS", "2008-07-03"): {"children": [("LPS", 0.5)]},
    ("EFX", "1997-07-01"): {"children": [("CPS", 0.5)]},
    ("EBAY", "2015-07-20"): {"children": [("PYPL", 1.0)]},
    ("BAX", "2015-07-01"): {"children": [("BXLT", 1.0)]},
    ("JCI", "2012-10-01"): {"children": []},
    ("DD", "2019-04-02"): {"children": [("DOW", 1 / 3)]},
    ("DWDP", "2019-04-02"): {"children": [("DOW", 1 / 3)]},
    ("DD", "2019-06-03"): {"children": [("CTVA", 1 / 3)], "split": 1 / 3},
    ("DWDP", "2019-06-03"): {"children": [("CTVA", 1 / 3)], "split": 1 / 3},
    ("GE", "2023-01-04"): {"children": [("GEHC", 1 / 3)]},
    ("GE", "2024-04-02"): {"children": [("GEV", 0.25)]},
    ("XLF", "2016-09-19"): {"children": [("XLRE", np.nan)]},
    # Exelis (XLS, acquired 2015) is in no free price source: its first regular-way close $11.70 on
    # 2011-11-01 is quoted by the Rochester Business Journal, 2011-11-01 "Shares of ITT, new
    # spinoffs climb" (rbj.net/2011/11/01/shares-of-itt-new-spinoffs-climb), consistent with the
    # opening prices in ITT's FY2011 10-K (XLS 10.33, XYL 25.60, ITT 17.02)
    ("ITT", "2011-11-01"): {"children": [("XYL", 1.0), ("XLS", 1.0)], "split": 0.5,
                            "closes": {"XLS": 11.70}},
    ("COP", "2012-05-01"): {"children": [("PSX", 0.5)]},
    ("MMM", "2024-04-01"): {"children": [("SOLV", 0.25)]},
    ("JNJ", "2023-08-23"): {"children": [("KVUE", np.nan)]},
    ("T", "2022-04-11"): {"children": [("WBD", 0.241917)]},
    ("IBM", "2021-11-04"): {"children": [("KD", 0.2)]},
    ("PFE", "2020-11-17"): {"children": [("VTRS", 0.124079)]},
    ("MRK", "2021-06-03"): {"children": [("OGN", 0.1)]},
    ("DOV", "2018-05-09"): {"children": [("APY", 1 / 3)]},
    ("YUM", "2016-11-01"): {"children": [("YUMC", 1.0)]},
    ("TT", "2020-03-02"): {"children": [("IR", np.nan)]},
    ("RTX", "2020-04-03"): {"children": [("CARR", 1.0), ("OTIS", 0.5)]},
    ("UTX", "2020-04-03"): {"children": [("CARR", 1.0), ("OTIS", 0.5)]},
    ("EXC", "2022-02-02"): {"children": [("CEG", 1 / 3)]},
    ("HON", "2018-10-01"): {"children": [("REZI", 1 / 6)]},
    ("HON", "2018-10-29"): {"children": [("GTX", 0.1)]},
    ("MRO", "2011-07-01"): {"children": [("MPC", 0.5)]},
    ("EXPE", "2011-12-21"): {"children": [("TRIP", 0.5)], "split": 0.5},
    ("DISCA", "2014-08-07"): {"children": [("DISCK", 1.0)]},
    ("DISCK", "2014-08-07"): {"children": [("DISCK", 1.0)]},
}
# backwards-compatible view: (parent, ex) -> (first child, its ratio)
KNOWN_SPINOFFS = {k: ((v["children"][0] if v["children"] else (None, np.nan))) for k, v in _SPIN.items()}
COMMON = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7,
                   1 / 8, 1 / 10, 1 / 12, 1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 50, 3 / 2, 2 / 3, 5 / 4, 4 / 5,
                   4 / 3, 3 / 4, 5 / 2, 2 / 5, 5 / 3, 7 / 4, 9 / 4, 3 / 5])
STOCK_DIVIDENDS = np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.12, 1.15, 1.2, 1.25])
SPECIAL_DIV_FRACTION = 0.10


def _ex_close_known_at(raw_bars, ex):
    if raw_bars is not None and len(raw_bars) and "known_at" in raw_bars.columns:
        d = frames.to_naive_date(raw_bars.index)
        hit = raw_bars["known_at"][d == ex]
        if len(hit):
            return pd.Timestamp(hit.iloc[0])
    return pd.Timestamp(ex).tz_localize("America/New_York") + pd.Timedelta(hours=16)


def spin_value(spec: dict, ex, child_close) -> tuple[float, str]:
    """Value distributed per parent share = sum(child first ex-date close x ratio), or NaN."""
    if not spec.get("children"):
        return np.nan, ""
    fixed = spec.get("closes", {})
    if child_close is None and not all(c in fixed for c, _ in spec["children"]):
        return np.nan, ""
    tot, used = 0.0, []
    for c, r in spec["children"]:
        px = fixed[c] if c in fixed else child_close(c, ex)
        if not (np.isfinite(r) and np.isfinite(px)):
            return np.nan, ""
        tot += r * px
        used.append(f"{c} {px:.4g} x {r:.4g}")
    return tot, " + ".join(used)


def known_spinoff_rows(symbol: str, acts: pd.DataFrame, raw_bars: pd.DataFrame, child_close,
                       existing_types=("SPINOFF",)) -> pd.DataFrame:
    """SPINOFF (+ same-day reverse SPLIT) rows for known spin-offs of `symbol` that `acts` lacks
    (DoltHub segments carry no spin-offs: MRO 2011-07-01 MPC). Value-based amounts only."""
    sym = symbol.upper()
    rows = []
    have = set() if acts is None or not len(acts) else {
        (t, pd.Timestamp(e).normalize()) for t, e in zip(acts["type"], acts["ex_date"])}
    for (p, exs), spec in _SPIN.items():
        if p != sym:
            continue
        ex = pd.Timestamp(exs)
        if ("SPINOFF", ex) in have:
            continue
        if raw_bars is None or not len(raw_bars) or ex not in set(frames.to_naive_date(raw_bars.index)):
            continue
        D, how = spin_value(spec, ex, child_close)
        if not np.isfinite(D) or D <= 0:
            continue
        ka = _ex_close_known_at(raw_bars, ex)
        ch = spec["children"]
        rows.append({"symbol": sym, "type": "SPINOFF", "ex_date": ex, "known_at": ka,
                     "ratio": ch[0][1] if len(ch) == 1 else np.nan, "amount": D,
                     "new_symbol": "+".join(c for c, _ in ch), "fidelity": int(Fidelity.T2_PROXY),
                     "adj_factor": np.nan, "classify_note": f"known spin-off added (value: {how})"})
        if spec.get("split") and ("SPLIT", ex) not in have:
            rows.append({"symbol": sym, "type": "SPLIT", "ex_date": ex, "known_at": ka, "ratio": spec["split"],
                         "amount": 0.0, "new_symbol": None, "fidelity": int(Fidelity.T2_PROXY),
                         "adj_factor": spec["split"], "classify_note": "reverse split effected with the spin-off"})
    return pd.DataFrame(rows)


def classify(symbol: str, acts: pd.DataFrame, raw_bars: pd.DataFrame | None, child_close=None) -> pd.DataFrame:
    """child_close(sym, ex_date) -> first close of a spun-off child on/after the ex-date (used for
    value-based spin-off amounts: D = sum(child close x ratio); the price-factor method inflates
    large distributions, e.g. HPQ/HPE)."""
    if acts is None or not len(acts):
        return acts
    a = acts.copy()
    a["adj_factor"] = np.where(a["type"] == "SPLIT", a["ratio"], np.nan)
    a["classify_note"] = None
    closes = None
    if raw_bars is not None and len(raw_bars):
        closes = pd.Series(raw_bars["close"].to_numpy(), index=frames.to_naive_date(raw_bars.index))

    def prev_close(ex):
        if closes is None:
            return np.nan
        p = closes[closes.index < pd.Timestamp(ex).normalize()]
        return float(p.iloc[-1]) if len(p) else np.nan

    def is_common(f):
        return np.abs(np.log(f) - np.log(COMMON)).min() < 0.004

    sym = symbol.upper()
    exn = pd.to_datetime(a["ex_date"]).dt.normalize()
    add, drop = [], set()
    # 1) known spin-offs: one SPINOFF row per event (Yahoo encodes them as a fractional split, a
    #    'dividend', or BOTH -- DHR 2016, ABT 2004 -- which must not be applied twice), plus the
    #    same-day reverse split when there is one (EXPE 2011, ITT 2011, DD 2019)
    for (p, exs), spec in _SPIN.items():
        if p != sym:
            continue
        ex = pd.Timestamp(exs)
        on = a[exn == ex]
        if not len(on):
            continue
        pc = prev_close(ex)
        r_s = spec.get("split")
        spl = on[on["type"] == "SPLIT"]
        nonc = spl[[not is_common(float(f)) for f in spl["ratio"]]] if len(spl) else spl
        divs = on[on["type"].isin(["CASH_DIVIDEND", "SPECIAL_DIVIDEND"]) &
                  (on["amount"] > (0.03 * pc if np.isfinite(pc) else 0))]
        D, how = spin_value(spec, ex, child_close)
        ka = on["known_at"].iloc[0]
        if np.isfinite(D) and D > 0:
            ka = _ex_close_known_at(raw_bars, ex)          # value known at the child's first close
        elif len(divs):
            D, how = float(divs["amount"].max()), "Yahoo dividend encoding"
        elif len(nonc) and np.isfinite(pc) and raw_bars is not None and len(raw_bars) and \
                ex in set(frames.to_naive_date(raw_bars.index)):
            # a child price is missing (ITT 2011: Exelis): value implied by the ex-date OPEN,
            # D = P_prev - open_ex x split (known at the open)
            dd = frames.to_naive_date(raw_bars.index)
            o = float(raw_bars["open"].to_numpy()[np.argmax(dd == ex)])
            D, how = pc - o * (r_s or 1.0), f"ex-date open {o:.4g}" + (f" x split {r_s:g}" if r_s else "") + \
                " (a child price is unavailable)"
            ka = pd.Timestamp(ex).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
        elif len(nonc) and np.isfinite(pc):
            f = float(nonc["ratio"].iloc[0])
            D, how = pc * (1 - (r_s or 1.0) / f), f"Yahoo price factor {f:g}" + (f" / split {r_s:g}" if r_s else "")
        else:
            D = np.nan
        drop |= set(divs.index) | set(nonc.index) | (set(spl.index) if r_s else set())
        ch = spec["children"]
        if np.isfinite(D) and D > 0:
            add.append({"symbol": sym, "type": "SPINOFF", "ex_date": ex, "known_at": ka,
                        "ratio": ch[0][1] if len(ch) == 1 else np.nan, "amount": D,
                        "new_symbol": "+".join(c for c, _ in ch) or None, "fidelity": int(Fidelity.T2_PROXY),
                        "adj_factor": (pc / (pc - D)) if np.isfinite(pc) and pc > D else np.nan,
                        "classify_note": f"known spin-off; amount per parent share = {how}"})
        if r_s:
            add.append({"symbol": sym, "type": "SPLIT", "ex_date": ex, "known_at": on["known_at"].iloc[0],
                        "ratio": r_s, "amount": 0.0, "new_symbol": None, "fidelity": int(Fidelity.T2_PROXY),
                        "adj_factor": r_s, "classify_note": "reverse split effected with the spin-off"})
    # 2) unknown events: duplicate encodings (a large 'dividend' and a fractional split on the same
    #    day carrying the same value) collapse to one SPINOFF
    for ex in sorted(set(exn[~a.index.isin(drop)])):
        on = a[(exn == ex) & ~a.index.isin(drop)]
        spl = on[(on["type"] == "SPLIT")]
        spl = spl[[not is_common(float(f)) and float(f) > 1 for f in spl["ratio"]]] if len(spl) else spl
        divs = on[on["type"].isin(["CASH_DIVIDEND", "SPECIAL_DIVIDEND"]) & (on["amount"] > 0)]
        pc = prev_close(ex)
        if len(spl) and len(divs) and np.isfinite(pc):
            f = float(spl["ratio"].iloc[0])
            Dd = float(divs["amount"].max())
            if pc > Dd and abs(np.log(f) - np.log(pc / (pc - Dd))) < 0.1:
                drop |= set(spl.index) | set(divs.index)
                add.append({"symbol": sym, "type": "SPINOFF", "ex_date": ex, "known_at": on["known_at"].iloc[0],
                            "ratio": np.nan, "amount": Dd, "new_symbol": None, "fidelity": int(Fidelity.T2_PROXY),
                            "adj_factor": f, "classify_note": f"dividend {Dd:.4g} and split factor {f:g} encode the "
                                                              f"same distribution: one SPINOFF"})
    a = a.drop(index=list(drop))
    for i, r in a.iterrows():
        ex = pd.Timestamp(r["ex_date"]).normalize()
        if r["type"] == "SPLIT":
            f = float(r["ratio"])
            common = is_common(f)
            stockdiv = np.abs(f - STOCK_DIVIDENDS).min() < 0.0015
            if not common and not stockdiv and f > 1:
                pc = prev_close(ex)
                a.at[i, "type"] = "SPINOFF"
                a.at[i, "new_symbol"] = None
                a.at[i, "ratio"] = np.nan
                a.at[i, "amount"] = pc * (1 - 1 / f) if np.isfinite(pc) else 0.0
                a.at[i, "fidelity"] = int(Fidelity.T2_PROXY)
                a.at[i, "classify_note"] = f"heuristic: non-standard Yahoo split ratio (yahoo factor {f:g})"
            elif not common and not stockdiv:
                a.at[i, "fidelity"] = int(Fidelity.T3_MODELED)
                a.at[i, "classify_note"] = f"composite reverse split + distribution? yahoo factor {f:g} kept as SPLIT"
        elif r["type"] == "CASH_DIVIDEND" and r["amount"] > 0:
            pc = prev_close(ex)
            if np.isfinite(pc) and r["amount"] > SPECIAL_DIV_FRACTION * pc:
                a.at[i, "type"] = "SPECIAL_DIVIDEND"
                a.at[i, "classify_note"] = f"dividend {r['amount']:.4g} > 10% of prev close {pc:.4g}"
    if add:
        a = pd.concat([a, pd.DataFrame(add)], ignore_index=True)
    # SPINOFF before a same-day SPLIT: the amount is per share held at the previous close
    order = a["type"].map({"SPINOFF": 0}).fillna(1)
    a = a.assign(_o=order, _d=pd.to_datetime(a["ex_date"])).sort_values(["_d", "_o"], kind="mergesort")
    return a.drop(columns=["_o", "_d"]).reset_index(drop=True)

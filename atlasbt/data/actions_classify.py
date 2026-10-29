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

# parent, ex_date -> (child symbol, child shares per parent share). Public deal terms.
KNOWN_SPINOFFS = {
    ("HPQ", "2015-11-02"): ("HPE", 1.0),
    ("ABT", "2013-01-02"): ("ABBV", 1.0),
    ("DRI", "2015-11-10"): ("FCPT", 1 / 3),
    ("MO", "2007-03-30"): ("KFT", 0.692024),
    ("MO", "2008-03-31"): ("PM", 1.0),
    ("MDLZ", "2012-10-02"): ("KRFT", 1 / 3),
    ("CAH", "2009-09-01"): ("CFN", 0.5),
    ("DHR", "2016-07-05"): ("FTV", 0.5),
    ("DHR", "2023-10-02"): ("VLTO", 1 / 3),
    ("FIS", "2008-07-03"): ("LPS", 0.5),
    ("EFX", "1997-07-01"): ("CPS", 0.5),
    ("EBAY", "2015-07-20"): ("PYPL", 1.0),
    ("BAX", "2015-07-01"): ("BXLT", 1.0),
    ("JCI", "2012-10-01"): ("JCI", np.nan),
    ("DD", "2019-04-02"): ("DOW", 1 / 3),
    ("DD", "2019-06-03"): ("CTVA", 1 / 3),
    ("GE", "2023-01-04"): ("GEHC", 1 / 3),
    ("GE", "2024-04-02"): ("GEV", 0.25),
    ("XLF", "2016-09-19"): ("XLRE", np.nan),
    ("ITT", "2011-11-01"): ("XYL", 1.0),
    ("COP", "2012-05-01"): ("PSX", 0.5),
    ("MMM", "2024-04-01"): ("SOLV", 0.25),
    ("JNJ", "2023-08-23"): ("KVUE", np.nan),
    ("T", "2022-04-11"): ("WBD", 0.241917),
    ("IBM", "2021-11-04"): ("KD", 0.2),
    ("PFE", "2020-11-17"): ("VTRS", 0.124079),
    ("MRK", "2021-06-03"): ("OGN", 0.1),
    ("DOV", "2018-05-09"): ("APY", 1 / 3),
    ("YUM", "2016-11-01"): ("YUMC", 1.0),
    ("TT", "2020-03-02"): ("IR", np.nan),
    ("RTX", "2020-04-03"): ("CARR", 1.0),
    ("EXC", "2022-02-02"): ("CEG", 1 / 3),
    ("HON", "2018-10-01"): ("REZI", 1 / 6),
    ("HON", "2018-10-29"): ("GTX", 0.1),
}
COMMON = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7,
                   1 / 8, 1 / 10, 1 / 12, 1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 50, 3 / 2, 2 / 3, 5 / 4, 4 / 5,
                   4 / 3, 3 / 4, 5 / 2, 2 / 5, 5 / 3, 7 / 4, 9 / 4, 3 / 5])
STOCK_DIVIDENDS = np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.12, 1.15, 1.2, 1.25])
SPECIAL_DIV_FRACTION = 0.10


def classify(symbol: str, acts: pd.DataFrame, raw_bars: pd.DataFrame | None) -> pd.DataFrame:
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

    sym = symbol.upper()
    for i, r in a.iterrows():
        ex = pd.Timestamp(r["ex_date"]).normalize()
        if r["type"] == "SPLIT":
            f = float(r["ratio"])
            key = (sym, str(ex.date()))
            common = np.abs(np.log(f) - np.log(COMMON)).min() < 0.004
            stockdiv = np.abs(f - STOCK_DIVIDENDS).min() < 0.0015
            if key in KNOWN_SPINOFFS or (not common and not stockdiv and f > 1):
                child, cr = KNOWN_SPINOFFS.get(key, (None, np.nan))
                pc = prev_close(ex)
                a.at[i, "type"] = "SPINOFF"
                a.at[i, "new_symbol"] = child
                a.at[i, "ratio"] = cr
                a.at[i, "amount"] = pc * (1 - 1 / f) if np.isfinite(pc) else 0.0
                a.at[i, "fidelity"] = int(Fidelity.T2_PROXY)
                a.at[i, "classify_note"] = ("known spin-off" if key in KNOWN_SPINOFFS else
                                            "heuristic: non-standard Yahoo split ratio") + f" (yahoo factor {f:g})"
            elif not common and not stockdiv:
                a.at[i, "fidelity"] = int(Fidelity.T3_MODELED)
                a.at[i, "classify_note"] = f"composite reverse split + distribution? yahoo factor {f:g} kept as SPLIT"
        elif r["type"] == "CASH_DIVIDEND" and r["amount"] > 0:
            pc = prev_close(ex)
            if np.isfinite(pc) and r["amount"] > SPECIAL_DIV_FRACTION * pc:
                a.at[i, "type"] = "SPECIAL_DIVIDEND"
                a.at[i, "classify_note"] = f"dividend {r['amount']:.4g} > 10% of prev close {pc:.4g}"
    return a

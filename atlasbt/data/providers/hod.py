"""historicaloptiondata.com free sample (Delta Neutral 'L3' EOD, one day: 2019-08-15), on disk
under data_raw/historicaloptiondata_free_samples/Sample_L3_20190815.zip (scout download).

Every US listed option that day (incl. SPX/SPXW, QQQ, IWM -- absent from DoltHub) with bid,
ask, last, volume, open interest, IV and greeks. Used as (a) a one-day full-chain reference
and (b) a cross-check of DoltHub's sparse chain on the same date. ts := 2019-08-15 close
(16:00 ET; vendor EOD snapshot, exact capture time unknown) -> T2_PROXY.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import config
from ..store import ParquetStore, default_store
from .dolthub import snapshot_ts
from .dolthub_bulk import occ_symbols

SOURCE = "hod"
ZIP = config.DATA_RAW / "historicaloptiondata_free_samples" / "Sample_L3_20190815.zip"
CASH_INDEX = {"SPX", "SPXW", "XSP", "NDX", "RUT", "VIX", "DJX", "OEX", "XEO"}


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    ts = snapshot_ts(pd.to_datetime(df["DataDate"].iloc[0]))
    und = df["UnderlyingSymbol"].astype(str).str.upper().replace({"SPXW": "SPX"})   # SPXW series belong to the SPX chain
    right = np.where(df["Type"].str.lower().str.startswith("c"), "C", "P")
    exp = pd.to_datetime(df["Expiration"], format="%m/%d/%Y")
    strike = df["Strike"].astype(float)
    root = df["OptionSymbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
    out = pd.DataFrame({"ts": ts, "underlying": und, "expiry": exp, "strike": strike, "right": right,
                        "symbol": occ_symbols(root, exp, pd.Series(right, index=df.index), strike),
                        "bid": df["Bid"].astype(float), "ask": df["Ask"].astype(float)})
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["iv"] = df["IVMean"].astype(float)
    for a, b in (("delta", "Delta"), ("gamma", "Gamma"), ("vega", "Vega"), ("theta", "Theta")):
        out[a] = df[b].astype(float)
    out["underlying_price"] = df["UnderlyingPrice"].astype(float)
    out["underlying_price_source"] = "vendor"
    out["volume"] = df["Volume"].astype(float)
    out["open_interest"] = df["OpenInterest"].astype(float)
    # vendor files label SPX weeklies with root 'SPX': a non-third-Friday SPX expiry is an
    # SPXW (PM-settled) series
    third_fri = (exp.dt.weekday == 4) & (exp.dt.day.between(15, 21))
    root = root.where(~((root == "SPX") & ~third_fri), "SPXW")
    cash = root.isin(CASH_INDEX)
    out["exercise"] = np.where(cash, "EUROPEAN", "AMERICAN")
    out["settlement"] = np.where(root == "SPXW", "CASH_PM", np.where(cash, "CASH_AM", "PHYSICAL"))
    out["multiplier"] = 100.0
    out["fidelity"] = int(Fidelity.T2_PROXY)
    out["root"] = root
    return out


def load(symbols: Optional[Iterable[str]] = None, store: Optional[ParquetStore] = None) -> dict:
    store = store or default_store()
    zf = zipfile.ZipFile(ZIP)
    name = [n for n in zf.namelist() if re.match(r"(?i)L3_options_\d{8}\.csv$", n)][0]
    df = pd.read_csv(io.BytesIO(zf.read(name)))
    df.columns = [c.strip() for c in df.columns]
    if symbols:
        df = df[df["UnderlyingSymbol"].str.upper().isin({s.upper() for s in symbols})]
    c = canonical(df)
    for u, g in c.groupby("underlying"):
        store.write("option_chain", SOURCE, u, g.reset_index(drop=True), partition="year", time_col="ts",
                    key=["ts", "symbol"], fidelity=int(Fidelity.T2_PROXY), replace=True,
                    extra={"origin": "data_raw/historicaloptiondata_free_samples/Sample_L3_20190815.zip"})
    return {"rows": len(c), "underlyings": c["underlying"].nunique()}

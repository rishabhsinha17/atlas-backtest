"""Kenneth French data library (Dartmouth). For VALIDATION only (factor regressions,
sanity checks of strategy returns) -- not point-in-time: the library is rebuilt
each year with revised CRSP/Compustat data. Returns stored as DECIMALS (files
are in percent). Daily rows: known_at = that session's close (but note the
revision caveat; fidelity T2_PROXY)."""
from __future__ import annotations

import io
import logging
import re
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..http import client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.french")
SOURCE = "french"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"

DATASETS = {
    "F-F_Research_Data_Factors_daily": "Mkt-RF, SMB, HML, RF (daily)",
    "F-F_Research_Data_Factors": "monthly 3 factors",
    "F-F_Research_Data_5_Factors_2x3_daily": "5 factors daily",
    "F-F_Momentum_Factor_daily": "UMD daily",
    "F-F_Momentum_Factor": "UMD monthly",
    "F-F_ST_Reversal_Factor_daily": "short-term reversal daily",
    "F-F_LT_Reversal_Factor_daily": "long-term reversal daily",
    "10_Industry_Portfolios_daily": "10 industries VW daily",
    "12_Industry_Portfolios_daily": "12 industries VW daily",
    "49_Industry_Portfolios_daily": "49 industries VW daily",
    "10_Portfolios_Prior_12_2_Daily": "momentum deciles daily",
}


def parse_zip(content: bytes) -> pd.DataFrame:
    """First table of the CSV inside the zip (value-weighted block for portfolios)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    text = zf.read(name).decode("latin-1")
    lines = text.splitlines()
    # header = first line that starts with ',' followed by names and is followed by a date row
    hdr_i = None
    for i, ln in enumerate(lines[:-1]):
        if ln.strip().startswith(",") and re.match(r"^\s*\d{6,8}\s*,", lines[i + 1]):
            hdr_i = i
            break
    if hdr_i is None:
        raise ValueError("cannot locate header in French file")
    cols = [c.strip() for c in lines[hdr_i].split(",")[1:]]
    rows = []
    for ln in lines[hdr_i + 1:]:
        if not re.match(r"^\s*\d{6,8}\s*,", ln):
            break
        parts = [p.strip() for p in ln.split(",")]
        rows.append(parts)
    df = pd.DataFrame([r[1:] for r in rows], columns=cols)
    keys = [r[0] for r in rows]
    fmt = "%Y%m%d" if len(keys[0]) == 8 else "%Y%m"
    idx = pd.to_datetime(keys, format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    df.index = pd.DatetimeIndex(idx, name="date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    return df


def fetch_dataset(name: str, store: Optional[ParquetStore] = None, ttl_days: float = 7) -> dict:
    store = store or default_store()
    content = client().get(BASE.format(name=name), ttl=ttl_days * 86400)
    df = parse_zip(content)
    sch = frames.schedule()
    df["known_at"] = sch["close"].reindex(df.index).to_numpy() if "daily" in name.lower() else pd.NaT
    df.columns = [str(c) for c in df.columns]
    store.write("factors", SOURCE, name, df, fidelity=int(Fidelity.T2_PROXY), replace=True,
                extra={"units": "decimal returns", "pit": False,
                       "first_ts": str(df.index[0].date()), "last_ts": str(df.index[-1].date())})
    return {"symbol": name, "rows": len(df), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
            "columns": len(df.columns) - 1}

"""Real SIP NBBO sample 2016-2026 from Alpaca for validating the calibrated spread model (docs/SPREADS.md §10).

    ALPACA_API_KEY=... ALPACA_API_SECRET=... ATLASBT_CACHE=... \
    python atlasbt/execution/calibration/tools/alpaca_nbbo_sample.py [--work DIR] [--days N] [--assemble]

Design (all point-in-time):
* days: the 2nd Wednesday of every month 2016-01 .. last complete month (next session if not a session);
  processed quarterly months first so partial runs are balanced over eras;
* symbols: the 19 intraday ETFs (atlasbt.data.fetch.ALPACA_ETFS) + 50 PIT S&P 500 members spread evenly
  over the members' 20-day ADV$ ranks that day (identity-routed to their tape ticker, alpaca.route_tapes);
* windows: seven 1-minute windows at the middle of the model's 5-minute time-of-day buckets
  (open+2, +17, +62, +182, close-88, -28, -8 min = 09:32, 09:47, 10:32, 12:32, 14:32, 15:32, 15:52 ET);
* statistic: time-weighted mean relative half spread (ask - bid) / (2 mid) of the cleaned NBBO
  (alpaca.clean_nbbo: crossed / locked / non-firm updates dropped) over the part of the minute after the
  first update, plus the median and the number of updates;
* features (sessions strictly before the day, as in the calibration): raw prior close, mean close x volume
  over 20 sessions, std of daily log returns over 20 sessions (|r| > 40% dropped).
Output: one parquet per day in --work, assembled (--assemble) to ../alpaca_nbbo_2016_2026.csv.gz.
Only derived statistics are shipped; no raw quotes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("alpaca_nbbo_sample")
OUT = Path(__file__).resolve().parents[1] / "alpaca_nbbo_2016_2026.csv.gz"
#: (anchor, minutes) -> 1-minute window start; middle of the 5-minute ToD buckets
WINDOWS = [("open", 2), ("open", 17), ("open", 62), ("open", 182), ("close", -88), ("close", -28), ("close", -8)]
N_STOCKS = 50


def sample_days() -> list[pd.Timestamp]:
    from atlasbt.data import frames
    sch = frames.schedule()
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for m in pd.period_range("2016-01", (now - pd.Timedelta(days=1)).strftime("%Y-%m"), freq="M"):
        d = pd.Timestamp(m.start_time)
        weds = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-WED")
        if len(weds) < 2:
            continue
        cand = sch.index[(sch.index >= weds[1]) & (sch["close"] < now)]
        if len(cand):
            out.append(cand[0])
    # quarterly months first, then the others (balanced partial runs)
    return sorted(out, key=lambda d: (d.month % 3 != 1, d))


def features(portal, symbol: str, day: pd.Timestamp) -> dict:
    f = portal._daily_frame(symbol, day - pd.Timedelta(days=1), start=day - pd.Timedelta(days=60))
    if len(f) < 2:
        return {}
    c = f["close"].to_numpy(float)[-21:]
    v = f["volume"].to_numpy(float)[-21:]
    r = np.diff(np.log(c))
    r = r[np.isfinite(r) & (np.abs(r) <= 0.40)]
    return {"prev_close": float(c[-1]), "adv_dollar": float(np.mean(c[-20:] * v[-20:])),
            "sigma": float(np.std(r, ddof=1)) if len(r) >= 2 else np.nan, "n_sigma": int(len(r))}


def tw_stats(q: pd.DataFrame, w0: pd.Timestamp, w1: pd.Timestamp) -> dict:
    """Time-weighted mean / median relative half spread of a cleaned NBBO frame over [first update, w1)."""
    q = q[(q.index >= w0) & (q.index < w1)]
    if not len(q):
        return {"n_quotes": 0}
    mid = (q["ask"] + q["bid"]) / 2.0
    hs = ((q["ask"] - q["bid"]) / (2.0 * mid)).to_numpy() * 1e4
    t = q.index.asi8
    dt = np.diff(np.r_[t, w1.value]).astype(float)
    ok = np.isfinite(hs) & (hs < 5000)
    hs, dt = hs[ok], dt[ok]
    if dt.sum() <= 0:
        return {"n_quotes": int(len(q))}
    o = np.argsort(hs)
    cw = np.cumsum(dt[o]) / dt.sum()
    return {"tw_hs_bp": float(np.sum(hs * dt) / dt.sum()), "tw_med_hs_bp": float(hs[o][np.searchsorted(cw, 0.5)]),
            "n_quotes": int(len(q)), "cover_s": float(dt.sum() / 1e9), "mid": float(mid.iloc[-1])}


def run_day(portal, day: pd.Timestamp, work: Path, cli) -> int:
    from atlasbt.data import fetch as F
    from atlasbt.data import frames
    from atlasbt.data.providers import alpaca as A
    out_p = work / f"{day.date()}.parquet"
    if out_p.exists():
        return 0
    sch = frames.schedule()
    o, c = (pd.Timestamp(x) for x in sch.loc[day, ["open", "close"]])
    a = o + pd.Timedelta(minutes=30)
    members = portal.universe("SP500", a)
    feats = {}
    for s in members:
        ft = features(portal, s, day)
        if ft and ft["adv_dollar"] > 0:
            feats[s] = ft
    ranked = sorted(feats, key=lambda s: feats[s]["adv_dollar"])
    pick = [ranked[int(round(i))] for i in np.linspace(0, len(ranked) - 1, N_STOCKS)] if ranked else []
    etfs = [e for e in F.ALPACA_ETFS]
    for e in etfs:
        ft = features(portal, e, day)
        if ft:
            feats[e] = ft
    syms = list(dict.fromkeys(pick + [e for e in etfs if e in feats]))
    routes, _ = A.route_tapes(portal, {s: pd.DatetimeIndex([day]) for s in syms}, cli=cli)
    tape = dict(zip(routes["symbol"], routes["tape"]))
    rows = []
    wins = [((o if anc == "open" else c) + pd.Timedelta(minutes=m)) for anc, m in WINDOWS]
    tapes = sorted(set(tape.values()))

    def one(w0):
        w1 = w0 + pd.Timedelta(minutes=1)
        return w0, w1, A.fetch_quotes_raw(tapes, w0, w1 - pd.Timedelta(microseconds=1), cli=cli)

    with ThreadPoolExecutor(max_workers=3) as ex:
        for w0, w1, got in ex.map(one, [w for w in wins if w < c]):
            cleaned = {tk: A.clean_nbbo(q) for tk, q in got.items()}
            for s in syms:
                tk = tape.get(s)
                if tk is None:
                    continue
                cq, st = cleaned.get(tk, (pd.DataFrame(), {"raw": 0}))
                rec = {"date": day, "w0": w0, "mso": (w0 - o).total_seconds() / 60.0,
                       "mtc": (c - w0).total_seconds() / 60.0, "symbol": s, "tape": tk,
                       "etf": int(s in etfs), **feats[s], **(tw_stats(cq, w0, w1) if len(cq) else {"n_quotes": 0}),
                       "raw_updates": st.get("raw", 0), "crossed": st.get("crossed", 0), "locked": st.get("locked", 0)}
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(out_p)
    return len(df)


def assemble(work: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(work.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["w0"] = pd.to_datetime(df["w0"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["date", "w0", "mso", "mtc", "symbol", "tape", "etf", "prev_close", "adv_dollar", "sigma", "n_sigma",
            "tw_hs_bp", "tw_med_hs_bp", "n_quotes", "cover_s", "raw_updates", "crossed", "locked"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT, index=False, float_format="%.6g", compression="gzip")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--days", type=int, default=0, help="stop after N new days (0 = all)")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args(argv)
    from atlasbt.data import config
    work = Path(a.work or (config.CACHE_ROOT / "calibration" / "alpaca_nbbo"))
    work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.assemble:
        df = assemble(work)
        print(json.dumps({"rows": len(df), "days": int(df["date"].nunique()), "out": str(OUT)}))
        return
    from atlasbt.data.portal import Portal
    from atlasbt.data.providers import alpaca as A
    cli = A.client()
    p = Portal()
    n = 0
    for i, d in enumerate(sample_days()):
        try:
            k = run_day(p, d, work, cli)
        except Exception as e:  # keep going; the day is retried next run
            log.error("%s failed: %s", d.date(), e)
            continue
        if k:
            n += 1
            log.info("%s: %d rows (requests %d)", d.date(), k, cli.requests)
        if i % 20 == 19:
            p.clear_cache()
        if a.days and n >= a.days:
            break


if __name__ == "__main__":
    sys.exit(main())

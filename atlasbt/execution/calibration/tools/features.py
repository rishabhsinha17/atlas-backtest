import os
"""PIT daily features for calibration day D from daily bars of sessions < D."""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/atlas_bt')
from atlasbt.execution.spread import corwin_schultz_rolling, abdi_ranaldo_rolling
S = os.environ.get('SPREAD_WORK', '/tmp/spread_work')
_bars = None
def bars():
    global _bars
    if _bars is None:
        a = pd.read_parquet(f'{S}/dl/ohlcv/ohlcv_2016plus.parquet')
        fs = sorted(glob.glob(f'{S}/dl/dolt_2026/*.csv'))
        import os
        b = [pd.read_csv(f, dtype={'act_symbol': str}) for f in fs if os.path.getsize(f) > 5]
        b = [x for x in b if len(x)]
        if b:
            b = pd.concat(b); b['date'] = pd.to_datetime(b['date'])
            a = pd.concat([a[a.date < b.date.min()], b])
        a = a.rename(columns={'act_symbol': 'symbol'})
        _bars = a.sort_values(['symbol', 'date']).reset_index(drop=True)
    return _bars
def features(D, window=20, cs_window=21):
    D = pd.Timestamp(D)
    a = bars()
    dates = np.sort(a.date[a.date < D].unique())[-(cs_window + 3):]
    w = a[a.date.isin(dates)].copy()
    out = []
    for sym, g in w.groupby('symbol', sort=False):
        if len(g) < 15 or g.date.iloc[-1] != dates[-1]:
            continue
        c = g.close.to_numpy(float); h = g.high.to_numpy(float); l = g.low.to_numpy(float); v = g.volume.to_numpy(float)
        if not (c[-1] > 0): continue
        dv = (c * v)[-window:]
        r = np.diff(np.log(np.where(c > 0, c, np.nan)))[-window:]
        r = r[np.isfinite(r) & (np.abs(r) < 0.4)]
        if len(r) < 10: continue
        ok = (h > 0) & (l > 0) & (h >= l)
        cs = corwin_schultz_rolling(np.where(ok, h, np.nan), np.where(ok, l, np.nan), c, cs_window, 5)[-1]
        ar = abdi_ranaldo_rolling(np.where(ok, h, np.nan), np.where(ok, l, np.nan), c, cs_window, 5)[-1]
        out.append((sym, c[-1], np.nanmean(dv), np.std(r, ddof=1), cs, ar, len(g)))
    f = pd.DataFrame(out, columns=['symbol', 'prev_close', 'adv_dollar', 'sigma', 'cs_full', 'ar_full', 'n_obs'])
    f['date'] = D
    return f
if __name__ == '__main__':
    for D in sys.argv[1:]:
        f = features(D); f.to_parquet(f'{S}/out/feat_{D}.parquet'); print(D, len(f))

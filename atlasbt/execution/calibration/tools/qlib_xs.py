"""Cross-section of PIT features (prior 20 sessions) from qlib (Yahoo daily) at mid-year dates 2000-2010."""
import os, numpy as np, pandas as pd
Q = os.environ.get('SPREAD_WORK', '/tmp/spread_work') + '/dl/qlib'
cal = pd.to_datetime(open(f'{Q}/calendars/day.txt').read().split())
nt = pd.read_parquet('/home/user/atlas_bt/cache/store/reference/nasdaqtrader/DIRECTORY.parquet')
etfs = set(nt.symbol[nt.etf.astype(str).str.upper().isin(['Y', 'TRUE', '1'])].str.lower())
dates = [cal[cal.searchsorted(pd.Timestamp(f'{y}-07-01'))] for y in range(2000, 2011)] + [cal[cal.searchsorted(pd.Timestamp(f'{y}-01-15'))] for y in range(2001, 2011)]
def load(s, f):
    a = np.fromfile(f'{Q}/features/{s}/{f}.day.bin', dtype='<f4'); i = int(a[0])
    return a[1:].astype(float), i
rows = []
for s in os.listdir(f'{Q}/features'):
    if s in etfs or not s.isalpha(): continue
    try:
        c, i0 = load(s, 'close'); fa, _ = load(s, 'factor'); v, _ = load(s, 'volume')
    except Exception:
        continue
    n = len(c)
    for D in dates:
        j = cal.searchsorted(D) - i0      # index of session D in this series
        if j < 22 or j > n: continue
        w = slice(j - 21, j)
        cc = c[w]; ff = fa[w]; vv = v[w]
        if not np.all(np.isfinite(cc)) or np.any(cc <= 0) or np.any(ff <= 0): continue
        raw = cc / ff; vol = vv * ff
        r = np.diff(np.log(cc)); r = r[np.abs(r) < 0.4]
        if len(r) < 10: continue
        adv = np.mean((raw * vol)[-20:])
        if not adv > 0: continue
        rows.append((s, D, raw[-1], adv, np.std(r, ddof=1)))
df = pd.DataFrame(rows, columns=['symbol', 'date', 'prev_close', 'adv_dollar', 'sigma'])
df.to_parquet(os.environ.get('SPREAD_WORK', '/tmp/spread_work') + '/out/qlib_xs.parquet')
print(df.groupby('date').agg(n=('symbol', 'size'), adv_med=('adv_dollar', 'median'), px_med=('prev_close', 'median'), sig_med=('sigma', 'median')))

import os
"""Features (prior 20 sessions) for large NYSE names around decimalisation (qlib, Yahoo-derived)."""
import numpy as np, pandas as pd
Q = os.environ.get('SPREAD_WORK', '/tmp/spread_work') + '/dl/qlib'
cal = pd.to_datetime(open(f'{Q}/calendars/day.txt').read().split())
def load(s, f):
    a = np.fromfile(f'{Q}/features/{s}/{f}.day.bin', dtype='<f4'); i = int(a[0])
    return pd.Series(a[1:], index=cal[i:i + len(a) - 1]).astype(float)
rows = []
for s in ['ge', 'lu', 'pfe', 'emc', 'c', 'xom', 'mrk', 'ibm', 't', 'ko', 'wmt']:
    c = load(s, 'close'); fa = load(s, 'factor'); v = load(s, 'volume')
    raw = c / fa; vol = v * fa
    for D in ['2001-01-16', '2001-02-05']:
        if c.index[0] > pd.Timestamp("2000-12-01"): break
        w = c[c.index < D].iloc[-21:]
        r = np.log(w).diff().dropna()
        rr = raw[raw.index < D].iloc[-20:]; vv = vol[vol.index < D].iloc[-20:]
        rows.append((s.upper(), D, float(rr.iloc[-1]), float((rr * vv).mean()), float(r.std())))
df = pd.DataFrame(rows, columns=['symbol', 'date', 'price', 'adv_dollar', 'sigma'])
df.to_csv(os.environ.get('SPREAD_WORK', '/tmp/spread_work') + '/out/furfine_names.csv', index=False)
print(df.round(4).to_string())

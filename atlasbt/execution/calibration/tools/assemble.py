import os
import sys, pandas as pd, numpy as np
S = os.environ.get('SPREAD_WORK', '/tmp/spread_work')
CAL = '/home/user/atlas_bt/atlasbt/execution/calibration'
ref = pd.read_csv(f'{S}/out/ref_master_20260701.txt', sep='|', dtype=str, encoding='latin-1')
ref = ref[ref.Symbol != 'END']
ref['symbol'] = ref.Symbol.str.replace(' ', '.', regex=False)
types = dict(zip(ref.symbol, ref.Security_Type))
nt = pd.read_parquet('/home/user/atlas_bt/cache/store/reference/nasdaqtrader/DIRECTORY.parquet')
etfs = set(nt.symbol[nt.etf.astype(str).str.upper().isin(['Y', 'TRUE', '1'])])
def sectype(s):
    t = types.get(s)
    if t is not None:
        return t
    return 'ETF' if s in etfs else ('A' if s.replace('.', '').isalpha() else 'X')
COLS = ['symbol', 'sec_type', 'prev_close', 'adv_dollar', 'sigma', 'cs_full', 'ar_full', 'n_obs', 'rth_valid_s',
        'n_updates', 'tw_hs_bp', 'tw_med_hs_bp', 'tw_dollar_spread', 'tw_mid', 'tr_n', 'tr_dollar_vol',
        'eff_hs_bp_dvw', 'lit_eff_hs_bp_dvw']
for kind, d in [x.split(':') for x in sys.argv[1:]]:
    q = pd.read_csv(f'{S}/out/{"taq" if kind == "nbbo" else "iex"}_{d}.csv', dtype={'symbol': str}, keep_default_na=False, na_values=['nan', ''])
    q['symbol'] = q.symbol.str.replace(' ', '.', regex=False)
    f = pd.read_parquet(f'{S}/out/feat_{d}.parquet')
    m = q.merge(f, on='symbol', how='inner')
    m['sec_type'] = m.symbol.map(sectype)
    cols = COLS + ([f'b{k:02d}' for k in range(78)] if kind == 'nbbo' else [])
    m[cols].to_csv(f'{CAL}/{kind}_{d}.csv.gz', index=False, float_format='%.6g')
    print(kind, d, len(q), len(f), len(m), m.sec_type.value_counts().head(6).to_dict())

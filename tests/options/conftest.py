import numpy as np
import pandas as pd
import pytest

from atlasbt.pricing import bsm_greeks


def synthetic_chain(spot=100.0, session="2024-03-04", expiries=(("2024-03-15", "PHYSICAL"), ("2024-04-19", "PHYSICAL"),
                    ("2024-05-17", "PHYSICAL")), strikes=None, vol=0.2, skew=-0.4, r=0.04, half_spread=0.03,
                    underlying="XYZ"):
    """synthetic_* fixture (mechanics only): European BSM chain with a linear skew."""
    strikes = np.arange(70, 131, 1.0) if strikes is None else np.asarray(strikes, float)
    sess = pd.Timestamp(session)
    rows = []
    for e, st in expiries:
        e = pd.Timestamp(e)
        T = ((e - sess).days + 0.0) / 365.0
        for K in strikes:
            v = max(vol + skew * np.log(K / spot), 0.05)
            for cp in ("C", "P"):
                g = bsm_greeks(spot, K, T, r, 0.0, v, cp)
                m = float(g.price)
                rows.append(dict(ts=pd.Timestamp(session + " 21:00", tz="UTC"), underlying=underlying,
                                 expiry=e, strike=K, right=cp, bid=round(max(m - half_spread, 0.0), 2),
                                 ask=round(m + half_spread, 2), mid=np.nan, iv=v, delta=float(g.delta),
                                 gamma=float(g.gamma), vega=float(g.vega), theta=float(g.theta), underlying_price=spot,
                                 volume=100.0, open_interest=1000.0, exercise="AMERICAN", settlement=st,
                                 multiplier=100.0, fidelity=3))
    return pd.DataFrame(rows)


@pytest.fixture
def chain():
    return synthetic_chain()

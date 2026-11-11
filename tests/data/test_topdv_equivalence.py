"""SP500_TOP<N>_DV fast path (docs/PERF.md §6) == the original per-member _daily_frame code, bit for bit.

synthetic_ identity: the timeline / series layer is stubbed so the test pins the ranking mechanics only:
a ticker reused by two issuers inside the 120-day identity window (fail closed), a ticker that starts
mid-window (only that issuer's sessions count), a security key, an ETF (plain), an unknown identity,
NaN closes, a lineage hint and a non-session bar date. Every session of 2019-12..2020-06 is compared."""
import struct

import numpy as np
import pandas as pd

from atlasbt.data import frames
from atlasbt.data.portal import Portal, _session_date
from atlasbt.data.store import ParquetStore

T0, T1 = pd.Timestamp("2019-06-03"), pd.Timestamp("2020-07-31")
SWITCH = pd.Timestamp("2020-03-02")           # REU changes hands (two issuers)
LATE = pd.Timestamp("2020-02-10")             # NEWCO's first identity session


def _tl(rows):
    return pd.DataFrame(rows, columns=["start", "end", "cik", "via", "hint"])


TIMELINES = {
    "REU": _tl([(pd.Timestamp("2000-01-03"), SWITCH, 1, "as_of", None),
                (SWITCH, pd.Timestamp("2030-01-01"), 2, "as_of", None)]),
    "AAA": _tl([(pd.Timestamp("2000-01-03"), pd.Timestamp("2030-01-01"), 3, "as_of", "AAX")]),
    "NEWCO": _tl([(LATE, pd.Timestamp("2030-01-01"), 4, "as_of", None)]),
    "BBB": _tl([(pd.Timestamp("2000-01-03"), pd.Timestamp("2019-12-20"), 5, "as_of", None)]),  # carried after
    "GAP": _tl([]),
}


def _series(seed: int, nan_every: int = 0, extra_day: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = frames.schedule().index
    days = days[(days >= T0 - pd.Timedelta(days=200)) & (days <= T1)]
    if extra_day:                               # a bar on a non-session date (Saturday)
        days = days.append(pd.DatetimeIndex([pd.Timestamp("2020-01-11")])).sort_values()
    idx = (days + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
    close = rng.uniform(10, 500, len(idx))
    if nan_every:
        close[::nan_every] = np.nan
    return pd.DataFrame({"close": close, "volume": rng.integers(10_000, 5_000_000, len(idx)).astype("float64")},
                        index=pd.DatetimeIndex(idx, name="start"))


class _StubPortal(Portal):
    SERIES = {(1, None): _series(1, nan_every=7), (2, None): _series(2), (3, "AAX"): _series(3, extra_day=True),
              (4, None): _series(4), (5, None): _series(5, nan_every=3)}
    PLAIN = {"SPY": _series(9)}
    MEMBERS = ["REU", "AAA", "NEWCO", "BBB", "GAP", "SPY", "REU~CIK0000000001", "ZZZ~CIK0000000099"]

    def ticker_timeline(self, symbol):
        return TIMELINES.get(self._split_key(symbol)[0], TIMELINES["GAP"])

    def _is_plain(self, symbol):
        return symbol.upper() in self.PLAIN

    def _plain_frame(self, sym):
        return self.PLAIN[sym]

    def _cik_series(self, cik, symbol, hint=None):
        return self.SERIES.get((cik, hint), frames.empty_bars())

    def _universe(self, name, as_of):
        if name.upper() == "SP500":
            return list(self.MEMBERS)
        return super()._universe(name, as_of)


def _legacy(p, as_of, n, lookback=20):
    """The original _top_dollar_volume body (32d6a56)."""
    sess = _session_date(as_of)
    sch = frames.schedule().index
    prior = sch[sch < sess]
    lo, hi = prior[-lookback], prior[-1]
    scores = {}
    for s in p.universe("SP500", as_of):
        d = p._daily_frame(s, sess)
        if not len(d):
            continue
        dd = frames.to_naive_date(d.index)
        w = d.loc[(dd >= lo) & (dd <= hi)]
        if len(w) >= lookback // 2:
            scores[s] = float((w["close"] * w["volume"]).mean())
    return scores, sorted(scores, key=lambda k: -scores[k])[:n]


def test_topdv_fast_path_bit_identical(tmp_path):
    p = _StubPortal(ParquetStore(tmp_path / "store"))
    ref = _StubPortal(ParquetStore(tmp_path / "store"))
    sch = frames.schedule().index
    seen = set()
    for d in sch[(sch >= "2019-12-02") & (sch <= "2020-06-30")]:
        as_of = pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=10)
        old_scores, old5 = _legacy(ref, as_of, 5)
        assert p.universe("SP500_TOP5_DV", as_of) == old5
        assert p.universe("SP500_TOP2_DV", as_of) == old5[:2]
        new_scores = p._cache[("topdv_scores", "SP500", _session_date(as_of), 20)]
        assert list(new_scores) == list(old_scores)
        for k, v in old_scores.items():
            assert struct.pack("<d", v) == struct.pack("<d", new_scores[k]), (d, k)
        seen.add(tuple(old_scores))
    # the scenarios actually occur: REU fails closed around the switch, NEWCO enters, BBB carried, GAP never
    assert any("REU" not in s for s in seen) and any("REU" in s for s in seen)
    assert any("NEWCO" in s for s in seen) and any("NEWCO" not in s for s in seen)
    assert all("GAP" not in s and "ZZZ~CIK0000000099" not in s for s in seen)
    assert all("SPY" in s and "REU~CIK0000000001" in s and "BBB" in s for s in seen)

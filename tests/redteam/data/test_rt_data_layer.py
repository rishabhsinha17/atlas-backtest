"""Red-team tests for atlasbt.data against the REAL cache (cache/store).

Each test asserts the CORRECT behaviour; a failure is a finding (see the red-team report).
All tests skip when the real store (or the needed file) is absent.
Reference values: DoltHub stocks.ohlcv rows already in the store (as-traded, PIT ticker) and
public facts quoted in the docstrings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.data import config, frames, qa
from atlasbt.data.portal import Portal
from atlasbt.data.store import ParquetStore

STORE_ROOT = config.STORE_ROOT
ASOF = pd.Timestamp("2026-10-28 23:00", tz="UTC")


def _need(*paths):
    for kind, src, sym in paths:
        st = ParquetStore(STORE_ROOT)
        if not st.exists(kind, src, sym):
            pytest.skip(f"real cache lacks {kind}/{src}/{sym}")


@pytest.fixture(scope="module")
def portal():
    if not (STORE_ROOT / "bars_1d" / "yahoo").is_dir():
        pytest.skip("real cache absent")
    return Portal(ParquetStore(STORE_ROOT))


def _day(p, sym, d, adjusted=False):
    b = p.bars(sym, "1d", pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1), ASOF, adjusted=adjusted)
    assert len(b) == 1, (sym, d, len(b))
    return b.iloc[0], b.attrs


def _day_or_closed(p, sym, d, adjusted=False):
    """Like _day, but a FAIL-CLOSED answer (identity not established -> DataUnavailableError /
    no bar) is an accepted outcome for the identity cases (integrator: "fixed or fail closed")."""
    from atlasbt.core.types import DataUnavailableError
    try:
        b = p.bars(sym, "1d", pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1), ASOF, adjusted=adjusted)
    except DataUnavailableError:
        return None, {"source": "FAIL_CLOSED"}
    if not len(b):
        return None, {"source": "FAIL_CLOSED"}
    return b.iloc[0], b.attrs


def _dolt_close(p, sym, d):
    d0 = p.store.read("bars_1d", "dolthub", sym)
    d0.index = frames.to_naive_date(d0.index)
    return float(d0.loc[pd.Timestamp(d), "close"])


# --------------------------------------------------------------------------- identity / splice
@pytest.mark.parametrize("sym,day", [
    ("BBT", "2019-12-06"),    # BB&T (member 1997-12-04..2019-12-09); Yahoo 'BBT' is another security
    ("PARA", "2023-04-17"),   # Paramount Global (member 2022-02-17..2025-08-08); Yahoo 'PARA' is a ~$1 instrument
    ("IR", "2019-06-03"),     # old Ingersoll-Rand plc (member, now TT); Yahoo 'IR' = Gardner Denver
])
def test_member_prices_match_pit_dolthub(portal, sym, day):
    """Served raw close for an S&P 500 member during its membership must match the as-traded
    DoltHub print of the same ticker on the same day (it is 30-97% off)."""
    _need(("bars_1d", "dolthub", sym), ("bars_1d", "yahoo", sym))
    row, attrs = _day(portal, sym, day)
    ref = _dolt_close(portal, sym, day)
    assert row["close"] == pytest.approx(ref, rel=0.02), f"{sym} {day}: served {row['close']:.4f} ({attrs['source']}) vs as-traded {ref}"


@pytest.mark.parametrize("sym,day,floor", [
    ("CPWR", "2008-06-10", 5.0),   # Compuware traded ~$9-10 in 2008 (member until 2012-01-03)
    ("EP", "2008-06-10", 10.0),    # El Paso Corp traded ~$20 in 2008 (member until 2012-05-25)
    # verifier correction: the S&P member is the PIT ticker FISV (Fiserv); 'FI' in 2018 was Frank's
    # International, so the PIT universe now returns FISV and FI must NOT serve Fiserv prices in 2018
    ("FISV", "2018-01-03", 50.0),  # Fiserv traded ~$130 in 2018
    ("GENZ", "2010-03-02", 40.0),  # Genzyme traded ~$52 in 2010-03
])
def test_member_prices_not_from_reused_ticker(portal, sym, day, floor):
    """Either the right company's price, or FAIL CLOSED (no bar) -- never a reused ticker's."""
    row, attrs = _day_or_closed(portal, sym, day)
    if row is None:
        return
    assert row["close"] > floor, f"{sym} {day}: served {row['close']} ({attrs['source']}) -- wrong company"


def test_fi_2018_is_franks_not_fiserv(portal):
    row, attrs = _day_or_closed(portal, "FI", "2018-01-03")
    assert row is None or row["close"] < 20, f"FI 2018 served {row['close']} -- Fiserv prices under Frank's ticker"
    assert "FISV" in portal.universe("SP500", pd.Timestamp("2018-01-03 21:00", tz="UTC"))


def test_foxa_splice_fake_return(portal):
    """FOXA (21CF, then Fox Corp) spliced: DoltHub 50.53 on 2019-03-11, Yahoo when-issued Fox Corp
    38.00 on 2019-03-12 -> a fake -24.8% return for an index member."""
    _need(("bars_1d", "yahoo", "FOXA"))
    from atlasbt.core.types import DataUnavailableError
    try:
        b = portal.bars("FOXA", "1d", "2019-03-11", "2019-03-13", ASOF, adjusted=True)
    except DataUnavailableError:
        return                    # fail closed: 21CF's FOXA identity is not established
    if len(b) < 2:
        return
    r = float(b["close"].iloc[-1] / b["close"].iloc[0] - 1)
    assert abs(r) < 0.10, f"FOXA 2019-03-12 return {r:.3f} from source {b.attrs['source']}"


@pytest.mark.parametrize("sym,day", [("UTX", "2005-06-13"), ("BK", "1998-08-14"), ("SYMC", "2004-12-01"),
                                     ("MMC", "2002-07-01")])
def test_alias_history_splits_are_adjusted(portal, sym, day):
    """Bars spliced from a same-CIK alias (UTX<-RTX, BK<-BNY, ...) must be adjusted with the
    alias's splits. _actions_frame() returns the DoltHub fallback (non-empty, splits from 2014
    only) and never the alias's Yahoo actions -> adjusted 2:1 split days show -50%."""
    from atlasbt.core.types import DataUnavailableError
    try:
        b = portal.bars(sym, "1d", pd.Timestamp(day) - pd.Timedelta(days=5), pd.Timestamp(day) + pd.Timedelta(days=1),
                        ASOF, adjusted=True)
    except DataUnavailableError:
        return                    # fail closed (e.g. old Bank of New York, CIK 9626, pre-2007)
    if len(b) < 2:
        return
    r = float(b["close"].iloc[-1] / b["close"].iloc[-2] - 1)
    assert r > -0.25, f"{sym} adjusted return on {day} = {r:.3f} (source {b.attrs['source']})"


def test_viac_alias_junk(portal):
    """VIAC sessions missing from DoltHub are filled from alias 'PARA' (Yahoo junk ~$1)."""
    b = portal.bars("VIAC", "1d", "2021-04-28", "2021-05-06", ASOF, adjusted=True)
    r = np.log(b["close"]).diff().abs().max()
    assert r < np.log(1.5), f"VIAC max |log ret| {r:.2f}; closes {b['close'].round(3).tolist()}"


# --------------------------------------------------------------------------- survivorship flag
def test_top_dv_universe_reports_survivorship_bias(portal):
    """SP500_TOP50_DV is drawn from PIT S&P members WITH data; at 2008-06 36% of members (MER, WB,
    LEHMQ, WAMUQ, CFC, AABA, XTO, DELL, EMC, ...) have no prices, yet the report says no bias."""
    asof = pd.Timestamp("2008-06-02 20:00", tz="UTC")
    base = portal.universe_survivorship_bias("SP500", asof)
    assert base["coverage"] < 0.9
    rep = portal.universe_survivorship_bias("SP500_TOP50_DV", asof)
    assert rep["survivorship_bias"] is True, rep


# --------------------------------------------------------------------------- events
def test_q4_release_eps_actual_is_quarterly(portal):
    """AAPL 2023-11-02 release (FQ4) eps_actual must be the quarter's diluted EPS (1.46), not FY (6.13).
    providers/ziplime.build_events ignores ZipLime's `quarters` column."""
    ev = [e for e in portal.events("EARNINGS", ["AAPL"], "2023-11-01", "2023-11-03", ASOF)]
    assert ev, "no AAPL release"
    e = ev[0]
    assert e.payload["eps_actual"] < 2.0, (e.payload["eps_actual"], e.payload.get("eps_estimate"))


def test_q4_release_eps_regression_jpm(portal):
    """Regression (fixed during review via figure_quarters): JPM 2024-01-12 Q4 release carried FY GAAP
    EPS 16.23 as eps_actual vs a 3.65 quarterly consensus."""
    ev = portal.events("EARNINGS", ["JPM"], "2024-01-11", "2024-01-13", ASOF)
    assert ev and ev[0].payload["eps_actual"] < 6.0, ev[0].payload["eps_actual"] if ev else None
    assert ev[0].payload.get("eps_actual_gaap") is None or ev[0].payload["eps_actual_gaap"] < 6.0


# --------------------------------------------------------------------------- rates
def test_dtb3_not_known_on_fed_holiday(portal):
    """DTB3 for Fri 2025-10-10 is published in H.15 on Tue 2025-10-14 ~16:15 ET (Columbus Day is a
    Federal Reserve holiday; NYSE is open), but known_at = Mon 2025-10-13 16:30 ET."""
    r = portal.rate_series("DTB3")
    ka = r.loc[pd.Timestamp("2025-10-10"), "known_at"]
    assert ka >= pd.Timestamp("2025-10-14 20:15", tz="UTC"), ka
    v = portal.risk_free_rate(pd.Timestamp("2025-10-13 21:00", tz="UTC"), "3M")
    prev = portal.rate_series("DTB3").loc[pd.Timestamp("2025-10-09"), "value"] / 100
    from atlasbt.data import rates
    assert v == pytest.approx(rates.discount_to_money_market(prev, 91)), "used 2025-10-10 value before publication"


# --------------------------------------------------------------------------- actions
def test_spinoff_not_reported_as_split(portal):
    """Yahoo encodes the HPE spin-off as a 2.202 'split' on 2015-11-02; it is served as
    ActionType.SPLIT (share count x2.202 in the ledger) instead of a distribution."""
    acts = portal.corporate_actions(["HPQ"], "2015-11-01", "2015-11-03", ASOF)
    splits = [a for a in acts if a.type.value == "SPLIT"]
    assert not splits, splits


# --------------------------------------------------------------------------- QA
def test_qa_flags_dolthub_glitch(portal):
    """DoltHub CB 2014-09-08 whole bar 13.6% below neighbours (91.71 vs 106.13/105.51) is a
    spike-and-revert glitch; qa.bad_ticks (25% floor) misses it."""
    _need(("bars_1d", "dolthub", "CB"))
    d = portal.store.read("bars_1d", "dolthub", "CB")
    d.index = frames.to_naive_date(d.index)
    flags = qa.bad_ticks(d["close"])
    assert bool(flags.loc[pd.Timestamp("2014-09-08")]), "glitch not flagged"


# --------------------------------------------------------------------------- intraday contract
def test_60m_rth_known_at_capped_at_close(portal):
    """Native Yahoo 60m RTH bar 15:30-16:00 ET is complete at 16:00 but known_at = 16:30
    (frames.intraday_bars does not cap at the close, despite the comment); 12:30 on half days."""
    _need(("bars_60m", "yahoo", "SPY"))
    b = portal.store.read("bars_60m", "yahoo", "SPY")
    rth = b[b["session"] == "RTH"]
    closes = frames.schedule().reindex(frames.to_naive_date(rth.index))["close"].to_numpy()
    late = (rth["known_at"].to_numpy() > closes).sum()
    assert late == 0, f"{late} RTH 60m bars known after the session close"


# --------------------------------------------------------------------------- fundamentals
def test_fundamentals_future_period_end_blocks_updates(portal):
    """AAL shares_outstanding fact tagged period_end 2027-07-17 (filed 2026-07-23) wins
    'latest period' for every as_of until 2027 -> newer filings are never served."""
    _need(("fundamentals", "sec", "AAL"))
    f = portal.fundamentals(["AAL"], ["shares_outstanding"], ASOF)
    pe = pd.Timestamp(f["period_end"].iloc[0])
    assert pe <= ASOF.tz_localize(None), f"served period_end {pe.date()} (in the future)"


# --------------------------------------------------------------------------- synthetic chain PIT
def test_synthetic_params_point_in_time(portal):
    """Synthetic chains for 2012 use skew/wing/spread parameters fitted on later DoltHub chains
    (only atm_ratio has a PIT schedule). Agreed remedy (verified finding #6): such chains are
    not blocked but must carry the distinct T3 reason 'params_fit_post_as_of'."""
    import pandas as pd
    from atlasbt.core.types import Fidelity
    as_of = pd.Timestamp("2012-06-15 21:00", tz="UTC")
    chain = portal.option_chain("SPY", as_of)
    assert len(chain), "expected a synthetic fallback chain for 2012"
    assert (chain["fidelity"].astype(int) >= int(Fidelity.T3_MODELED)).all()
    reasons = " ".join(map(str, chain.get("reason", pd.Series(dtype=str)).dropna().unique())) + \
        " " + str(chain.attrs)
    assert "params_fit_post_as_of" in reasons, reasons[:300]


# --------------------------------------------------------------------------- identity regression set
# (integrator list) each case: FIXED (served close matches the as-traded DoltHub print of the PIT
# ticker, or the expected security id) or FAIL CLOSED. Never another company's prices.
KNOWN = [
    ("BBT", "2019-12-06", "dolt", None), ("PARA", "2023-04-17", "dolt", None), ("VIAC", "2021-04-30", "dolt", None),
    ("IR", "2019-06-03", "dolt", None), ("UTX", "2019-06-03", "dolt", None), ("SYMC", "2018-06-01", "dolt", None),
    ("MMC", "2015-06-01", "dolt", None), ("HRS", "2019-01-03", "dolt", None), ("COG", "2015-01-05", "dolt", None),
    ("CTRA", "2022-06-01", "dolt", None), ("FISV", "2018-01-03", "dolt", None),
    ("CPWR", "2008-06-10", "floor", 5.0), ("EP", "2008-06-10", "floor", 10.0), ("GENZ", "2010-03-02", "floor", 40.0),
    ("WEN", "2005-01-04", "floor", 25.0), ("AIT", "1998-06-01", "floor", 35.0), ("MCIC", "1997-06-02", "floor", 20.0),
    ("SII", "2005-01-04", "floor", 40.0),
    ("CCU", "2005-01-04", "sid", "CIK0000739708"), ("CNC", "2000-06-01", "sid", "CIK_CONSECO"),
    ("CB", "2010-06-01", "sid", "CIK0000020171"), ("FOXA", "2019-03-12", "sid", "CIK0001308161"),
    ("BK", "1998-08-14", "sid", "CIK0000009626"),
]


@pytest.mark.parametrize("sym,day,kind,ref", KNOWN)
def test_known_identity_cases(portal, sym, day, kind, ref):
    from atlasbt.core.types import DataUnavailableError
    try:
        b = portal.bars(sym, "1d", pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1), ASOF)
    except DataUnavailableError:
        assert kind != "dolt", f"{sym} {day}: fail closed although a verified as-traded source exists"
        return
    if not len(b):
        return
    c = float(b["close"].iloc[0])
    if kind == "dolt":
        assert c == pytest.approx(_dolt_close(portal, sym if sym != "CBS" else "VIAC", day), rel=0.02), (sym, c, b.attrs)
    elif kind == "floor":
        assert c > ref, f"{sym} {day}: {c} from {b.attrs.get('source')} -- wrong company"
    else:
        assert b.attrs.get("security_id") == ref, f"{sym} {day}: served security {b.attrs.get('security_id')}"


def test_cbs_history_from_dolthub_viac(portal):
    """Verifier correction: DoltHub files CBS 2011-2017 under VIAC; with identity verified (CIK
    813828 lineage CBS->VIAC->PARA + price continuity into the CBS-labelled rows) CBS is served."""
    b = portal.bars("CBS", "1d", "2012-06-01", "2012-06-02", ASOF)
    assert len(b) == 1 and b.attrs["security_id"] == "CIK0000813828"
    assert float(b["close"].iloc[0]) == pytest.approx(_dolt_close(portal, "VIAC", "2012-06-01"), rel=0.001)


# --------------------------------------------------------------------------- pre-2006 identity (integrator)
@pytest.mark.parametrize("sym,day,sid,lo,hi", [
    ("KO", "1996-06-28", "CIK0000021344", 40, 60),      # continuous member, Wikipedia CIK confirms
    ("GE", "1996-06-28", "CIK0000040545", 70, 100),
    ("ORCL", "2005-06-30", "CIK0000777676", 10, 16),    # holdco succession 777676 -> 1341439 (2006)
    ("DIS", "2012-06-29", "CIK0001001039", 40, 55),     # holdco succession 1001039 -> 1744489 (2019)
    ("XOM", "1996-06-28", "CIK0000034088", 75, 100),    # Exxon Corp (traded as XON; fja member 'XOM'); Yahoo line = Exxon
])
def test_pre_evidence_and_succession_served(portal, sym, day, sid, lo, hi):
    b = portal.bars(sym, "1d", pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1), ASOF)
    assert len(b) == 1 and b.attrs["security_id"] == sid and lo < float(b["close"].iloc[0]) < hi


@pytest.mark.parametrize("sym,day,bad", [("T", "2004-06-01", "CIK0000732717"),   # SBC took 'T' in 2005-11
                                         ("CB", "2012-06-01", "CIK0000896159")])
def test_pre_evidence_renamed_or_acquirer_not_served(portal, sym, day, bad):
    from atlasbt.core.types import DataUnavailableError
    try:
        b = portal.bars(sym, "1d", pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1), ASOF)
    except DataUnavailableError:
        return
    assert not len(b) or (bad is not None and b.attrs.get("security_id") != bad), (sym, b.attrs.get("security_id"))


def test_cvna_2021_2022_history_not_cut(portal):
    """Coordinator regression: CVNA's genuine 2021-2022 crash (alternating large days) must not
    trigger the Yahoo glitch-cluster cut; 2019-2022 history is served."""
    b = portal.bars("CVNA", "1d", pd.Timestamp("2019-01-01"), pd.Timestamp("2023-01-01"), ASOF)
    assert len(b) > 900 and b.index[0] < pd.Timestamp("2019-01-10", tz="UTC")
    assert not any("trim" in x for x in b.attrs.get("identity_log", []))


def test_has_agrees_with_bars_for_sp500_sample(portal):
    """Coordinator: has() must reflect identity-verified availability. Over a deterministic sample
    of S&P 500 members at several dates, the windowed has() equals 'bars() serves >= 1 bar', and
    universe(require_bars=True) keeps exactly those members."""
    from atlasbt.core.types import DataUnavailableError
    for day in ("1998-06-30", "2005-06-30", "2012-06-29", "2019-06-28", "2026-10-27"):
        a = pd.Timestamp(day + " 21:00", tz="UTC")
        mem = portal.universe("SP500", a)
        sample = mem[::25]
        s0 = a - pd.Timedelta(days=8)
        served = set()
        for s in sample:
            try:
                ok = len(portal.bars(s, "1d", s0, a, a)) > 0
            except DataUnavailableError:
                ok = False
            assert portal.has(s, "bars", "1d", start=s0, end=a, as_of=a) == ok, (day, s)
            if ok:
                served.add(s)
        req = set(portal.universe("SP500", a, require_bars=True))
        assert served == req & set(sample), day


@pytest.mark.parametrize("sym,day,sid,lo,hi", [
    ("FI", "2024-01-02", "CIK0000798354", 110, 160),     # Fiserv as FI (FISV -> FI -> FISV while in the index)
    ("BALL", "2024-01-02", "CIK0000009389", 45, 70),     # BLL -> BALL 2022-05-10 (ZipLime as-of ticker lags)
    ("QRVO", "2020-06-01", "CIK0001604778", 80, 130),    # a 1-bar Yahoo stub must not evict DoltHub history
])
def test_membership_continuity_and_stub_cases(portal, sym, day, sid, lo, hi):
    b = portal.bars(sym, "1d", pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1), ASOF)
    assert len(b) == 1 and b.attrs["security_id"] == sid and lo < float(b["close"].iloc[0]) < hi, \
        (float(b["close"].iloc[0]) if len(b) else None, b.attrs.get("security_id"))


@pytest.mark.parametrize("day", ["1998-06-30", "2003-06-30", "2008-06-30", "2013-06-28", "2019-06-28", "2025-06-30"])
def test_security_master_covers_every_universe_member(portal, day):
    """Coordinator: security_master(as_of) has one row per universe('SP500') member exactly as
    spelled (security keys included), resolved for that issuer by CIK, with a sector or an explicit
    sector_unknown_reason; members that bars() can serve always have a sector."""
    a = pd.Timestamp(day + " 21:00", tz="UTC")
    u = portal.universe("SP500", a)
    sm = portal.security_master(a).set_index("symbol")
    assert not [k for k in u if k not in sm.index]
    sub = sm.loc[u]
    assert (sub["sector"].notna() | sub["sector_unknown_reason"].notna()).all()
    for k in [k for k in u if "~CIK" in k][:20]:
        assert int(sub.loc[k, "cik"]) == int(k.split("~CIK")[1])
    unknown = sub[sub["sector"].isna()].index
    served = [k for k in unknown if portal.has(k, "bars", "1d", start=a - pd.Timedelta(days=8), end=a, as_of=a)]
    assert not served, f"tradable members without sector: {served[:10]}"


@pytest.mark.parametrize("day", ["2003-06-30", "2012-06-29", "2019-06-28", "2025-06-30"])
def test_key_resolution_consistent_for_all_members(portal, day):
    """Coordinator (BLOCKING): for EVERY universe('SP500') member at several dates: windowed has()
    equals 'bars() serves >= 1 bar'; fidelity_of(member, 'bars', '1d') != NA whenever has() is
    True; a security_master row exists; instrument() keeps the spelling."""
    from atlasbt.core.types import DataUnavailableError, Fidelity
    a = pd.Timestamp(day + " 21:00", tz="UTC")
    s0 = a - pd.Timedelta(days=8)
    u = portal.universe("SP500", a)
    sm = set(portal.security_master(a)["symbol"])
    bad = []
    for k in u:
        try:
            served = len(portal.bars(k, "1d", s0, a, a)) > 0
        except DataUnavailableError:
            served = False
        h = portal.has(k, "bars", "1d", start=s0, end=a, as_of=a)
        f = portal.fidelity_of(k, "bars", "1d")
        if h != served or (h and f == Fidelity.NA) or (portal.has(k, "bars", "1d") and f == Fidelity.NA) \
                or k not in sm or portal.instrument(k).symbol != k:
            bad.append((k, h, served, int(f), k in sm))
    assert not bad, bad[:10]

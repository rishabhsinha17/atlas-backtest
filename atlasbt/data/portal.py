"""atlasbt.data.portal.Portal -- point-in-time DataPortal (core.protocols.DataPortal).

Every public method takes `as_of` (tz-aware UTC; naive values are interpreted as UTC) and never
returns information whose knowledge time is after `as_of`:
  bars          known_at <= as_of (daily: session close; intraday: bar end)
  actions       known_at <= as_of (and adjustments only use ex_date <= as_of session)
  events        known_at <= as_of
  option chain  snapshot ts <= as_of
  rates         known_at <= as_of (next business day 16:30 ET publication rule)
  fundamentals  filing acceptance <= as_of
Source priority & fidelity are documented in docs/DATA.md.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from atlasbt.core.clock import ET
from atlasbt.core.types import (ActionType, AssetClass, CorporateAction, DataUnavailableError, ExerciseStyle,
                                Fidelity, Instrument, MarketEvent, OptionQuote, Right, SettlementType)

from . import adjust, frames, rates
from . import universes as U
from .store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.portal")

DAILY_SOURCES = ("yahoo", "dolthub", "tiingo", "cboe")
INTRADAY_SOURCES = ("yahoo", "alpaca", "polygon", "databento", "alphavantage", "tiingo")   # T1 sources
PROXY_INTRADAY_SOURCES = ("dukascopy",)                                                       # T2, opt-in
QUOTE_SOURCES = ("alpaca", "databento", "iex", "dukascopy")
SOURCE_FIDELITY = {"yahoo": Fidelity.T1_BAR, "dolthub": Fidelity.T1_BAR, "tiingo": Fidelity.T1_BAR,
                   "cboe": Fidelity.T1_BAR, "alpaca": Fidelity.T1_BAR, "polygon": Fidelity.T1_BAR,
                   "databento": Fidelity.T1_BAR, "alphavantage": Fidelity.T1_BAR,
                   "dukascopy": Fidelity.T2_PROXY, "qlib": Fidelity.T2_PROXY}
QUOTE_FIDELITY = {"alpaca": Fidelity.T0_QUOTE, "databento": Fidelity.T2_PROXY, "iex": Fidelity.T2_PROXY,
                  "dukascopy": Fidelity.T2_PROXY}
INDEX_SYMBOLS = {"VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "VIX1D", "VVIX", "SKEW", "VXN", "RVX", "VXD", "GVZ",
                 "OVX", "VXEEM", "VXEFA", "VXTLT", "PUT", "BXM", "BXMD", "BXMW", "CNDR", "BFLY", "CLL", "CMBO",
                 "PPUT", "PUTY", "RXM", "BXY", "COR1M", "COR3M", "DSPX", "SPX", "OEX", "XEO", "RUT", "DJX",
                 "VXAPL", "VXAZN", "VXGOG", "VXGS", "VXIBM", "VXGDX", "VXSLV"}
_OCC = re.compile(r"^(?P<root>[A-Z0-9.\- ]{1,6})\s*(?P<d>\d{6})(?P<r>[CP])(?P<k>\d{8})$")
RATE_SERIES = {"1M": ("DTB4WK", "discount", 28), "3M": ("DTB3", "discount", 91), "6M": ("DTB6", "discount", 182),
               "1Y": ("DTB1YR", "discount", 364), "1Y_CMT": ("DGS1", "be", None),
               "ON": ("DFF", "mm", 1), "SOFR": ("SOFR", "mm", 1),
               "2Y": ("DGS2", "be", None), "5Y": ("DGS5", "be", None), "10Y": ("DGS10", "be", None),
               "30Y": ("DGS30", "be", None)}

# Option data tier by SOURCE and timestamp quality (coordinator rule): exchange-timestamped quote
# snapshots / OPRA NBBO -> T0; EOD captured at a documented time -> T1; EOD with approximate capture
# time -> T2; synthetic -> T3 (option_chain(synthetic=True) / option_quotes fallback).
OPTION_SOURCE_FIDELITY = {
    "databento": Fidelity.T0_QUOTE,     # OPRA consolidated quotes, exchange timestamps (key-gated)
    "polygon": Fidelity.T0_QUOTE,       # SIP/OPRA quote snapshots (key-gated)
    "alpaca": Fidelity.T0_QUOTE,        # OPRA snapshots (key-gated)
    "cboe_eod": Fidelity.T1_BAR,        # Cboe EOD summary captured at 15:45 ET (documented time)
    "hod": Fidelity.T2_PROXY,           # historicaloptiondata EOD, capture minute not documented
    "dolthub": Fidelity.T2_PROXY,       # EOD near the close, capture minute unknown (§4.6)
    "synthetic": Fidelity.T3_MODELED,
}


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _session_date(ts) -> pd.Timestamp:
    return _utc(ts).tz_convert(ET).tz_localize(None).normalize()


def _naive_date(d) -> pd.Timestamp:
    t = pd.Timestamp(d)
    if t.tzinfo is not None:
        t = t.tz_convert(ET).tz_localize(None)
    return t.normalize()


class Portal:
    """Point-in-time DataPortal over the parquet store."""

    def __init__(self, store: Optional[ParquetStore] = None, allow_proxy_intraday: bool = False):
        self.store = store or default_store()
        self.allow_proxy_intraday = allow_proxy_intraday
        self._cache: dict = {}

    # ================================================================ helpers
    def _read(self, kind, source, symbol, **kw) -> pd.DataFrame:
        key = (kind, source, symbol.upper(),
               tuple(sorted((k, tuple(v) if isinstance(v, list) else str(v)) for k, v in kw.items())))
        if key not in self._cache:
            self._cache[key] = self.store.read(kind, source, symbol, **kw)
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

    def _bar_source(self, symbol: str, freq: str, source: Optional[str] = None) -> Optional[str]:
        kind = f"bars_{freq}"
        if source:
            return source if self.store.exists(kind, source, symbol) else None
        cands = DAILY_SOURCES if freq == "1d" else INTRADAY_SOURCES + (
            PROXY_INTRADAY_SOURCES if self.allow_proxy_intraday else ())
        for s in cands:
            if self.store.exists(kind, s, symbol):
                return s
        return None

    # ================================================================ reference
    def instrument(self, symbol: str) -> Instrument:
        s = symbol.upper()
        m = _OCC.match(s)
        if m and len(s) >= 15:
            from .option_meta import AM_INDEX_ROOTS, last_trading_day
            root = m.group("root").strip()
            vendor = pd.Timestamp("20" + m.group("d"))
            exp, settle_d = last_trading_day([vendor], [root])
            strike = int(m.group("k")) / 1000.0
            right = Right(m.group("r"))
            und = {"SPXW": "SPX", "NDXP": "NDX", "RUTW": "RUT"}.get(root, root)
            if root in AM_INDEX_ROOTS | {"SPXW", "XSP", "NDXP", "RUTW"}:
                settle = SettlementType.CASH_AM if root in AM_INDEX_ROOTS else SettlementType.CASH_PM
                ex = ExerciseStyle.EUROPEAN
            else:
                settle, ex = SettlementType.PHYSICAL, ExerciseStyle.AMERICAN
            # keep the vendor OCC symbol; expiry = last trading day (may differ from the symbol date)
            return Instrument(symbol=s, asset_class=AssetClass.OPTION, multiplier=100.0, underlying=und,
                              expiry=exp[0], strike=strike, right=right, exercise=ex, settlement=settle)
        if s in INDEX_SYMBOLS:
            return Instrument(s, AssetClass.INDEX)
        return Instrument(s, AssetClass.ETF if self._is_etf(s) else AssetClass.STOCK)

    def _is_etf(self, symbol: str) -> bool:
        s = symbol.upper()
        if s in U.CORE_ETFS or s in U.SECTOR_SPDRS:
            return True
        d = self._directory()
        if len(d):
            hit = d[d["symbol"].str.upper().str.replace(".", "-", regex=False) == s]
            if len(hit):
                return bool(hit["etf"].iloc[-1])
        return False

    def _directory(self) -> pd.DataFrame:
        return self._read("reference", "nasdaqtrader", "directory")

    def security_master(self, as_of) -> pd.DataFrame:
        """One row per known symbol. NOT fully PIT: SIC and names are the latest SEC values,
        GICS is Wikipedia's *current* classification (gics_sector_current); sector_etf only
        lists a Sector SPDR that already existed at as_of. first/last bar dates use only bars
        known at as_of."""
        as_of = _utc(as_of)
        sm = self._read("reference", "sec", "security_master")
        rows = {}
        if len(sm):
            for r in sm.reset_index().to_dict("records"):
                rows[r["symbol"]] = {"symbol": r["symbol"], "name": r.get("name"), "cik": r.get("cik"),
                                     "sic": r.get("sic"), "sic_description": r.get("sic_description"),
                                     "sector": r.get("sector_sic"), "exchange": r.get("exchanges")}
        cur = self._read("reference", "sp500", "current_constituents")
        if len(cur):
            for r in cur.reset_index().to_dict("records"):
                sym = str(r["symbol"]).replace(".", "-")
                d = rows.setdefault(sym, {"symbol": sym, "name": r.get("name"), "cik": r.get("cik")})
                d["gics_sector_current"] = r.get("gics_sector")
                d["gics_sub_industry_current"] = r.get("gics_sub_industry")
                d.setdefault("sector", None)
                if not d.get("sector"):
                    d["sector"] = r.get("gics_sector")
        for etf, (sector, inception) in U.SECTOR_SPDRS.items():
            rows.setdefault(etf, {"symbol": etf, "name": f"Select Sector SPDR {sector}"})
        for s in U.CORE_ETFS:
            rows.setdefault(s, {"symbol": s})
        df = pd.DataFrame(list(rows.values()))
        df["asset_class"] = [("ETF" if self._is_etf(s) else "STK") for s in df["symbol"]]
        df["sector_etf"] = [U.sector_etf_for(sec, as_of) if isinstance(sec, str) else None
                            for sec in df.get("sector", pd.Series([None] * len(df)))]
        members = set(self.universe("SP500", as_of)) if self.store.exists("reference", "sp500", "membership") else set()
        df["in_sp500"] = df["symbol"].isin(members)
        firsts, lasts = [], []
        for s in df["symbol"]:
            src = self._bar_source(s, "1d")
            if src:
                b = self._read("bars_1d", src, s, columns=["known_at"])
                b = b[b["known_at"] <= as_of]
                firsts.append(b.index[0] if len(b) else pd.NaT)
                lasts.append(b.index[-1] if len(b) else pd.NaT)
            else:
                firsts.append(pd.NaT)
                lasts.append(pd.NaT)
        df["first_bar"] = firsts
        df["last_bar"] = lasts
        df.attrs.update({"source": "sec+wikipedia+nasdaqtrader", "fidelity": int(Fidelity.T2_PROXY),
                         "pit_note": "SIC/GICS are current classifications, not point-in-time"})
        return df.sort_values("symbol").reset_index(drop=True)

    def universe(self, name: str, as_of) -> list[str]:
        as_of = _utc(as_of)
        n = name.upper()
        if n in ("SP500", "S&P500", "SPX_MEMBERS"):
            m = self._read("reference", "sp500", "membership")
            if not len(m):
                raise DataUnavailableError("S&P 500 membership not in store (run fetch --only sp500)")
            d = _session_date(as_of)
            # a member is in the index for sessions in [start, end)
            cur = m[(m["start"] <= d) & (m["end"].isna() | (m["end"] > d))]
            # PIT tickers (the symbol as it traded then, '.' -> '-'), never today's Yahoo ticker;
            # Portal.bars / security_id resolve them to the security they denoted at the date
            return sorted({str(b).upper().replace(".", "-") for b in cur["base"]})
        mt = re.fullmatch(r"SP500_TOP(\d+)_DV", n)
        if mt:
            return self._top_dollar_volume("SP500", int(mt.group(1)), as_of)
        if n in U.STATIC_UNIVERSES:
            out = []
            for s in U.STATIC_UNIVERSES[n]:
                src = self._bar_source(s, "1d")
                if not src:
                    continue
                b = self._read("bars_1d", src, s, columns=["known_at"])
                if len(b) and b["known_at"].iloc[0] <= as_of:   # listed (has a completed bar) by as_of
                    out.append(s)
            return out
        raise KeyError(f"unknown universe {name!r}; known: SP500, {', '.join(U.STATIC_UNIVERSES)}")

    def _top_dollar_volume(self, base: str, n: int, as_of: pd.Timestamp, lookback: int = 20) -> list[str]:
        """PIT top-n members of `base` by mean raw close x volume over the `lookback` sessions
        strictly BEFORE the session of as_of (never uses the current session's bar)."""
        sess = _session_date(as_of)
        key = ("topdv", base, n, sess, lookback)
        if key in self._cache:
            return self._cache[key]
        sch = frames.schedule().index
        prior = sch[sch < sess]
        if len(prior) < lookback:
            return []
        lo, hi = prior[-lookback], prior[-1]
        scores = {}
        for s in self.universe(base, as_of):
            d = self._daily_frame(s, sess)
            if not len(d):
                continue
            dd = frames.to_naive_date(d.index)
            m = (dd >= lo) & (dd <= hi)
            w = d.loc[m]
            if len(w) >= lookback // 2:
                scores[s] = float((w["close"] * w["volume"]).mean())
        out = sorted(scores, key=lambda k: -scores[k])[:n]
        self._cache[key] = out
        return out

    def universe_survivorship_bias(self, name: str, as_of) -> dict:
        from .survivorship import universe_bias_report
        return universe_bias_report(self, name, as_of)

    # ================================================================ prices
    def bars(self, symbol: str, freq: str, start, end, as_of, adjusted: bool = False,
             sessions: Sequence[str] = ("RTH",), source: Optional[str] = None) -> pd.DataFrame:
        freq = {"1h": "60m", "1min": "1m", "D": "1d", "1D": "1d"}.get(freq, freq)
        as_of, start, end = _utc(as_of), _utc(start), _utc(end)
        alias = None
        ident = None
        if freq == "1d" and source is None:
            rdate = _session_date(min(end - pd.Timedelta(microseconds=1), as_of))
            ident = self._identity_frame(symbol, rdate)
            if not len(ident):
                if not self._is_plain(symbol):
                    raise DataUnavailableError(f"{symbol}: security identity not established at {rdate.date()} "
                                               f"(fail closed); log: {ident.attrs.get('identity_log', [])[-1:]}")
                native = self._bar_source(symbol, freq, source)
                ident = None
            else:
                native = "identity"
        else:
            native = self._bar_source(symbol, freq, source)
        resample_from = None
        if native is None and freq not in ("1d", "1m"):
            # build coarser intraday bars from 1m (causal: a bar is known at its end)
            if self._bar_source(symbol, "1m", source):
                resample_from = "1m"
        if native is None and resample_from is None:
            raise DataUnavailableError(f"no {freq} bars for {symbol}")
        if native:
            df = ident if native == "identity" else self._read(f"bars_{freq}", native, symbol)
            src = (df.attrs.get("source") or "identity") if native == "identity" else native
        else:
            src = self._bar_source(symbol, "1m", source)
            key = ("resampled", src, symbol.upper(), freq)
            if key not in self._cache:
                self._cache[key] = resample_bars(self._read("bars_1m", src, symbol), freq)
            df = self._cache[key]
        if df.empty:
            return frames.empty_bars()
        sel = (df.index >= start) & (df.index < end) & (df["known_at"] <= as_of)
        out = df.loc[sel]
        if freq != "1d" and sessions:
            out = out[out["session"].isin(tuple(sessions))]
        cols = [c for c in ("open", "high", "low", "close", "volume", "vwap", "trades", "session", "known_at")
                if c in out.columns]
        out = out[cols].copy()
        if adjusted and len(out):
            out = self._adjust(symbol, out, as_of, src, ident)
        fid = SOURCE_FIDELITY.get(str(src).split("+")[0], Fidelity.T1_BAR)
        if ident is not None and len(out) and "src_ticker" in df.columns:
            used = set(df.loc[out.index, "src_ticker"]) | set(df.loc[out.index, "src"])
            if used - {symbol.upper(), "yahoo"}:
                fid = max(fid, Fidelity.T2_PROXY)          # renamed ticker / DoltHub segment spliced in
        out.attrs.update({"source": src, "fidelity": int(fid), "adjusted": adjusted, "symbol": symbol.upper(),
                          "freq": freq})
        if ident is not None:
            out.attrs["security_id"] = ident.attrs.get("security_id")
            out.attrs["identity_log"] = ident.attrs.get("identity_log")
            from .identity import norm as _norm
            note = self._cache.get(("resolve_note", _norm(symbol), pd.Timestamp(rdate).normalize()))
            if note:
                out.attrs["fidelity"] = int(max(fid, Fidelity.T2_PROXY))
                out.attrs["identity_log"] = list(out.attrs["identity_log"] or []) + [note]
        return out

    def _adjust(self, symbol: str, bars: pd.DataFrame, as_of: pd.Timestamp, src: str,
                ident: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if ident is None and self._is_plain(symbol) is False:
            ident = self._identity_frame(symbol, _session_date(as_of))
        if ident is not None and not len(ident) and bars["session"].ne("DAY").any():
            # intraday-only data (no verified daily history): Yahoo intraday is recent and belongs to
            # the ticker's CURRENT owner, so its own (reclassified) Yahoo actions apply
            ident = self._plain_frame(symbol.upper())
            ident.attrs["actions"] = self._yahoo_actions(symbol)
        acts = self.corporate_actions([symbol], pd.Timestamp("1900-01-01"), _session_date(as_of), as_of,
                                      _ident=ident)
        acts = [a for a in acts if _naive_date(a.ex_date) <= _session_date(as_of) and a.type != ActionType.DELIST]
        if not acts:
            return bars
        daily = ident if ident is not None and len(ident) else (
            self._plain_frame(symbol.upper()) if self._is_plain(symbol) else frames.empty_bars())
        daily_src = (daily.attrs.get("security_id") or "plain") if len(daily) else None
        if daily_src:
            dd = frames.to_naive_date(daily.index)
            raw_close = pd.Series(daily["close"].to_numpy(), index=dd)
        else:   # no daily bars: session closes from the intraday RTH bars themselves
            rth = bars[bars["session"] == "RTH"] if "session" in bars else bars
            raw_close = rth["close"].groupby(frames.to_naive_date(rth.index)).last()
            dd = pd.DatetimeIndex(raw_close.index)
            daily_src = None
        # dividend factors need the raw close before ex: compute on the daily grid, then map.
        # Memoized per visible action set (the engine asks for history at every step).
        fkey = ("factors", symbol.upper(), daily_src, len(dd),
                tuple((a.type.value, a.ex_date, a.ratio, a.amount) for a in acts))
        if daily_src and fkey in self._cache:
            pf_d, vf_d = self._cache[fkey]
        else:
            pf_d, vf_d = adjust.adjustment_factors(dd, raw_close, acts)
            if daily_src:
                self._cache[fkey] = (pf_d, vf_d)
        bd = frames.to_naive_date(bars.index)
        if len(bars) and (bars["session"] == "DAY").all() and dd.get_indexer(frames.to_naive_date(bars.index)).min() >= 0:
            pos = dd.get_indexer(bd)
        else:
            pos = dd.searchsorted(bd, side="left")       # intraday: factor of its session date
            pos = np.clip(pos, 0, len(dd) - 1)
            # a session missing from daily data: use the next daily factor (same basis)
        pf, vf = pf_d[pos], vf_d[pos]
        # intraday bars on/after the last daily bar's date get factor of that bar; fine (no later actions)
        return adjust.apply_factors(bars, pf, vf)

    # ---------------------------------------------------------------- identity (red-team #1-#4)
    def identity_index(self):
        if "identity_index" not in self._cache:
            from .identity import IdentityIndex
            self._cache["identity_index"] = IdentityIndex(self.store)
        return self._cache["identity_index"]

    def _is_plain(self, symbol: str) -> bool:
        """ETFs and indices bypass the identity layer (their tickers are not reused in practice)."""
        s = symbol.upper()
        if s in INDEX_SYMBOLS or s in U.CORE_ETFS or s in U.SECTOR_SPDRS:
            return True
        if self.store.exists("bars_1d", "cboe", s) and not self.store.exists("bars_1d", "yahoo", s):
            return True
        if self._is_etf(s):
            # an ETF today, but was the ticker an operating company's (GENZ: Genzyme)? -> identity layer
            from .identity import norm
            e = self.identity_index().by_ticker.get(norm(s))
            return e is None or not len(e)
        return False

    def _name_hint(self, symbol: str, date) -> Optional[str]:
        """Security name of the S&P 500 member `symbol` around `date` (Wikipedia changes / current
        constituents), used to veto identities resolved from carried-only evidence."""
        from .identity import norm
        key = ("name_hints",)
        if key not in self._cache:
            hints: dict = {}
            w = self._read("reference", "sp500", "wiki_changes")
            for r in w.to_dict("records") if len(w) else []:
                for col, ncol in (("added", "added_name"), ("removed", "removed_name")):
                    t, n = r.get(col), r.get(ncol)
                    if isinstance(t, str) and isinstance(n, str):
                        hints.setdefault(norm(t), []).append((pd.Timestamp(r["effective_date"]), n))
            cur = self._read("reference", "sp500", "current_constituents")
            for r in cur.to_dict("records") if len(cur) else []:
                hints.setdefault(norm(r["symbol"]), []).append((pd.Timestamp.now().normalize(), r["name"]))
            self._cache[key] = hints
        h = self._cache[key].get(norm(symbol))
        if not h:
            return None
        d = pd.Timestamp(date)
        best = min(h, key=lambda x: abs((x[0] - d).days))
        return best[1] if abs((best[0] - d).days) <= 3 * 365 else None

    def _member_name(self, symbol: str, date) -> tuple[Optional[str], Optional[int]]:
        """(name, wikipedia_cik) of the S&P 500 member `symbol` for the membership interval that
        contains `date`: current constituents (name + CIK) if the interval is still open, else the
        Wikipedia removal (or addition) row at the interval's end (start), +-15 days. (None, None)
        when `symbol` was not a member at `date` or no name is recorded for that interval."""
        from .identity import norm
        sym, d = norm(symbol), pd.Timestamp(date).normalize()
        m = self._read("reference", "sp500", "membership")
        if not len(m):
            return None, None
        keys = m["base"].astype(str).map(norm)
        iv = m[(keys == sym) & (m["start"] <= d) & (m["end"].isna() | (m["end"] > d))]
        if not len(iv):
            return None, None
        r = iv.iloc[0]
        if pd.isna(r["end"]):
            cur = self._read("reference", "sp500", "current_constituents")
            hit = cur[cur["symbol"].astype(str).map(norm) == sym] if len(cur) else cur
            if len(hit):
                ck = pd.to_numeric(hit["cik"].iloc[0], errors="coerce")
                return str(hit["name"].iloc[0]), (int(ck) if pd.notna(ck) else None)
            return None, None
        w = self._read("reference", "sp500", "wiki_changes")
        if not len(w):
            return None, None
        tol = pd.Timedelta(days=15)
        for col, ncol, when in (("removed", "removed_name", r["end"]), ("added", "added_name", r["start"])):
            hit = w[(w[col].astype(str).map(norm) == sym) & ((w["effective_date"] - pd.Timestamp(when)).abs() <= tol)]
            hit = hit[hit[ncol].notna()]
            if len(hit):
                return str(hit[ncol].iloc[0]), None
        return None, None

    def security_id(self, symbol: str, as_of) -> Optional[str]:
        """'CIK##########' for the security `symbol` denoted at as_of, 'TKR:<sym>' for ETFs/indices,
        None when the identity cannot be established (fail closed)."""
        kind, key = self._resolve(symbol, _session_date(_utc(as_of)))
        if kind == "cik":
            return f"CIK{key:010d}"
        return f"TKR:{key}" if kind == "plain" else None

    def _resolve(self, symbol: str, date) -> tuple[Optional[str], object]:
        from .identity import names_match, cik_names, cik_name_at, norm
        sym = norm(symbol)
        d = pd.Timestamp(date).normalize()
        ck = ("resolve", sym, d)
        if ck in self._cache:
            return self._cache[ck]
        if self._is_plain(sym):
            res = ("plain", sym)
        else:
            idx = self.identity_index()
            c_asof = int(idx.label_dates(sym, pd.DatetimeIndex([d]))[0])
            c = c_asof or idx.ticker_cik_at(sym, d, allow_carried=True)
            res = (None, None)

            def renamed_since(cik_: int) -> bool:
                # issuer renamed between d and its first as-of evidence under this ticker -> its
                # ticker may have changed too (SBC Communications became AT&T Inc. and took "T" in
                # 2005-11; "T" in 2004 was AT&T Corp.) -> not established
                e = idx.by_ticker.get(sym)
                e = e[(e["conf"] == 0) & (e["cik"] == cik_)] if e is not None else None
                if e is None or not len(e) or e["date"].iloc[0] <= d:
                    return False
                n0, n1 = cik_name_at(cik_, d), cik_name_at(cik_, e["date"].iloc[0])
                return n0 is None or n1 is None or not names_match(n1, [n0])

            if c:
                hint = self._member_name(sym, d)[0] or self._name_hint(sym, d)
                if not c_asof and hint and not names_match(hint, cik_names(c)):
                    log.info("identity veto %s %s: CIK %s names %s do not match member name %r",
                             sym, d.date(), c, cik_names(c)[:2], hint)
                elif not c_asof and renamed_since(int(c)):
                    log.info("identity veto %s %s: CIK %s was named %r then (renamed before its first as-of "
                             "evidence under %s)", sym, d.date(), c, cik_name_at(int(c), d), sym)
                else:
                    res = ("cik", int(c))
                    if not c_asof:
                        self._cache[("resolve_note", sym, d)] = (
                            f"carried evidence: {sym} {d.date()} attributed to CIK {c} from carried ticker "
                            f"evidence within {120} days, name checked (T2)")
            else:
                # before the ticker's first as-of evidence (2006+): accept the earliest evidenced
                # issuer only if the S&P member of this membership interval confirms it
                pc = idx.pre_evidence_cik(sym, d)
                if pc:
                    name, wcik = self._member_name(sym, d)
                    ok = (wcik is not None and int(wcik) == pc) or (name is not None and names_match(name, cik_names(pc)))
                    if not ok and wcik is not None and idx.is_successor(pc, int(wcik), sym):
                        # member's CIK is the successor issuer on the same ticker (holdco reorganization)
                        ok = any(names_match(a, cik_names(int(wcik))) for a in cik_names(pc))
                    if ok and renamed_since(pc):
                        log.info("identity pre-evidence %s %s: CIK %s was named %r then (renamed before 2006 "
                                 "evidence; fail closed)", sym, d.date(), pc, cik_name_at(pc, d))
                        ok = False
                    if ok:
                        res = ("cik", int(pc))
                        self._cache[("resolve_note", sym, d)] = (
                            f"pre-evidence: {sym} {d.date()} precedes the first as-of ticker evidence; issuer "
                            f"CIK {pc} confirmed by S&P member {'CIK' if wcik is not None and int(wcik) == pc else 'name'}"
                            f" {name!r}, no rename since (T2)")
                    else:
                        log.info("identity pre-evidence %s %s: CIK %s not confirmed (member %r; fail closed)",
                                 sym, d.date(), pc, name)
        self._cache[ck] = res
        return res

    def _identity_frame(self, symbol: str, date=None) -> pd.DataFrame:
        """Verified RAW daily history of the security `symbol` denoted on `date` (default: latest
        session). attrs: source, security_id, identity_log, actions (frame). Empty if the identity
        cannot be established (fail closed)."""
        from .identity import build_series, norm
        if date is None:
            date = frames.schedule().index[frames.schedule().index <= pd.Timestamp.now().normalize()][-1]
        kind, key = self._resolve(symbol, date)
        if kind == "plain":
            return self._plain_frame(key)
        if kind != "cik":
            out = frames.empty_bars()
            out.attrs.update({"source": None, "security_id": None, "identity_log": ["FAIL CLOSED: identity unknown"]})
            return out
        idx = self.identity_index()
        main, _ = idx.lineage(key, norm(symbol))
        ck = ("identity_series", key, tuple(main))
        if ck not in self._cache:
            got = self._load_identity(key, main)
            if got is not None:
                self._cache[ck] = got
                return got
            m, acts, lg = build_series(self, idx, key, norm(symbol))
            m.attrs.update({"source": "+".join(sorted(set(m["src"]))) if len(m) else None,
                            "security_id": f"CIK{key:010d}", "identity_log": lg, "actions": acts})
            self._cache[ck] = m
        return self._cache[ck]

    @staticmethod
    def _identity_store_key(cik: int, main) -> str:
        return f"CIK{cik:010d}_{'_'.join(main)[:60]}"

    def _load_identity(self, cik: int, main) -> Optional[pd.DataFrame]:
        """Persisted series from `python -m atlasbt.data.fetch --only identity` (same build version)."""
        from .identity import BUILD_VERSION
        k = self._identity_store_key(cik, main)
        if not self.store.exists("identity", "series", k):
            return None
        meta = self.store.meta("identity", "series", k)
        if meta.get("build_version") != BUILD_VERSION:
            return None
        m = self.store.read("identity", "series", k)
        if meta.get("empty"):
            m = frames.empty_bars()
        acts = self.store.read("identity", "actions", k) if self.store.exists("identity", "actions", k) else pd.DataFrame()
        if len(acts) and "ex_date" not in acts.columns:
            acts = acts.reset_index()
        m.attrs.update({"source": meta.get("served_sources"), "security_id": f"CIK{cik:010d}",
                        "identity_log": meta.get("identity_log", []), "actions": acts})
        return m

    def persist_identity(self, symbol: str, date=None) -> dict:
        """Build and store the verified series of the security `symbol` denotes on `date`."""
        from .identity import BUILD_VERSION, build_series, norm
        if date is None:
            date = frames.schedule().index[frames.schedule().index <= pd.Timestamp.now().normalize()][-1]
        kind, key = self._resolve(symbol, date)
        if kind != "cik":
            return {"symbol": symbol, "date": str(pd.Timestamp(date).date()), "identity": kind}
        idx = self.identity_index()
        main, _ = idx.lineage(key, norm(symbol))
        k = self._identity_store_key(key, main)
        if self.store.exists("identity", "series", k) and \
                self.store.meta("identity", "series", k).get("build_version") == BUILD_VERSION:
            return {"symbol": symbol, "key": k, "cached": True}
        m, acts, lg = build_series(self, idx, key, norm(symbol))
        extra = {"build_version": BUILD_VERSION, "identity_log": lg[:200], "empty": not len(m),
                 "served_sources": "+".join(sorted(set(m["src"]))) if len(m) else None}
        store_m = m if len(m) else pd.DataFrame({"close": [np.nan]}, index=pd.DatetimeIndex(
            [pd.Timestamp("1900-01-01", tz="UTC")], name="start"))
        self.store.write("identity", "series", k, store_m, replace=True, extra=extra)
        if len(acts):
            a = acts.copy()
            for c in a.columns:
                if a[c].dtype == object:
                    a[c] = a[c].astype(str).where(a[c].notna(), None)
            self.store.write("identity", "actions", k, a.reset_index(drop=True), replace=True,
                             key=["ex_date", "type", "src_ticker"])
        return {"symbol": symbol, "key": k, "rows": len(m), "actions": len(acts)}

    def _plain_frame(self, sym: str) -> pd.DataFrame:
        key = ("plain_frame", sym)
        if key not in self._cache:
            src = "yahoo" if self.store.exists("bars_1d", "yahoo", sym) else self._bar_source(sym, "1d")
            f = self._read("bars_1d", src, sym).copy() if src else frames.empty_bars()
            f.attrs.update({"source": src, "security_id": f"TKR:{sym}", "identity_log": [],
                            "actions": self._yahoo_actions(sym) if src == "yahoo" else pd.DataFrame()})
            self._cache[key] = f
        return self._cache[key]

    def _yahoo_actions(self, sym: str) -> pd.DataFrame:
        """Yahoo actions with spin-off / special-dividend reclassification (actions_classify)."""
        key = ("yahoo_actions", sym.upper())
        if key not in self._cache:
            from .actions_classify import classify
            a = self._read("actions", "yahoo", sym)
            self._cache[key] = classify(sym, a, self._read("bars_1d", "yahoo", sym)) if len(a) else a
        return self._cache[key]

    def _daily_frame(self, symbol: str, date=None) -> pd.DataFrame:
        return self._identity_frame(symbol, date)

    def _identity_any(self, symbol: str) -> pd.DataFrame:
        """Identity frame at the latest session, else at the ticker's last as-of evidence date
        (historical tickers such as GENZ)."""
        f = self._identity_frame(symbol)
        if len(f) or self._is_plain(symbol):
            return f
        from .identity import norm
        e = self.identity_index().by_ticker.get(norm(symbol))
        if e is not None and len(e):
            return self._identity_frame(symbol, e["date"].iloc[-1])
        return f

    def quotes(self, symbol: str, start, end, as_of, source: Optional[str] = None) -> pd.DataFrame:
        as_of, start, end = _utc(as_of), _utc(start), _utc(end)
        for s in ((source,) if source else QUOTE_SOURCES):
            if self.store.exists("quotes", s, symbol):
                q = self.store.read("quotes", s, symbol, start=start, end=min(end, as_of + pd.Timedelta(microseconds=1)))
                q = q[(q.index >= start) & (q.index < end) & (q.index <= as_of)]
                q = q[[c for c in ("bid", "ask", "bid_size", "ask_size") if c in q.columns]]
                q.attrs.update({"source": s, "fidelity": int(QUOTE_FIDELITY.get(s, Fidelity.T2_PROXY))})
                return q
        raise DataUnavailableError(f"no quotes for {symbol}")

    EVENT_SOURCES = {"EARNINGS": [("events", "ziplime", "EARNINGS"), ("reference", "dolthub", "earnings_calendar")],
                     "EARNINGS_SCHEDULE": [("reference", "dolthub", "earnings_calendar")],
                     "INDEX_ADD": [("events", "sp500", "INDEX")], "INDEX_DEL": [("events", "sp500", "INDEX")],
                     "FOMC": [("events", "fomc", "FOMC")], "FOMC_SCHEDULE": [("events", "fomc", "FOMC")]}

    def _event_symbols(self, kind: str) -> set:
        key = ("event_symbols", kind)
        if key not in self._cache:
            syms: set = set()
            if kind in ("EARNINGS", "EARNINGS_SCHEDULE"):
                if kind == "EARNINGS" and self.store.exists("events", "ziplime", "EARNINGS"):
                    syms |= set(self.store.read("events", "ziplime", "EARNINGS", columns=["symbol"])["symbol"])
                c = self._read("reference", "dolthub", "earnings_calendar")
                if len(c):
                    syms |= set(c["act_symbol"].str.upper().str.replace(".", "-", regex=False))
            elif kind in ("INDEX_ADD", "INDEX_DEL"):
                e = self._read("events", "sp500", "INDEX")
                syms = set(e.loc[e["kind"] == kind, "symbol"]) if len(e) else set()
            elif kind == "EARNINGS_RELEASE":
                syms = {s.replace("EARNINGS_RELEASE_", "") for s in self.store.symbols("events", "sec")}
            self._cache[key] = syms
        return self._cache[key]

    def _fundamental_fields(self, symbol: Optional[str] = None) -> set:
        from .providers.sec import FIELD_MAP
        if symbol is None:
            if not self.store.symbols("fundamentals", "sec"):
                return set()
            return {f + sfx for f in FIELD_MAP for sfx in ("", "_q", "_fy", "_ytd")}
        f = self._read("fundamentals", "sec", symbol, columns=["field"])
        return set(f["field"]) if len(f) else set()

    def has(self, symbol: str, kind: str, freq: Optional[str] = None,
            sessions: Optional[Sequence[str]] = None) -> bool:
        """kind in {'bars','quotes','options','actions','borrow','fundamentals','events'}.
        symbol '*' asks whether ANY data of that kind exists (events: freq = event kind such as
        EARNINGS / INDEX_ADD / INDEX_DEL / FOMC / BUYBACK / HALT; fundamentals: freq = field).
        Symbols as returned by universe() (Yahoo style, renamed tickers) resolve through the
        spliced daily history (Yahoo + DoltHub + CIK alias)."""
        any_sym = symbol in ("*", None, "")
        if kind == "events":
            k = (freq or "").upper()
            if k == "EARNINGS_RELEASE":
                srcs_ok = bool(self.store.symbols("events", "sec"))
            else:
                srcs_ok = any(self.store.exists(*t) for t in self.EVENT_SOURCES.get(k, []))
            if not srcs_ok:
                return False                      # e.g. BUYBACK, HALT: no source integrated
            if any_sym or k in ("FOMC", "FOMC_SCHEDULE"):
                return True
            return symbol.upper() in self._event_symbols(k)
        if kind == "fundamentals":
            fields = self._fundamental_fields(None if any_sym else symbol)
            return bool(fields) and (freq is None or freq in fields)
        if any_sym:
            if kind == "bars":
                f = freq or "1d"
                return any(self.store.symbols(f"bars_{f}", s) for s in self.store.sources(f"bars_{f}"))
            if kind == "options":
                return bool(self._option_sources("*", freq))
            if kind == "quotes":
                return any(self.store.symbols("quotes", s) for s in QUOTE_SOURCES)
            if kind == "actions":
                return bool(self.store.symbols("actions", "yahoo"))
            return kind == "borrow" and bool(self.store.symbols("borrow", "iborrowdesk"))
        if kind == "bars":
            f = freq or "1d"
            if f == "1d":
                return len(self._identity_any(symbol)) > 0
            src = self._bar_source(symbol, f) or (self._bar_source(symbol, "1m") if f != "1m" else None)
            if src is None:
                return False
            if sessions:
                fr = f if self._bar_source(symbol, f) else "1m"
                tags = self._session_tags(symbol, fr, self._bar_source(symbol, fr))
                return set(sessions) <= tags
            return True
        if kind == "quotes":
            return any(self.store.exists("quotes", s, symbol) for s in QUOTE_SOURCES)
        if kind == "options":
            return bool(self._option_sources(symbol, freq))
        if kind == "actions":
            return len(self._actions_frame(symbol)) > 0
        if kind == "borrow":
            return self.store.exists("borrow", "iborrowdesk", symbol)
        return False

    def _session_tags(self, symbol: str, freq: str, src: str) -> set:
        key = ("session_tags", symbol.upper(), freq, src)
        if key not in self._cache:
            b = self._read(f"bars_{freq}", src, symbol)
            self._cache[key] = set(b["session"].unique()) if len(b) else set()
        return self._cache[key]

    def fidelity_of(self, symbol: str, kind: str, freq: Optional[str] = None,
                    sessions: Optional[Sequence[str]] = None) -> Fidelity:
        if kind == "bars" and sessions and freq and freq != "1d" and not self.has(symbol, "bars", freq, sessions):
            return Fidelity.NA
        if kind == "bars":
            src = self._bar_source(symbol, freq or "1d")
            if src is None and (freq or "1d") not in ("1d", "1m"):
                src = self._bar_source(symbol, "1m")
            if src is None and freq and freq != "1d" and any(
                    self.store.exists(f"bars_{freq}", s, symbol) or self.store.exists("bars_1m", s, symbol)
                    for s in PROXY_INTRADAY_SOURCES):
                return Fidelity.T2_PROXY       # only CFD proxy bars exist (opt-in)
            return SOURCE_FIDELITY.get(src, Fidelity.T1_BAR) if src else Fidelity.NA
        if kind == "quotes":
            for s in QUOTE_SOURCES:
                if self.store.exists("quotes", s, symbol):
                    return QUOTE_FIDELITY[s]
            return Fidelity.NA
        if kind == "options":
            srcs = self._option_sources(symbol, freq)
            if srcs:
                return min(OPTION_SOURCE_FIDELITY.get(src, Fidelity.T2_PROXY) for _, src in srcs)
            # no real chains: the synthetic fallback (EOD only) is T3 when the underlying has bars
            if freq in (None, "1d", "EOD", "eod") and symbol not in ("*", None, "") and self.has(symbol, "bars", "1d"):
                return Fidelity.T3_MODELED
            return Fidelity.NA
        if kind == "actions":
            return Fidelity.T2_PROXY if self.has(symbol, "actions") else Fidelity.NA
        if kind == "borrow":
            return Fidelity.T1_BAR if self.has(symbol, "borrow") else Fidelity.T2_PROXY
        if kind == "fundamentals":
            return Fidelity.T1_BAR if self.has(symbol, "fundamentals", freq) else Fidelity.NA
        if kind == "events":
            k = (freq or "").upper()
            if not self.has(symbol, "events", k):
                return Fidelity.NA
            return {"EARNINGS": Fidelity.T1_BAR, "EARNINGS_RELEASE": Fidelity.T1_BAR, "FOMC": Fidelity.T1_BAR}.get(
                k, Fidelity.T2_PROXY)
        return Fidelity.NA

    # ================================================================ actions & events
    def _actions_frame(self, symbol: str, date=None, ident: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Actions of the security `symbol` denotes on `date`: they follow the verified price
        segments (identity layer); ETFs/indices use their own (reclassified) Yahoo actions."""
        f = ident if ident is not None else self._identity_frame(symbol, date)
        a = f.attrs.get("actions")
        return a if a is not None else pd.DataFrame()

    def _dolthub_actions(self, symbol: str) -> pd.DataFrame:
        """Fallback for symbols Yahoo lacks (e.g. delisted): DoltHub dividend/split tables.
        DoltHub dividend amounts are split-adjusted to the latest basis (AAPL 1987 $0.12 is
        stored as 0.00214); we un-adjust with DoltHub splits, which only start in 2014 ->
        pre-2014 amounts may be wrong for names that split before 2014 (fidelity T3)."""
        key = ("dolt_actions", symbol.upper())
        if key in self._cache:
            return self._cache[key]
        s = symbol.upper().replace("-", ".")
        dv = self._read("reference", "dolthub", "dividend")
        sp = self._read("reference", "dolthub", "split")
        rows = []
        if len(sp):
            g = sp[sp["act_symbol"] == s]
            for r in g.itertuples():
                ex = pd.Timestamp(r.ex_date)
                if not float(r.for_factor) or not float(r.to_factor):
                    continue                     # malformed DoltHub split row (0 factor)
                rows.append({"symbol": symbol.upper(), "type": "SPLIT", "ex_date": ex,
                             "known_at": _prev_close(ex), "ratio": float(r.to_factor) / float(r.for_factor),
                             "amount": 0.0, "fidelity": int(Fidelity.T2_PROXY)})
        if len(dv):
            g = dv[dv["act_symbol"] == s]
            spl = pd.DataFrame([r for r in rows if r["type"] == "SPLIT"])
            for r in g.itertuples():
                ex = pd.Timestamp(r.ex_date)
                f = 1.0
                if len(spl):
                    f = float(np.prod(spl.loc[spl["ex_date"] > ex, "ratio"])) if (spl["ex_date"] > ex).any() else 1.0
                rows.append({"symbol": symbol.upper(), "type": "CASH_DIVIDEND", "ex_date": ex,
                             "known_at": _prev_close(ex), "ratio": 1.0, "amount": float(r.amount) * f,
                             "fidelity": int(Fidelity.T2_PROXY if ex >= pd.Timestamp("2014-03-28") else Fidelity.T3_MODELED)})
        out = pd.DataFrame(rows)
        self._cache[key] = out
        return out

    def corporate_actions(self, symbols: Optional[Iterable[str]], start, end, as_of,
                          _ident: Optional[pd.DataFrame] = None) -> list[CorporateAction]:
        """Actions with ex_date in [start, end] and known_at <= as_of for the security each symbol
        denotes at min(end, as_of) (fail closed: none if its identity is unknown)."""
        as_of = _utc(as_of)
        s0, e0 = _naive_date(start), _naive_date(end)
        rdate = min(e0, _session_date(as_of))
        syms = list(symbols) if symbols is not None else self.store.symbols("actions", "yahoo")
        out: list[CorporateAction] = []
        for sym in syms:
            ident = _ident if _ident is not None else self._identity_frame(sym, rdate)
            a = self._actions_frame(sym, rdate, ident)
            if a is not None and len(a):
                ex = pd.to_datetime(a["ex_date"])
                m = (ex >= s0) & (ex <= e0) & (pd.to_datetime(a["known_at"], utc=True) <= as_of)
                for r in a[m].to_dict("records"):
                    ratio = r.get("ratio", 1.0)
                    out.append(CorporateAction(
                        symbol=sym.upper(), type=ActionType(r["type"]), ex_date=pd.Timestamp(r["ex_date"]).normalize(),
                        known_at=pd.Timestamp(r["known_at"]),
                        ratio=float(ratio) if ratio is not None and np.isfinite(float(ratio)) else 1.0,
                        amount=float(r.get("amount", 0.0) or 0.0),
                        pay_date=None if pd.isna(r.get("pay_date", pd.NaT)) else pd.Timestamp(r["pay_date"]),
                        new_symbol=r.get("new_symbol") if isinstance(r.get("new_symbol"), str) else None,
                        fidelity=Fidelity(int(r.get("fidelity", 2) if pd.notna(r.get("fidelity", 2)) else 2))))
            out.extend(self._delist_actions(sym, s0, e0, as_of, ident))
        return sorted(out, key=lambda x: (x.ex_date, x.symbol))

    def _delist_actions(self, sym, s0, e0, as_of, ident=None) -> list[CorporateAction]:
        from .survivorship import delisting_for
        d = delisting_for(self, sym, ident)
        if d is None:
            return []
        ex, known, px = d
        if s0 <= ex <= e0 and known <= as_of:
            return [CorporateAction(symbol=sym.upper(), type=ActionType.DELIST, ex_date=ex, known_at=known,
                                    amount=px, fidelity=Fidelity.T2_PROXY)]
        return []

    def events(self, kind: str, symbols: Optional[Iterable[str]], start, end, as_of) -> list[MarketEvent]:
        """Events with known_at in [start, end] and known_at <= as_of."""
        from . import events as ev
        df = ev.load_events(self, kind, symbols, _utc(start), _utc(end))
        if df.empty:
            return []
        as_of = _utc(as_of)
        df = df[df["known_at"] <= as_of]
        out = []
        base = {"kind", "symbol", "known_at", "effective_at", "fidelity", "source"}
        for r in df.to_dict("records"):
            payload = {k: (None if (not isinstance(v, (list, dict, str)) and pd.isna(v)) else v)
                       for k, v in r.items() if k not in base}
            if payload.get("estimate_fidelity") is not None:
                payload["estimate_fidelity"] = int(payload["estimate_fidelity"])
            # an estimate known after as_of must not leak
            eka = payload.get("estimate_known_at")
            if eka is not None and pd.Timestamp(eka) > as_of:
                payload["eps_estimate"], payload["estimate_known_at"] = None, None
            out.append(MarketEvent(kind=r["kind"], symbol=r.get("symbol"), known_at=pd.Timestamp(r["known_at"]),
                                   effective_at=None if pd.isna(r.get("effective_at")) else pd.Timestamp(r["effective_at"]),
                                   payload={**payload, "source": r.get("source")},
                                   fidelity=Fidelity(int(r.get("fidelity", 1)))))
        return out

    # ================================================================ options
    MAX_CHAIN_STALE_DAYS = 4.0

    def _chain_frame(self, underlying: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, str]:
        """Real chain rows (annotated) for the latest year(s) <= as_of; '' source if none."""
        from .option_meta import annotate_chain
        for src in ("dolthub", "hod"):
            if not self.store.exists("option_chain", src, underlying):
                continue
            y = as_of.year
            for yr in (y, y - 1):
                key = ("chain_year", src, underlying, yr)
                if key not in self._cache:
                    raw = self.store.read("option_chain", src, underlying, start=pd.Timestamp(f"{yr}-01-01", tz="UTC"),
                                          end=pd.Timestamp(f"{yr + 1}-01-01", tz="UTC"), time_col="ts")
                    self._cache[key] = annotate_chain(raw.reset_index(drop=True)) if len(raw) else raw
                df = self._cache[key]
                df = df[df["ts"] <= as_of] if len(df) else df
                if len(df):
                    return df, src
        return pd.DataFrame(), ""

    def option_chain(self, underlying: str, as_of, min_dte: int = 0, max_dte: int = 400,
                     synthetic: bool = True) -> pd.DataFrame:
        """Latest REAL chain snapshot with ts <= as_of (fidelity column T2_PROXY). If there is
        none (underlying not covered, or the latest snapshot is older than
        MAX_CHAIN_STALE_DAYS) and `synthetic` is True, a SYNTHETIC chain from
        atlasbt.options.synthetic is returned instead: fidelity column T3_MODELED and
        attrs['synthetic'] = True. Real and synthetic rows are never mixed in one frame."""
        as_of = _utc(as_of)
        u = underlying.upper()
        df, src = self._chain_frame(u, as_of)
        if not df.empty:
            ts = df["ts"].max()
            if (as_of - ts).total_seconds() / 86400 <= self.MAX_CHAIN_STALE_DAYS or not synthetic:
                return self._finish_real_chain(u, df[df["ts"] == ts].copy(), ts, as_of, src, min_dte, max_dte)
        if synthetic:
            return self.synthetic_chain(u, as_of, min_dte, max_dte)
        return frames.empty_chain()

    def _finish_real_chain(self, u, snap, ts, as_of, src, min_dte, max_dte) -> pd.DataFrame:
        sd = _session_date(ts)
        dte = (pd.to_datetime(snap["expiry"]) - sd).dt.days
        snap = snap[(dte >= min_dte) & (dte <= max_dte)].copy()
        if snap["underlying_price"].isna().any():
            px = self._close_at(u, ts)
            miss = snap["underlying_price"].isna()
            snap.loc[miss, "underlying_price"] = px
            snap.loc[miss, "underlying_price_source"] = "pit_bar_close_at_ts"
        cols = [c for c in frames.CHAIN_COLS if c in snap.columns] + \
            [c for c in ("root", "settlement_date", "kind", "underlying_price_source") if c in snap.columns]
        snap = snap[cols].reset_index(drop=True)
        snap.attrs.update({"source": src, "fidelity": int(Fidelity.T2_PROXY), "synthetic": False, "snapshot_ts": ts,
                           "stale_days": (as_of - ts).total_seconds() / 86400,
                           "note": "real EOD quotes stamped at the session close (capture minute unknown); "
                                   "underlying_price = PIT raw bar close at ts unless vendor-supplied"})
        return snap

    # ---------------------------------------------------------------- synthetic (T3)
    def _spot(self, u: str, as_of: pd.Timestamp) -> float:
        """Last raw price known at as_of: 1m close if intraday data exists, else daily close
        (Cboe index level for SPX/NDX/RUT...)."""
        sym = {"SPXW": "SPX"}.get(u, u)
        if self._bar_source(sym, "1m"):
            b = self.bars(sym, "1m", as_of - pd.Timedelta(days=5), as_of, as_of, sessions=("RTH",))
            if len(b):
                return float(b["close"].iloc[-1])
        src = self._bar_source(sym, "1d")
        if not src:
            return float("nan")
        b = self._read("bars_1d", src, sym)
        b = b[b["known_at"] <= as_of]
        return float(b["close"].iloc[-1]) if len(b) else float("nan")

    def _div_yield(self, u: str, as_of: pd.Timestamp, spot: float) -> float:
        sym = "SPY" if u in ("SPX", "SPXW", "XSP") else u
        acts = self.corporate_actions([sym], _session_date(as_of) - pd.Timedelta(days=365), _session_date(as_of), as_of)
        # regular cash dividends only: specials and spin-off values are not a recurring yield
        d = sum(a.amount for a in acts if a.type == ActionType.CASH_DIVIDEND)
        ref = self._spot(sym, as_of) if sym != u else spot
        return float(d / ref) if ref and np.isfinite(ref) and ref > 0 else 0.0

    def _rv_ratio(self, u: str, as_of: pd.Timestamp, n: int = 63) -> float:
        def rv(sym):
            src = self._bar_source(sym, "1d")
            if not src:
                return float("nan")
            b = self.bars(sym, "1d", as_of - pd.Timedelta(days=200), as_of, as_of, adjusted=True)
            r = np.log(b["close"]).diff().dropna().tail(n)
            return float(r.std()) if len(r) >= n // 2 else float("nan")
        a, m = rv(u), rv("SPY")
        return a / m if np.isfinite(a) and np.isfinite(m) and m > 0 else float("nan")

    def synthetic_chain(self, underlying: str, as_of, min_dte: int = 0, max_dte: int = 400) -> pd.DataFrame:
        """atlasbt.options.synthetic chain (fidelity T3_MODELED, attrs['synthetic']=True) built
        from PIT inputs only: spot (last raw price), r (3M T-bill money-market -> continuous),
        q (trailing-12m dividends / spot), rv_ratio (63d realized vol vs SPY) for unmapped names."""
        from atlasbt.options.synthetic import INDEX_MAP, SPX_LIKE, SyntheticChainGenerator
        as_of = _utc(as_of)
        u = underlying.upper()
        if "synthetic_gen" not in self._cache:
            self._cache["synthetic_gen"] = SyntheticChainGenerator()
        gen = self._cache["synthetic_gen"]
        spot = self._spot(u, as_of)
        if not np.isfinite(spot):
            out = frames.empty_chain()
            out.attrs.update(fidelity=int(Fidelity.NA), synthetic=True, reason=f"no PIT spot for {u}")
            return out
        try:
            r = rates.money_market_to_continuous(self.risk_free_rate(as_of, "3M"), 91)
        except DataUnavailableError:
            r = 0.0
        q = self._div_yield(u, as_of, spot)
        rvr = None if (u in SPX_LIKE or u in INDEX_MAP) else self._rv_ratio(u, as_of)
        ch = gen.chain(u, as_of, spot, r, q, max_dte=max_dte, min_dte=min_dte, rv_ratio=rvr)
        ch["underlying_price_source"] = "pit_last_price"
        ch.attrs.update(synthetic=True, inputs={"spot": spot, "r_cont": r, "q": q, "rv_ratio": rvr},
                        fidelity=int(ch.attrs.get("fidelity", Fidelity.T3_MODELED)))
        return ch

    def _close_at(self, symbol: str, ts: pd.Timestamp) -> float:
        src = self._bar_source(symbol, "1d")
        if not src:
            return float("nan")
        b = self._read("bars_1d", src, symbol)
        hit = b[b["known_at"] == ts]
        if len(hit):
            return float(hit["close"].iloc[-1])
        b = b[b["known_at"] <= ts]
        return float(b["close"].iloc[-1]) if len(b) else float("nan")

    @staticmethod
    def _match_instrument(df: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
        """Rows of `df` for `instrument`: by OCC symbol, else by (root, expiry=last trading day,
        strike, right) so instruments built with Instrument.option(...) from the annotated
        chain match vendor OCC symbols (which carry the settlement date for SPX AM)."""
        hit = df[df["symbol"] == instrument.symbol]
        if len(hit) or instrument.expiry is None:
            return hit
        root = instrument.symbol[:6].strip()
        m = (pd.to_datetime(df["expiry"]) == pd.Timestamp(instrument.expiry).normalize()) & \
            (np.isclose(df["strike"].astype(float), float(instrument.strike))) & (df["right"] == instrument.right.value)
        if "root" in df.columns:
            m &= df["root"] == root
        return df[m]

    def option_quote(self, instrument: Instrument, as_of, max_stale_days: float = 4.0) -> Optional[OptionQuote]:
        """Latest REAL quote (T2) for the contract with ts <= as_of, or None (stale/missing).
        Synthetic marks are the OPTIONS agent's job (atlasbt.options.marking)."""
        as_of = _utc(as_of)
        df, src = self._chain_frame((instrument.underlying or "").upper(), as_of)
        if df.empty:
            return None
        hit = self._match_instrument(df, instrument)
        if hit.empty:
            return None
        r = hit.sort_values("ts").iloc[-1]
        if (as_of - r["ts"]).total_seconds() / 86400 > max_stale_days:
            return None
        und = r["underlying_price"] if pd.notna(r["underlying_price"]) else self._close_at(instrument.underlying, r["ts"])
        return OptionQuote(instrument=instrument, ts=pd.Timestamp(r["ts"]), bid=float(r["bid"]), ask=float(r["ask"]),
                           underlying_price=float(und), iv=float(r["iv"]), delta=float(r["delta"]),
                           gamma=float(r["gamma"]), vega=float(r["vega"]), theta=float(r["theta"]),
                           volume=float(r["volume"]), open_interest=float(r["open_interest"]),
                           fidelity=Fidelity(int(r["fidelity"])))

    def _option_sources(self, symbol: str, freq: Optional[str]) -> list[tuple[str, str]]:
        """[(store kind, source)] holding REAL option data for `symbol` ('*' = any) at `freq`.
        EOD (None / '1d'): kind option_chain. Intraday ('1m', '5m', ...): kind option_bars_<freq>
        and timestamped quote snapshots (kind option_quotes, e.g. Databento OPRA when keyed)."""
        eod = freq in (None, "1d", "1D", "EOD", "eod")
        kinds = ["option_chain"] if eod else [f"option_bars_{freq}", "option_quotes"]
        out = []
        for k in kinds:
            for src in self.store.sources(k):
                if symbol in ("*", None, ""):
                    ok = bool(self.store.symbols(k, src))
                else:
                    ok = self.store.exists(k, src, symbol) or self.store.exists(k, src, symbol.upper())
                if ok:
                    out.append((k, src))
        return out

    def option_quotes(self, instrument: Instrument, start, end, as_of, synthetic: bool = True) -> pd.DataFrame:
        """REAL snapshots (see _real_option_quotes). If there are none in the window and
        `synthetic`, atlasbt.options.marking.synthetic_option_quotes supplies session-close
        snapshots (chain-anchored marks T2_PROXY / pure model T3_MODELED, `reason` column,
        attrs['synthetic']=True). Real and synthetic rows are never mixed in one frame."""
        real = self._real_option_quotes(instrument, start, end, as_of)
        if len(real) or not synthetic:
            return real
        try:
            from atlasbt.options.marking import synthetic_option_quotes
        except ImportError:
            return real
        syn = synthetic_option_quotes(instrument, start, end, as_of, self, real_first=False)
        syn.attrs["synthetic"] = True
        return syn

    def _real_option_quotes(self, instrument: Instrument, start, end, as_of) -> pd.DataFrame:
        """All REAL snapshots of `instrument` with ts in (start, end] and ts <= as_of, ascending,
        indexed by ts, option_chain columns (+ root, settlement_date, kind, source). DoltHub
        snapshots are stamped at the session close (capture minute unknown; T2_PROXY)."""
        from .option_meta import annotate_chain
        start, end, as_of = _utc(start), _utc(end), _utc(as_of)
        hi = min(end, as_of)
        und = (instrument.underlying or "").upper()
        parts = []
        for src in ("dolthub", "hod"):
            if not self.store.exists("option_chain", src, und):
                continue
            for yr in range(start.year, hi.year + 1):
                key = ("chain_year", src, und, yr)
                if key not in self._cache:
                    raw = self.store.read("option_chain", src, und, start=pd.Timestamp(f"{yr}-01-01", tz="UTC"),
                                          end=pd.Timestamp(f"{yr + 1}-01-01", tz="UTC"), time_col="ts")
                    self._cache[key] = annotate_chain(raw.reset_index(drop=True)) if len(raw) else raw
                df = self._cache[key]
                if len(df):
                    df = df[(df["ts"] > start) & (df["ts"] <= hi)]
                    df = self._match_instrument(df, instrument) if len(df) else df
                    if len(df):
                        parts.append(df.assign(source=src))
        extra = ["root", "settlement_date", "kind", "underlying_price_source", "source"]
        cols = [c for c in frames.CHAIN_COLS if c != "ts"] + extra
        if not parts:
            out = pd.DataFrame(columns=cols)
            out.index = pd.DatetimeIndex([], tz="UTC", name="ts")
            return out
        out = pd.concat(parts, ignore_index=True).sort_values(["ts", "source"])
        out = out.drop_duplicates("ts", keep="first")           # 'dolthub' sorts before 'hod'
        if out["underlying_price"].isna().any():
            out["underlying_price"] = [u if pd.notna(u) else self._close_at(und, t)
                                       for u, t in zip(out["underlying_price"], out["ts"])]
        out = out.set_index(pd.DatetimeIndex(out["ts"], name="ts"))
        out = out[[c for c in cols if c in out.columns]]
        out.attrs.update({"fidelity": int(Fidelity.T2_PROXY), "symbol": instrument.symbol, "synthetic": False})
        return out

    # ================================================================ rates, borrow, fundamentals
    def rate_series(self, series_id: str) -> pd.DataFrame:
        return self._read("rates", "fred", series_id)

    MAX_RATE_STALE_DAYS = 10
    # per requested tenor: series tried in order (first = the native series for that tenor)
    RATE_CHAINS = {"1M": ["1M", "3M"], "3M": ["3M"], "6M": ["6M", "3M"], "1Y": ["1Y", "1Y_CMT", "3M"],
                   "ON": ["ON"], "SOFR": ["SOFR", "ON"], "2Y": ["2Y"], "5Y": ["5Y"], "10Y": ["10Y"], "30Y": ["30Y"]}

    def risk_free_rate_info(self, as_of, tenor: str = "3M") -> tuple[float, str, Fidelity, pd.Timestamp]:
        """(rate, series_used, fidelity, observation_date). Rate as in risk_free_rate().
        fidelity T1_BAR when the native series for `tenor` was used (e.g. DTB4WK for 1M, available
        from 2001-07-31), T2_PROXY when a fallback tenor was substituted (e.g. 3M for 1M before
        2001-07-31, or 3M for 1Y after DTB1YR ended in 2001 if DGS1 is missing). A series whose
        latest value known at as_of is older than MAX_RATE_STALE_DAYS is treated as unavailable."""
        as_of = _utc(as_of)
        t = tenor.upper()
        if t not in self.RATE_CHAINS:
            raise KeyError(f"tenor {tenor!r}; known {list(self.RATE_CHAINS)}")
        stale = None
        for i, tt in enumerate(self.RATE_CHAINS[t]):
            sid, kind, days = RATE_SERIES[tt]
            df = self.rate_series(sid)
            if not len(df):
                continue
            df = df[df["known_at"] <= as_of]
            if not len(df):
                continue
            obs = pd.Timestamp(df.index[-1])
            v = float(df["value"].iloc[-1]) / 100.0
            if kind == "discount":
                r = rates.discount_to_money_market(v, days)
            elif kind == "be":
                r = rates.bond_equivalent_to_money_market(v)
            else:
                r = v
            if (_session_date(as_of) - obs).days > self.MAX_RATE_STALE_DAYS:
                if stale is None or obs > stale[3]:
                    stale = (r, sid, Fidelity.T2_PROXY, obs)
                continue
            return r, sid, (Fidelity.T1_BAR if i == 0 else Fidelity.T2_PROXY), obs
        if stale is not None:          # every candidate is stale: freshest value, flagged T2
            return stale
        raise DataUnavailableError(f"no {tenor} rate known at {as_of}")

    def risk_free_rate(self, as_of, tenor: str = "3M") -> float:
        """Simple annualized money-market (ACT/360) rate in decimals, known at as_of.
        T-bill discount quotes are converted (atlasbt.data.rates); DGS* are bond-equivalent
        and returned rebased to 360 days; DFF/SOFR are already ACT/360. Fallback tenors are
        possible -- use risk_free_rate_info() to see which series was used and its fidelity."""
        return self.risk_free_rate_info(as_of, tenor)[0]

    def borrow(self, symbol: str, as_of) -> tuple[bool, float, Fidelity]:
        from .providers import borrow as B
        if self.has(symbol, "borrow"):
            b = self._read("borrow", "iborrowdesk", symbol)
            if len(b) and "fee" in b.columns and "time" in b.columns:
                t = pd.to_datetime(b["time"], utc=True)
                b = b[t <= _utc(as_of)]
                if len(b):
                    fee = float(b["fee"].iloc[-1]) / 100.0
                    avail = float(b.get("available", pd.Series([1])).iloc[-1]) > 0
                    return avail, fee, Fidelity.T1_BAR
        member = False
        try:
            member = symbol.upper() in set(self.universe("SP500", as_of))
        except Exception:
            pass
        return B.default_fee(symbol, is_etf=self._is_etf(symbol), is_index_member=member)

    def fundamentals(self, symbols: Optional[Iterable[str]], fields: Sequence[str], as_of) -> pd.DataFrame:
        """Latest value per (symbol, field) among facts filed (accepted) at or before as_of:
        the most recent period_end, and for that period the most recently filed value."""
        as_of = _utc(as_of)
        syms = list(symbols) if symbols is not None else self.store.symbols("fundamentals", "sec")
        parts = []
        for s in syms:
            f = self._read("fundamentals", "sec", s)
            if not len(f):
                continue
            ka = pd.to_datetime(f["known_at"], utc=True)
            f = f[f["field"].isin(list(fields)) & (ka <= as_of)]
            # a fact cannot describe a period ending after it was filed (mis-tagged XBRL contexts,
            # e.g. AAL shares_outstanding tagged period_end 2027-07-17 in a 2026 filing)
            pe = pd.to_datetime(f["period_end"])
            kd = pd.to_datetime(f["known_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
            f = f[pe <= kd + pd.Timedelta(days=7)]
            if not len(f):
                continue
            f = f.sort_values(["field", "period_end", "known_at"])
            parts.append(f.groupby("field", as_index=False).tail(1))
        cols = ["symbol", "field", "value", "period_end", "known_at", "source", "period_start", "fy", "fp", "form",
                "accn", "unit", "concept"]
        if not parts:
            return pd.DataFrame(columns=cols)
        out = pd.concat(parts, ignore_index=True)
        out = out[[c for c in cols if c in out.columns]].reset_index(drop=True)
        out.attrs.update({"source": "sec", "fidelity": int(Fidelity.T1_BAR)})
        return out

    def fundamentals_history(self, symbols: Optional[Iterable[str]], fields: Sequence[str], as_of,
                             start=None) -> pd.DataFrame:
        """EVERY filed value (all periods, all filings incl. restatements and prior-year
        comparatives) with known_at <= as_of [and period_end >= start], sorted by symbol, field,
        period_end, known_at. Columns: symbol, field, value, period_start, period_end,
        fiscal_period, fy, form, known_at, accn, concept, duration_days, source.
        To rebuild a PIT TTM / YoY value at time t, use rows with known_at <= t and, per period,
        the latest known_at (that is what fundamentals() does for the latest period)."""
        as_of = _utc(as_of)
        syms = list(symbols) if symbols is not None else self.store.symbols("fundamentals", "sec")
        st = pd.Timestamp(start) if start is not None else None
        if st is not None and st.tzinfo is not None:
            st = st.tz_convert(ET).tz_localize(None)
        parts = []
        for s in syms:
            f = self._read("fundamentals", "sec", s)
            if not len(f):
                continue
            m = f["field"].isin(list(fields)) & (pd.to_datetime(f["known_at"], utc=True) <= as_of)
            if st is not None:
                m &= pd.to_datetime(f["period_end"]) >= st
            if m.any():
                parts.append(f[m])
        cols = ["symbol", "field", "value", "period_start", "period_end", "fiscal_period", "fy", "form", "known_at",
                "accn", "concept", "duration_days", "source"]
        if not parts:
            return pd.DataFrame(columns=cols)
        out = pd.concat(parts, ignore_index=True).rename(columns={"fp": "fiscal_period"})
        out = out[[c for c in cols if c in out.columns]]
        out = out.sort_values(["symbol", "field", "period_end", "known_at"]).reset_index(drop=True)
        out.attrs.update({"source": "sec", "fidelity": int(Fidelity.T1_BAR)})
        return out


def _prev_close(ex_date) -> pd.Timestamp:
    from .providers.yahoo import conservative_known_at
    return conservative_known_at(pd.Timestamp(ex_date))


def resample_bars(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Causally aggregate 1m canonical bars into `freq` bars anchored at each session's
    open (RTH) / at clock multiples (PRE/POST). Bars never span sessions or session parts."""
    if df.empty:
        return df
    td = frames.freq_to_timedelta(freq)
    d = frames.to_naive_date(df.index)
    sch = frames.schedule().reindex(d)
    opens = pd.DatetimeIndex(sch["open"].to_numpy())
    st = df.index
    # bucket start: open + k*td for RTH, floor to td for PRE/POST
    rth = (df["session"] == "RTH").to_numpy()
    k = ((st - opens) // td).to_numpy()
    rth_bucket = opens + pd.to_timedelta(k * td.value, unit="ns")
    other_bucket = st.floor(td)
    bucket = pd.DatetimeIndex(np.where(rth, rth_bucket.asi8, other_bucket.asi8)).tz_localize("UTC")
    g = df.assign(_b=bucket, _s=df["session"].to_numpy())
    agg = g.groupby(["_b", "_s"], sort=True).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                 close=("close", "last"), volume=("volume", "sum"),
                                                 _last=("known_at", "max"))
    agg = agg.reset_index().rename(columns={"_b": "start", "_s": "session"}).set_index("start")
    closes = pd.DatetimeIndex(frames.schedule().reindex(frames.to_naive_date(agg.index))["close"].to_numpy())
    end = agg.index + td
    # a bucket is complete (known) at its nominal end, capped at the session close for RTH
    known = pd.DatetimeIndex(np.where((agg["session"] == "RTH").to_numpy() & (end > closes), closes.asi8, end.asi8)).tz_localize("UTC")
    agg["known_at"] = known
    return agg.drop(columns=["_last"])[["open", "high", "low", "close", "volume", "session", "known_at"]]

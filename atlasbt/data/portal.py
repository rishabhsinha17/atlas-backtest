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
            syms = [y if isinstance(y, str) and y else str(b).replace(".", "-")
                    for y, b in zip(cur["yahoo"], cur["base"])]
            return sorted(set(syms))
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
            d = self._daily_frame(s)
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
        if freq == "1d" and source is None:
            spliced = self._daily_frame(symbol)
            if len(spliced):
                native = "spliced"
                alias = spliced.attrs.get("alias_of")
            else:
                native = self._bar_source(symbol, freq, source)     # e.g. cboe / tiingo
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
            df = self._daily_frame(symbol) if native == "spliced" else self._read(f"bars_{freq}", native, symbol)
            src = df.attrs.get("source") or native if native == "spliced" else native
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
            out = self._adjust(symbol, out, as_of, src)
        out.attrs.update({"source": src, "fidelity": int(SOURCE_FIDELITY.get(str(src).split("+")[0], Fidelity.T1_BAR)),
                          "adjusted": adjusted, "symbol": symbol.upper(), "freq": freq})
        if alias:
            # same-CIK renamed ticker spliced in: identity resolved via ZipLime CIKs -> T2
            out.attrs["alias_of"] = alias
            out.attrs["fidelity"] = int(Fidelity.T2_PROXY)
        return out

    def _adjust(self, symbol: str, bars: pd.DataFrame, as_of: pd.Timestamp, src: str) -> pd.DataFrame:
        acts = self.corporate_actions([symbol], pd.Timestamp("1900-01-01"), _session_date(as_of), as_of)
        acts = [a for a in acts if _naive_date(a.ex_date) <= _session_date(as_of)]
        if not acts:
            return bars
        spliced = self._daily_frame(symbol)
        daily_src = "spliced" if len(spliced) else self._bar_source(symbol, "1d")
        if daily_src:
            daily = spliced if daily_src == "spliced" else self._read("bars_1d", daily_src, symbol)
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

    def alias_for(self, symbol: str) -> Optional[str]:
        """Same-company ticker with daily data when `symbol` itself has none (renames such as
        UTX->RTX, HRS->LHX, KORS->CPRI), resolved through the CIK of ZipLime's as-of tickers.
        Returns None if no alias with data exists. Served bars carry attrs['alias_of']."""
        key = ("alias", symbol.upper())
        if key in self._cache:
            return self._cache[key]
        res = None
        if "cik_groups" not in self._cache:
            z = self.store.read("events", "ziplime", "EARNINGS", columns=["symbol", "cik", "known_at"])
            if len(z):
                z = z[z["cik"].notna()].sort_values("known_at")
                self._cache["cik_groups"] = (z.groupby("symbol")["cik"].last().to_dict(),
                                             z.groupby("cik")["symbol"].apply(lambda x: list(dict.fromkeys(x[::-1]))).to_dict())
            else:
                self._cache["cik_groups"] = ({}, {})
        sym2cik, cik2syms = self._cache["cik_groups"]
        c = sym2cik.get(symbol.upper())
        if c is not None:
            for alt in cik2syms.get(c, []):
                if alt != symbol.upper() and (self.store.exists("bars_1d", "yahoo", alt)
                                              or self.store.exists("bars_1d", "dolthub", alt)):
                    res = alt
                    break
        self._cache[key] = res
        return res

    def _daily_frame(self, symbol: str, _alias: bool = True) -> pd.DataFrame:
        """Spliced RAW daily history: Yahoo, then DoltHub stocks.ohlcv for sessions Yahoo lacks
        (Yahoo keeps only stubs for some delisted/merged names, e.g. AVB), then the same
        company's renamed ticker (`alias_for`, CIK-resolved) for sessions neither has.
        attrs['source'] lists the parts used (e.g. 'yahoo+dolthub')."""
        key = ("daily_spliced", symbol.upper(), _alias)
        if key in self._cache:
            return self._cache[key]
        parts, used = [], []
        for src in ("yahoo", "dolthub"):
            if self.store.exists("bars_1d", src, symbol):
                f = self._read("bars_1d", src, symbol)
                if len(f):
                    parts.append(f)
                    used.append(src)
        al = self.alias_for(symbol) if _alias else None
        if al:
            f = self._daily_frame(al, _alias=False)
            if len(f):
                parts.append(f)
                used.append(f"alias:{al}")
        if not parts:
            out = frames.empty_bars()
        elif len(parts) == 1:
            out = parts[0]
        else:
            cols = [c for c in parts[0].columns]
            out = pd.concat([p_.reindex(columns=cols) for p_ in parts])
            out = out[~out.index.duplicated(keep="first")].sort_index()
        out = out.copy()
        out.attrs["source"] = "+".join(used) if used else None
        out.attrs["alias_of"] = al if (al and any(u.startswith("alias") for u in used)) else None
        self._cache[key] = out
        return out

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
                     "FOMC": [("events", "fomc", "FOMC")]}

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
            if any_sym or k == "FOMC":
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
                return bool(self.store.symbols("option_chain", "dolthub"))
            if kind == "quotes":
                return any(self.store.symbols("quotes", s) for s in QUOTE_SOURCES)
            if kind == "actions":
                return bool(self.store.symbols("actions", "yahoo"))
            return kind == "borrow" and bool(self.store.symbols("borrow", "iborrowdesk"))
        if kind == "bars":
            f = freq or "1d"
            if f == "1d":
                return len(self._daily_frame(symbol)) > 0 or self._bar_source(symbol, "1d") is not None
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
            return self.store.exists("option_chain", "dolthub", symbol) or \
                self.store.exists("option_chain", "hod", symbol)
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
            return Fidelity.T2_PROXY if self.has(symbol, "options") else Fidelity.NA
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
    def _actions_frame(self, symbol: str) -> pd.DataFrame:
        a = self._read("actions", "yahoo", symbol)
        if len(a):
            return a
        d = self._dolthub_actions(symbol)
        if len(d):
            return d
        al = self.alias_for(symbol)
        if al:
            a = self._read("actions", "yahoo", al)
            if len(a):
                return a.assign(symbol=symbol.upper())
        return d

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

    def corporate_actions(self, symbols: Optional[Iterable[str]], start, end, as_of) -> list[CorporateAction]:
        as_of = _utc(as_of)
        s0, e0 = _naive_date(start), _naive_date(end)
        syms = list(symbols) if symbols is not None else self.store.symbols("actions", "yahoo")
        out: list[CorporateAction] = []
        for sym in syms:
            a = self._actions_frame(sym)
            if a is None or not len(a):
                pass
            else:
                ex = pd.to_datetime(a["ex_date"])
                m = (ex >= s0) & (ex <= e0) & (pd.to_datetime(a["known_at"], utc=True) <= as_of)
                for r in a[m].to_dict("records"):
                    out.append(CorporateAction(
                        symbol=sym.upper(), type=ActionType(r["type"]), ex_date=pd.Timestamp(r["ex_date"]).normalize(),
                        known_at=pd.Timestamp(r["known_at"]), ratio=float(r.get("ratio", 1.0) or 1.0),
                        amount=float(r.get("amount", 0.0) or 0.0),
                        pay_date=None if pd.isna(r.get("pay_date", pd.NaT)) else pd.Timestamp(r["pay_date"]),
                        new_symbol=r.get("new_symbol") or None, fidelity=Fidelity(int(r.get("fidelity", 2)))))
            out.extend(self._delist_actions(sym, s0, e0, as_of))
        return sorted(out, key=lambda x: (x.ex_date, x.symbol))

    def _delist_actions(self, sym, s0, e0, as_of) -> list[CorporateAction]:
        from .survivorship import delisting_for
        d = delisting_for(self, sym)
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
        d = sum(a.amount for a in acts if a.type in (ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND))
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

    def option_quotes(self, instrument: Instrument, start, end, as_of) -> pd.DataFrame:
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
            f = f[f["field"].isin(list(fields)) & (pd.to_datetime(f["known_at"], utc=True) <= as_of)]
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

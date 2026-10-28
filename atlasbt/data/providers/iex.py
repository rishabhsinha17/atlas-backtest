"""IEX HIST (TOPS) -> quotes / trades for selected symbols and days.

Index: https://iextrading.com/api/1.0/hist?date=YYYYMMDD -> [{link, date, feed, version, size}]
Files: pcapng (gzip) of the IEX-TP v1 multicast feed, free since 2016-12-12. TOPS 1.5/1.6 =
IEX top-of-book (QuoteUpdate) + last sale (TradeReport). Recent days are 10-16 GB, early
years ~0.5-2 GB -> download only targeted days; files are streamed, never fully loaded.

Message layouts (little-endian; IEX TOPS spec v1.6):
  IEX-TP segment header (40 B): version u1, reserved u1, protocol id u2, channel id u4,
      session id u4, payload len u2, message count u2, stream offset u8, first seq u8, send time i8
  then `message count` x [length u2][message]
  QuoteUpdate 'Q' (0x51), 42 B: type u1, flags u1, timestamp i8 (ns since epoch), symbol 8s,
      bid size u4, bid price i8 (x1e-4), ask price i8 (x1e-4), ask size u4
  TradeReport 'T' (0x54), 38 B: type u1, sale flags u1, timestamp i8, symbol 8s, size u4,
      price i8 (x1e-4), trade id i8

FIDELITY: IEX is ONE venue (~2-3% of US volume). Its BBO is not the NBBO (IEX quotes are often
wider or absent) -> quotes are T2_PROXY; IEX trades are real prints but a small, non-random
subset of the tape (T2 for volume-based logic).
"""
from __future__ import annotations

import gzip
import logging
import struct
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.iex")
SOURCE = "iex"
HIST_URL = "https://iextrading.com/api/1.0/hist"
_Q = struct.Struct("<BBq8sIqqI")
_T = struct.Struct("<BBq8sIqq")


def _pcapng_packets(fh) -> Iterator[bytes]:
    """Yield raw packet bytes from Enhanced/Simple Packet Blocks of a pcapng stream."""
    while True:
        hdr = fh.read(8)
        if len(hdr) < 8:
            return
        btype, blen = struct.unpack("<II", hdr)
        body = fh.read(blen - 8)
        if btype == 6:                       # Enhanced Packet Block
            caplen = struct.unpack_from("<I", body, 12)[0]
            yield body[20:20 + caplen]
        elif btype == 3:                     # Simple Packet Block
            olen = struct.unpack_from("<I", body, 0)[0]
            yield body[4:4 + olen]


def _udp_payload(pkt: bytes) -> Optional[bytes]:
    off = 14
    eth = struct.unpack_from(">H", pkt, 12)[0]
    if eth == 0x8100:                         # VLAN tag
        eth = struct.unpack_from(">H", pkt, 16)[0]
        off = 18
    if eth != 0x0800:
        return None
    ihl = (pkt[off] & 0x0F) * 4
    if pkt[off + 9] != 17:                    # UDP
        return None
    return pkt[off + ihl + 8:]


def iter_messages(path: str | Path, symbols: Optional[Iterable[str]] = None) -> Iterator[tuple]:
    """Yield ('Q', ts_ns, sym, bid_size, bid, ask, ask_size) and ('T', ts_ns, sym, size, price)."""
    want = {s.upper().encode().ljust(8) for s in symbols} if symbols else None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        for pkt in _pcapng_packets(fh):
            pl = _udp_payload(pkt)
            if not pl or len(pl) < 40:
                continue
            n = struct.unpack_from("<H", pl, 14)[0]
            pos = 40
            for _ in range(n):
                if pos + 2 > len(pl):
                    break
                mlen = struct.unpack_from("<H", pl, pos)[0]
                m = pl[pos + 2:pos + 2 + mlen]
                pos += 2 + mlen
                if not m:
                    continue
                t = m[0]
                if t == 0x51 and mlen >= _Q.size:
                    _, _, ts, sym, bs, bp, ap, asz = _Q.unpack_from(m)
                    if want is None or sym in want:
                        yield ("Q", ts, sym.decode().strip(), bs, bp / 1e4, ap / 1e4, asz)
                elif t == 0x54 and mlen >= _T.size:
                    _, _, ts, sym, sz, px, _tid = _T.unpack_from(m)
                    if want is None or sym in want:
                        yield ("T", ts, sym.decode().strip(), sz, px / 1e4)


def parse_tops(path: str | Path, symbols: Iterable[str]) -> tuple[dict, dict]:
    """-> ({sym: quotes frame}, {sym: trades frame}). Zero prices mean 'no quote on that side'."""
    q, t = {}, {}
    for msg in iter_messages(path, symbols):
        if msg[0] == "Q":
            q.setdefault(msg[2], []).append(msg[1:2] + msg[3:])
        else:
            t.setdefault(msg[2], []).append(msg[1:2] + msg[3:])
    qf = {}
    for s, rows in q.items():
        df = pd.DataFrame(rows, columns=["ts", "bid_size", "bid", "ask", "ask_size"])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ns", utc=True), name="ts")
        df.loc[df["bid"] <= 0, "bid"] = float("nan")
        df.loc[df["ask"] <= 0, "ask"] = float("nan")
        qf[s] = df[["bid", "ask", "bid_size", "ask_size"]].astype(float)
    tf = {}
    for s, rows in t.items():
        df = pd.DataFrame(rows, columns=["ts", "size", "price"])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ns", utc=True), name="ts")
        tf[s] = df
    return qf, tf


def load_tops_file(path: str | Path, symbols: Iterable[str], store: Optional[ParquetStore] = None) -> dict:
    store = store or default_store()
    qf, tf = parse_tops(path, symbols)
    out = {}
    for s, df in qf.items():
        store.write("quotes", SOURCE, s, df.reset_index().drop_duplicates().set_index("ts"), partition="year",
                    fidelity=int(Fidelity.T2_PROXY), key=["ts", "bid", "ask", "bid_size", "ask_size"],
                    extra={"venue": "IEX only (not NBBO)", "file": Path(path).name})
        out[s] = {"quotes": len(df), "trades": len(tf.get(s, []))}
    for s, df in tf.items():
        store.write("trades", SOURCE, s, df, partition="year", fidelity=int(Fidelity.T2_PROXY),
                    key=["ts", "size", "price"], extra={"venue": "IEX only"})
    return out

"""Reason-coded cash ledger. Every cash movement in the account is one entry;
``sum(entries) == cash`` (the initial deposit is itself an entry) is an
invariant checked in debug mode. Reason values are the stable codes of
``atlasbt.core.schemas.LEDGER_REASONS``.
"""
from __future__ import annotations

import enum
from typing import Optional

import pandas as pd


class Reason(str, enum.Enum):
    DEPOSIT = "DEPOSIT"
    TRADE = "TRADE"                        # principal of a buy/sell (qty*price*multiplier)
    COMMISSION = "COMMISSION"
    EXCHANGE_FEES = "EXCHANGE_FEE"        # exchange, clearing, ORF, OCC, pass-throughs
    REGULATORY_FEES = "REGULATORY_FEE"    # SEC s.31, FINRA TAF, CAT
    DIVIDEND = "DIVIDEND"                  # long: gross dividend received (pay date)
    WITHHOLDING = "WITHHOLDING_TAX"            # tax withheld on dividends
    DIVIDEND_IN_LIEU = "DIVIDEND_IN_LIEU"  # short: payment in lieu of dividend
    CASH_IN_LIEU = "SPLIT_CASH_IN_LIEU"          # fractional shares after splits
    DELISTING = "DELIST_CASHOUT"                # cash-out at delisting value
    INTEREST_CREDIT = "CASH_INTEREST"
    INTEREST_DEBIT = "MARGIN_INTEREST"
    BORROW_FEE = "BORROW_FEE"
    SHORT_REBATE = "SHORT_REBATE"
    EXERCISE = "OPTION_EXERCISE"
    ASSIGNMENT = "OPTION_ASSIGNMENT"
    CASH_SETTLEMENT = "OPTION_CASH_SETTLEMENT"
    WITHDRAWAL = "WITHDRAWAL"


class Ledger:
    __slots__ = ("_rows", "_total")

    def __init__(self):
        self._rows: list[tuple] = []
        self._total = 0.0

    def post(self, ts: pd.Timestamp, reason: Reason, amount: float, symbol: str = "",
             note: str = "", ref: Optional[int] = None) -> None:
        if amount == 0.0:
            return
        self._rows.append((ts, reason.value, float(amount), symbol, note, ref))
        self._total += float(amount)

    @property
    def total(self) -> float:
        return self._total

    def recomputed_total(self) -> float:
        return float(sum(r[2] for r in self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=["ts", "reason", "amount", "symbol", "note", "ref"])
        df = df[["ts", "amount", "reason", "symbol", "note", "ref"]]
        if len(df):
            df["balance_change_cum"] = df["amount"].cumsum()
        else:
            df["balance_change_cum"] = pd.Series(dtype=float)
        return df

    def by_reason(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._rows:
            out[r[1]] = out.get(r[1], 0.0) + r[2]
        return out

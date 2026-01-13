# src/regimes/rules.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    explanation: str


def label_regime_row(row: pd.Series) -> RegimeResult:
    """
    Rule-based regime labeling v0.

    Priority:
      1) If SMA(10) exists, use trend + return rules.
      2) If SMA(10) is missing, fall back to return-only rules.
    """
    close = row["close"]
    sma_10 = row["sma_10"]
    log_ret = row["log_return_1"]

    # If we don't even have a return yet, we truly can't say anything.
    if pd.isna(log_ret):
        return RegimeResult(regime="unknown", explanation="insufficient data (log_return_1 is NaN)")

    # Fallback path: SMA not available yet (early rows).
    if pd.isna(sma_10):
        if log_ret > 0:
            return RegimeResult(
                regime="bullish", explanation="SMA(10) unavailable, positive return"
            )
        if log_ret < 0:
            return RegimeResult(
                regime="bearish", explanation="SMA(10) unavailable, negative return"
            )
        return RegimeResult(regime="sideways", explanation="SMA(10) unavailable, near-zero return")

    # Main path: SMA is available.
    if close > sma_10 and log_ret > 0:
        return RegimeResult(regime="bullish", explanation="price above SMA(10) and positive return")

    if close < sma_10 and log_ret < 0:
        return RegimeResult(regime="bearish", explanation="price below SMA(10) and negative return")

    return RegimeResult(regime="sideways", explanation="mixed signals between trend and return")


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies label_regime_row row-wise and returns a dataframe with:
      - regime
      - regime_explanation
    """
    results = df.apply(label_regime_row, axis=1)

    out = pd.DataFrame(
        {
            "regime": results.map(lambda r: r.regime),
            "regime_explanation": results.map(lambda r: r.explanation),
        },
        index=df.index,
    )

    return out

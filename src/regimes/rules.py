# src/regimes/rules.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    explanation: str


def _resolve_col(row: pd.Series, base: str) -> str:
    """
    Resolve a column name from a row that may be in long form (base)
    or wide form (base_x / base_y).
    Priority:
      1) base
      2) base_x
      3) base_y
    """
    if base in row.index:
        return base
    if f"{base}_x" in row.index:
        return f"{base}_x"
    if f"{base}_y" in row.index:
        return f"{base}_y"
    raise KeyError(
        f"Expected column '{base}' (or suffixed) not found. cols={list(row.index)}"
    )


def label_regime_row(row: pd.Series) -> RegimeResult:
    """
    Rule-based regime labeling v0.

    Priority:
      1) If SMA(10) exists, use trend + return rules.
      2) If SMA(10) is missing, fall back to return-only rules.

    Works with both:
      - long features (close, sma_10, log_return_1)
      - wide features (close_x, sma_10_x, log_return_1_x, ...)
    """
    close_col = _resolve_col(row, "close")
    sma_col = _resolve_col(row, "sma_10")
    ret_col = _resolve_col(row, "log_return_1")

    close = row[close_col]
    sma_10 = row[sma_col]
    log_ret = row[ret_col]

    # If we don't even have a return yet, we truly can't say anything.
    if pd.isna(log_ret):
        return RegimeResult(
            regime="unknown",
            explanation="insufficient data (log_return_1 is NaN)",
        )

    # Fallback path: SMA not available yet (early rows).
    if pd.isna(sma_10):
        if log_ret > 0:
            return RegimeResult(
                regime="bullish",
                explanation="SMA(10) unavailable, positive return",
            )
        if log_ret < 0:
            return RegimeResult(
                regime="bearish",
                explanation="SMA(10) unavailable, negative return",
            )
        return RegimeResult(
            regime="sideways",
            explanation="SMA(10) unavailable, near-zero return",
        )

    # Main path: SMA is available.
    if close > sma_10 and log_ret > 0:
        return RegimeResult(
            regime="bullish",
            explanation="price above SMA(10) and positive return",
        )

    if close < sma_10 and log_ret < 0:
        return RegimeResult(
            regime="bearish",
            explanation="price below SMA(10) and negative return",
        )

    return RegimeResult(
        regime="sideways",
        explanation="mixed signals between trend and return",
    )


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

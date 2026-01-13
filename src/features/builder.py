from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBuildConfig:
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    close_col: str = "close"
    sort_keys: tuple[str, ...] = ("symbol", "timestamp")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_features(bars: pd.DataFrame, config: FeatureBuildConfig | None = None) -> pd.DataFrame:
    """
    Deterministic feature builder v0.

    Input requirements:
      - timestamp column
      - symbol column
      - close column

    Output columns (fixed order):
      timestamp, symbol, close, log_return_1, sma_10
    """
    cfg = config or FeatureBuildConfig()
    _require_columns(bars, [cfg.timestamp_col, cfg.symbol_col, cfg.close_col])

    df = bars.copy()

    # Normalize dtypes for stability
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])
    df[cfg.symbol_col] = df[cfg.symbol_col].astype("string")
    df[cfg.close_col] = pd.to_numeric(df[cfg.close_col], errors="coerce").astype("float64")

    # Deterministic ordering
    df = df.sort_values(list(cfg.sort_keys), kind="mergesort").reset_index(drop=True)

    g = df.groupby(cfg.symbol_col, sort=False, group_keys=False)

    # Features
    prev_close = g[cfg.close_col].shift(1)
    df["log_return_1"] = np.log(df[cfg.close_col] / prev_close)

    df["sma_10"] = (
        g[cfg.close_col].rolling(window=10, min_periods=10).mean().reset_index(level=0, drop=True)
    )

    # Fixed output order
    out = df[[cfg.timestamp_col, cfg.symbol_col, cfg.close_col, "log_return_1", "sma_10"]].copy()
    out = out.sort_values(list(cfg.sort_keys), kind="mergesort").reset_index(drop=True)
    return out

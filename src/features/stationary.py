from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_PAIRWISE_BASE_COLS = (
    "close_x",
    "log_return_1_x",
    "sma_10_x",
    "close_y",
    "log_return_1_y",
    "sma_10_y",
)


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace([0.0, -0.0], np.nan)
    return numer / denom2


def augment_pairwise_stationary_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add relative / stationary pairwise features while keeping the original
    feature table intact.

    The returned frame preserves the original columns and appends:
      - trend_x
      - trend_y
      - trend_gap
      - return_gap
      - close_ratio
      - sma_ratio
    """
    out = df.copy()
    added: list[str] = []

    if not set(_PAIRWISE_BASE_COLS).issubset(out.columns):
        return out, added

    close_x = _as_float_series(out, "close_x")
    ret_x = _as_float_series(out, "log_return_1_x")
    sma_x = _as_float_series(out, "sma_10_x")
    close_y = _as_float_series(out, "close_y")
    ret_y = _as_float_series(out, "log_return_1_y")
    sma_y = _as_float_series(out, "sma_10_y")

    trend_x = _safe_ratio(close_x, sma_x) - 1.0
    trend_y = _safe_ratio(close_y, sma_y) - 1.0
    trend_gap = trend_x - trend_y
    return_gap = ret_x - ret_y
    close_ratio = _safe_ratio(close_x, close_y) - 1.0
    sma_ratio = _safe_ratio(sma_x, sma_y) - 1.0

    derived = {
        "trend_x": trend_x,
        "trend_y": trend_y,
        "trend_gap": trend_gap,
        "return_gap": return_gap,
        "close_ratio": close_ratio,
        "sma_ratio": sma_ratio,
    }

    for name, values in derived.items():
        out[name] = pd.Series(values, index=out.index, dtype="float64")
        added.append(name)

    return out, added


def summarize_feature_ranges(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    """
    Capture simple training or runtime feature envelope statistics.
    """
    stats: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").astype("float64")
        finite = series[np.isfinite(series.to_numpy(dtype=float, copy=False))]

        if finite.empty:
            stats[col] = {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "n_missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            }
            continue

        stats[col] = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "n_missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
        }

    return stats

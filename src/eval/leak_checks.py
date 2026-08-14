from __future__ import annotations

import numpy as np
import pandas as pd


def assert_monotonic_by_symbol(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> None:
    if timestamp_col not in df.columns or symbol_col not in df.columns:
        raise ValueError(f"Missing columns for monotonic check: {timestamp_col}, {symbol_col}")

    # groupwise monotonic increasing in timestamp
    bad = []
    for sym, g in df.groupby(symbol_col, sort=False):
        ts = pd.to_datetime(g[timestamp_col])
        if not ts.is_monotonic_increasing:
            bad.append(str(sym))
    if bad:
        raise AssertionError(f"Timestamps not monotonic increasing for symbols: {bad}")


def assert_log_return_1_no_future_leak(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    symbol_col: str = "symbol",
    logret_col: str = "log_return_1",
    tol: float = 1e-12,
) -> None:
    """
    Ensures log_return_1 matches log(close / close.shift(1)) per symbol.
    This catches accidental shift(-1) (future leak) or other misalignment.
    """
    for col in (close_col, symbol_col, logret_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    g = df.groupby(symbol_col, sort=False, group_keys=False)

    prev_close = g[close_col].shift(1)
    expected = np.log(df[close_col].astype(float) / prev_close.astype(float))

    actual = pd.to_numeric(df[logret_col], errors="coerce").astype(float)

    # Compare where both are finite
    mask = np.isfinite(expected.to_numpy()) & np.isfinite(actual.to_numpy())
    if mask.sum() == 0:
        # nothing to compare, don't silently pass incorrect behavior
        raise AssertionError("No finite values to compare for log_return_1 leak check")

    max_abs_err = float(np.max(np.abs(expected.to_numpy()[mask] - actual.to_numpy()[mask])))
    if max_abs_err > tol:
        raise AssertionError(
            f"log_return_1 mismatch vs shift(1) definition, max_abs_err={max_abs_err}"
        )


def assert_sma_10_is_backward_looking(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    symbol_col: str = "symbol",
    sma_col: str = "sma_10",
    window: int = 10,
    tol: float = 1e-12,
) -> None:
    """
    Ensures sma_10 equals rolling mean of past 10 closes (including current),
    min_periods=10 per symbol. Catches centered windows or shift(-k) leakage.
    """
    for col in (close_col, symbol_col, sma_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    g = df.groupby(symbol_col, sort=False, group_keys=False)
    expected = (
        g[close_col]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    actual = pd.to_numeric(df[sma_col], errors="coerce").astype(float)

    mask = np.isfinite(expected.to_numpy()) & np.isfinite(actual.to_numpy())
    if mask.sum() == 0:
        # it's okay if your dataset is too short, but then the SMA should be all NaN
        if np.isfinite(actual.to_numpy()).any():
            raise AssertionError("sma_10 has finite values but expected none (dataset too short?)")
        return

    max_abs_err = float(np.max(np.abs(expected.to_numpy()[mask] - actual.to_numpy()[mask])))
    if max_abs_err > tol:
        raise AssertionError(f"sma_10 mismatch vs backward rolling mean, max_abs_err={max_abs_err}")

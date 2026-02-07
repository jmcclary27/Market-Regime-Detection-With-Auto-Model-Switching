from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.leak_checks import (
    assert_log_return_1_no_future_leak,
    assert_monotonic_by_symbol,
    assert_sma_10_is_backward_looking,
)
from src.features.builder import build_features


def test_leak_checks_pass_on_feature_builder_output() -> None:
    n = 40
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    bars = pd.DataFrame(
        {
            "timestamp": list(ts) + list(ts),
            "symbol": ["SPY"] * n + ["QQQ"] * n,
            "close": list(np.linspace(100, 120, n)) + list(np.linspace(200, 230, n)),
        }
    )

    feat = build_features(bars)

    assert_monotonic_by_symbol(feat, timestamp_col="timestamp", symbol_col="symbol")
    assert_log_return_1_no_future_leak(feat, close_col="close", symbol_col="symbol", logret_col="log_return_1")
    assert_sma_10_is_backward_looking(feat, close_col="close", symbol_col="symbol", sma_col="sma_10", window=10)


def test_leak_check_catches_future_shift_in_log_return() -> None:
    n = 30
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["SPY"] * n,
            "close": np.linspace(100, 110, n),
        }
    )
    feat = build_features(bars)

    # Introduce a future leak bug: shift(-1) instead of shift(1)
    g = feat.groupby("symbol", sort=False, group_keys=False)
    next_close = g["close"].shift(-1)
    feat_bad = feat.copy()
    feat_bad["log_return_1"] = np.log(next_close / feat_bad["close"])

    with pytest.raises(AssertionError):
        assert_log_return_1_no_future_leak(feat_bad, close_col="close", symbol_col="symbol", logret_col="log_return_1")


def test_leak_check_catches_centered_sma() -> None:
    n = 30
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["SPY"] * n,
            "close": np.linspace(100, 110, n),
        }
    )
    feat = build_features(bars)

    # Introduce leakage bug: centered rolling mean uses future points
    feat_bad = feat.copy()
    feat_bad["sma_10"] = (
        feat_bad.groupby("symbol", sort=False)["close"]
        .rolling(window=10, min_periods=10, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )

    with pytest.raises(AssertionError):
        assert_sma_10_is_backward_looking(feat_bad, close_col="close", symbol_col="symbol", sma_col="sma_10", window=10)

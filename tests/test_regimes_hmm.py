from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.load_config import load_config
from src.regimes.hmm import label_regimes_hmm


def test_hmm_labeling_contract() -> None:
    """
    Contract test for HMM regime labeling.

    This test does NOT depend on:
    - data/features/latest.parquet
    - trained HMM artifacts on disk

    It only verifies:
    - output length matches input
    - output schema is correct
    - regime labels are from the allowed set

    It builds a minimal synthetic DataFrame with the required observation columns.
    """
    cfg = load_config("src/config/settings.yaml")

    n = 200
    rng = np.random.default_rng(0)

    # Build synthetic, time-ordered price series so SMA makes sense
    close_x = 100 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    close_y = 200 + np.cumsum(rng.normal(0.0, 1.5, size=n))

    df = pd.DataFrame(
        {
            "close_x": close_x,
            "close_y": close_y,
        }
    )

    # Simple SMA(10) with min_periods=1 to avoid NaNs early
    df["sma_10_x"] = df["close_x"].rolling(10, min_periods=1).mean()
    df["sma_10_y"] = df["close_y"].rolling(10, min_periods=1).mean()

    # Log returns (fill first value to 0 so we don't introduce NaNs)
    df["log_return_1_x"] = np.log(df["close_x"]).diff().fillna(0.0)
    df["log_return_1_y"] = np.log(df["close_y"]).diff().fillna(0.0)

    # Keep only what the HMM code requires (extra cols are fine too, but be explicit)
    df = df[
        [
            "log_return_1_x",
            "log_return_1_y",
            "close_x",
            "sma_10_x",
            "close_y",
            "sma_10_y",
        ]
    ]

    labels = label_regimes_hmm(df, cfg=cfg)

    assert len(labels) == len(df)
    assert list(labels.columns) == ["regime", "regime_explanation"]

    allowed = {"bullish", "bearish", "sideways", "unknown"}
    assert set(labels["regime"].dropna().unique()).issubset(allowed)

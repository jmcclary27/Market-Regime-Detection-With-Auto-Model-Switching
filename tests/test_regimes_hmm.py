from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.load_config import load_config
from src.regimes.hmm import label_regimes_hmm


def test_hmm_labeling_contract() -> None:
    """
    Contract test for HMM regime labeling.

    This test is an *integration-style contract* because label_regimes_hmm()
    requires trained artifacts on disk (models/regimes/hmm/latest/*).

    In CI (clean checkout), those artifacts won't exist, so we skip.
    Locally, if you've trained the HMM artifacts, the test runs and enforces:
    - output length matches input
    - output schema is correct
    - regime labels are from the allowed set
    """
    artifacts_dir = Path("models/regimes/hmm/latest")
    required = [
        artifacts_dir / "model.joblib",
        artifacts_dir / "scaler.joblib",
        artifacts_dir / "state_mapping.json",
        artifacts_dir / "metadata.json",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("HMM artifacts missing, run tools/train_hmm_regime.py to generate them")

    cfg = load_config("src/config/settings.yaml")

    n = 200
    rng = np.random.default_rng(0)

    close_x = 100 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    close_y = 200 + np.cumsum(rng.normal(0.0, 1.5, size=n))

    df = pd.DataFrame({"close_x": close_x, "close_y": close_y})
    df["sma_10_x"] = df["close_x"].rolling(10, min_periods=1).mean()
    df["sma_10_y"] = df["close_y"].rolling(10, min_periods=1).mean()
    df["log_return_1_x"] = np.log(df["close_x"]).diff().fillna(0.0)
    df["log_return_1_y"] = np.log(df["close_y"]).diff().fillna(0.0)

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

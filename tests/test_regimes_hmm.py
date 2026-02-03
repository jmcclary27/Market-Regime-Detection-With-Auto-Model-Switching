from __future__ import annotations

import pandas as pd

from src.config.load_config import load_config
from src.regimes.hmm import label_regimes_hmm


def test_hmm_labeling_contract() -> None:
    cfg = load_config("src/config/settings.yaml")

    # This assumes you've trained and have models/regimes/hmm/latest/*
    df = pd.read_parquet("data/features/latest.parquet").head(200)

    labels = label_regimes_hmm(df, cfg=cfg)

    assert len(labels) == len(df)
    assert list(labels.columns) == ["regime", "regime_explanation"]

    allowed = {"bullish", "bearish", "sideways", "unknown"}
    assert set(labels["regime"].dropna().unique()).issubset(allowed)

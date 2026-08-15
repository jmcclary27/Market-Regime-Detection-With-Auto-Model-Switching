from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.regimes.hmm import label_regimes_hmm


class _IdentityScaler:
    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values


class _SpreadDirectionModel:
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.asarray(values[:, 2] >= 0.0, dtype=np.int64)


def _write_hmm_artifacts(tmp_path: Path) -> dict[str, object]:
    artifacts_root = tmp_path / "hmm"
    latest = artifacts_root / "latest"
    latest.mkdir(parents=True)
    joblib.dump(_SpreadDirectionModel(), latest / "model.joblib")
    joblib.dump(_IdentityScaler(), latest / "scaler.joblib")
    (latest / "state_mapping.json").write_text(
        json.dumps({"0": "bearish", "1": "bullish"}), encoding="utf-8"
    )
    (latest / "metadata.json").write_text(
        json.dumps(
            {
                "obs_mode": "minimal",
                "obs_cols": ["ret_x", "ret_y", "spread_ret"],
                "per_state_stats": [
                    {"_state": 0, "mean_spread": -0.01},
                    {"_state": 1, "mean_spread": 0.01},
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"regimes": {"hmm": {"artifacts_dir": str(artifacts_root)}}}


def test_hmm_labeling_contract_uses_explicit_temp_artifacts(tmp_path: Path) -> None:
    cfg = _write_hmm_artifacts(tmp_path)
    features = pd.DataFrame(
        {
            "log_return_1_x": [-0.02, 0.03, np.nan],
            "log_return_1_y": [0.0, 0.01, 0.0],
        }
    )

    labels = label_regimes_hmm(features, cfg=cfg)

    assert list(labels.columns) == ["regime", "regime_explanation"]
    assert labels["regime"].tolist() == ["bearish", "bullish", "unknown"]
    assert "train_mean_spread=-0.01" in labels.loc[0, "regime_explanation"]
    assert labels.loc[2, "regime_explanation"] == "insufficient data for HMM observations"


def test_hmm_labeling_fails_clearly_when_artifacts_are_missing(tmp_path: Path) -> None:
    cfg = {"regimes": {"hmm": {"artifacts_dir": str(tmp_path / "missing")}}}
    features = pd.DataFrame({"log_return_1_x": [0.01], "log_return_1_y": [0.0]})

    with pytest.raises(FileNotFoundError, match="Missing HMM artifacts"):
        label_regimes_hmm(features, cfg=cfg)

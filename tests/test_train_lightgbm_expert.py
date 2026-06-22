from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tools.train_lightgbm_expert import TrainConfig, run


def _build_regime_labeled_fixture(path: Path, n: int = 540) -> None:
    timestamps = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    regimes = np.array(["bullish", "bearish", "sideways"] * (n // 3), dtype=object)

    base = np.sin(np.arange(n, dtype=float) / 9.0) + 0.2 * np.cos(np.arange(n, dtype=float) / 5.0)
    target_next = np.zeros(n, dtype=float)
    for i in range(n - 1):
        regime = str(regimes[i])
        if regime == "bullish":
            target_next[i] = 0.02 + 0.01 * base[i]
        elif regime == "bearish":
            target_next[i] = -0.02 - 0.01 * base[i]
        else:
            target_next[i] = 0.003 * np.sign(base[i]) + 0.0015 * base[i]

    log_return_1_x = np.zeros(n, dtype=float)
    log_return_1_x[1:] = target_next[:-1]
    log_return_1_y = 0.6 * log_return_1_x + 0.0005 * np.cos(np.arange(n, dtype=float) / 4.0)

    close_x = 100.0 + np.cumsum(log_return_1_x)
    close_y = 200.0 + np.cumsum(log_return_1_y)
    sma_10_x = pd.Series(close_x).rolling(10, min_periods=1).mean().to_numpy()
    sma_10_y = pd.Series(close_y).rolling(10, min_periods=1).mean().to_numpy()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_x": close_x,
            "log_return_1_x": log_return_1_x,
            "sma_10_x": sma_10_x,
            "close_y": close_y,
            "log_return_1_y": log_return_1_y,
            "sma_10_y": sma_10_y,
            # Strong current-row predictor for the shifted target.
            "feature_hint": target_next,
            "feature_hint_y": target_next * 0.8,
            "regime": regimes,
            "regime_explanation": [f"fixture_{r}" for r in regimes],
        }
    )
    df.to_parquet(path, index=False)


def _model_hash(path: Path) -> str:
    model = joblib.load(path)
    booster = model.booster_
    payload = json.dumps(booster.dump_model(), sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def test_train_lightgbm_experts_are_regime_specific_and_non_constant(tmp_path: Path) -> None:
    features_path = tmp_path / "data" / "regimes" / "latest.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _build_regime_labeled_fixture(features_path)

    output_dir = tmp_path / "models" / "experts"
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()

    hashes: dict[str, str] = {}
    for regime in ("bullish", "bearish", "sideways"):
        cfg = TrainConfig(
            features_path=str(features_path),
            regimes_path=None,
            target_col="log_return_1_x",
            target_expr=None,
            target_shift=-1,
            group_col=None,
            vol_window=None,
            min_regime_rows=100,
            regime=regime,
            model_name="lightgbm_expert",
            experiment_name="test-market-regime",
            run_name=f"test_{regime}",
            output_dir=str(output_dir),
            id_cols=["timestamp"],
            drop_cols=["regime", "regime_explanation"],
            time_col="timestamp",
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            early_stopping_rounds=20,
            num_boost_round=200,
            seed=42,
            params_json=None,
            mlflow_tracking_uri=tracking_uri,
        )

        run(cfg)

        latest_dir = output_dir / regime
        metadata = json.loads((latest_dir / "latest.json").read_text(encoding="utf-8"))
        assert metadata["regime"] == regime
        assert metadata["val_pred_nunique"] > 1
        assert metadata["test_pred_nunique"] > 1
        assert metadata["n_rows_used"] >= 100
        assert metadata["n_features"] > 6
        assert metadata["stationary_feature_columns"] == [
            "trend_x",
            "trend_y",
            "trend_gap",
            "return_gap",
            "close_ratio",
            "sma_ratio",
        ]
        assert "feature_range_stats" in metadata
        assert "trend_gap" in metadata["feature_range_stats"]

        hashes[regime] = _model_hash(latest_dir / "latest.joblib")

    assert len(set(hashes.values())) == 3

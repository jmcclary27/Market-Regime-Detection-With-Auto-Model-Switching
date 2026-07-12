from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.train import TrainConfig as BaselineConfig
from src.models.train import run as run_baseline
from tools.make_pretrained_expert import TrainConfig as RidgeExpertConfig
from tools.make_pretrained_expert import run as run_ridge_expert
from tools.train_arima_expert import TrainConfig as ArimaConfig
from tools.train_arima_expert import run as run_arima


def _write_features_and_regimes(tmp_path: Path, n: int = 300) -> tuple[Path, Path]:
    timestamps = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    angles = np.linspace(0.0, 16.0 * np.pi, n)
    returns = 0.002 * np.sin(angles) + 0.0002 * np.cos(angles * 0.37)
    close = 100.0 * np.exp(np.cumsum(returns))
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_x": close,
            "log_return_1_x": returns,
            "sma_10_x": pd.Series(close).rolling(10, min_periods=1).mean(),
            "close_y": close * 1.01,
            "log_return_1_y": returns * 0.8,
            "sma_10_y": pd.Series(close * 1.01).rolling(10, min_periods=1).mean(),
        }
    )
    regimes = pd.DataFrame(
        {
            "timestamp": timestamps,
            "regime": np.where(np.arange(n) < 210, "bullish", "bearish"),
        }
    )
    features_path = tmp_path / "features.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    features.to_parquet(features_path, index=False)
    regimes.to_parquet(regimes_path, index=False)
    return features_path, regimes_path


def test_baseline_training_writes_candidate_without_latest_pointer(tmp_path: Path) -> None:
    features_path, _ = _write_features_and_regimes(tmp_path)
    candidate_root = tmp_path / "models" / "candidates" / "baseline"

    model_path = run_baseline(
        BaselineConfig(
            features_path=features_path,
            baseline_models_dir=candidate_root,
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="candidate-baseline-test",
        )
    )

    artifact = joblib.load(model_path)
    assert artifact["candidate_only"] is True
    assert artifact["runtime_versions"]["scikit_learn"]
    assert not (candidate_root / "latest.joblib").exists()
    assert not (candidate_root / "latest.json").exists()


def test_pretrained_ridge_training_filters_regime_and_stays_candidate_only(tmp_path: Path) -> None:
    features_path, regimes_path = _write_features_and_regimes(tmp_path)
    candidate_root = tmp_path / "models" / "candidates" / "pretrained"

    candidate_path = run_ridge_expert(
        RidgeExpertConfig(
            features_path=features_path,
            regimes_path=regimes_path,
            regime="bullish",
            min_regime_rows=180,
            output_dir=candidate_root,
            model_name="ridge_bullish_test",
        )
    )

    metadata = json.loads(candidate_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["candidate_only"] is True
    assert metadata["regime_filter_applied"] is True
    assert "zero_return_test_rmse" in metadata["quality_gate"]
    assert metadata["n_rows_used"] == 210
    assert metadata["test_pred_nunique"] > 1
    assert "zero_return_test_rmse" in metadata["quality_gate"]
    assert not list((tmp_path / "models" / "pretrained").glob("*.joblib"))


def test_arima_training_filters_regime_and_does_not_publish_latest(tmp_path: Path) -> None:
    features_path, regimes_path = _write_features_and_regimes(tmp_path)
    candidate_root = tmp_path / "models" / "candidates" / "arima"

    out_dir = run_arima(
        ArimaConfig(
            features_path=str(features_path),
            regimes_path=str(regimes_path),
            target_col="log_return_1_x",
            target_expr=None,
            target_shift=-1,
            group_col=None,
            vol_window=None,
            min_regime_rows=180,
            regime="bullish",
            model_name="arima_bullish_test",
            experiment_name="candidate-arima-test",
            run_name="candidate-arima-test",
            output_dir=str(candidate_root),
            publish_latest=False,
            update_legacy_pointer=False,
            id_cols=["timestamp"],
            drop_cols=["regime", "regime_explanation"],
            time_col="timestamp",
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            p=1,
            d=0,
            q=0,
            trend="c",
            refit_interval=50,
            train_window=100,
            min_train_size=60,
            seed=42,
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
        )
    )

    metadata = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    assert metadata["candidate_only"] is True
    assert metadata["shadow_only"] is True
    assert metadata["regime_filter_applied"] is True
    assert metadata["regime_history_policy"] == "filter_to_training_regime"
    assert metadata["model_id"] == "expert_arima_bullish_arima_bullish_test"
    assert metadata["n_rows_used"] == 210
    assert metadata["val_pred_nunique"] > 1
    assert metadata["test_pred_nunique"] > 1
    assert not (candidate_root / "bullish" / "arima" / f"{metadata['model_id']}.json").exists()
    assert not (candidate_root / "bullish" / "latest.arima.json").exists()

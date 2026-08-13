from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.features.stationary import augment_pairwise_stationary_features
from src.inference.batch_predict import BatchPredictConfig, _walk_forward_arima_predict, run
from src.registry.registry import ActiveModelRef, write_active


def test_bundle_feature_contract_overrides_positional_column_order(tmp_path: Path) -> None:
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    # Train in f2/f1 order.  The serialized contract must preserve that order
    # even though inference receives f1/f2.
    model = LinearRegression().fit(features[["f2", "f1"]], [0.01, 0.02, 0.03])
    model_dir = tmp_path / "models" / "baseline" / "123"
    model_dir.mkdir(parents=True)
    joblib.dump({"model": model, "feature_cols": ["f2", "f1"]}, model_dir / "model.joblib")

    out = run(
        BatchPredictConfig(
            features_path=features_path,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "predictions",
            runs_dir=tmp_path / "runs",
            require_published_model_contract=False,
        )
    )

    actual = pd.read_parquet(out).sort_values("row_id")["y_pred"].to_numpy()
    expected = model.predict(features[["f2", "f1"]])
    assert np.allclose(actual, expected)


def test_return_target_is_removed_after_stationary_feature_engineering(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "close_x": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "log_return_1_x": [0.001, 0.002, -0.001, 0.003, -0.002, 0.001],
            "sma_10_x": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close_y": [90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
            "log_return_1_y": [0.0005, 0.001, -0.0005, 0.0015, -0.001, 0.0005],
            "sma_10_y": [89.0, 90.0, 91.0, 92.0, 93.0, 94.0],
        }
    )
    features_path = tmp_path / "regimes.parquet"
    features.to_parquet(features_path, index=False)

    augmented, _ = augment_pairwise_stationary_features(features)
    feature_columns = [column for column in augmented.columns if column != "log_return_1_x"]
    model = LinearRegression().fit(
        augmented[feature_columns], np.linspace(-0.004, 0.004, len(augmented))
    )
    model_path = tmp_path / "models" / "baseline" / "123" / "model.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump({"model": model, "feature_cols": feature_columns}, model_path)

    out = run(
        BatchPredictConfig(
            features_path=features_path,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "predictions",
            runs_dir=tmp_path / "runs",
            require_published_model_contract=False,
        )
    )

    actual = pd.read_parquet(out).sort_values("row_id")["y_pred"].to_numpy()
    expected = model.predict(augmented[feature_columns])
    assert np.allclose(actual, expected)


def test_active_artifact_is_emitted_once_with_its_canonical_model_name(tmp_path: Path) -> None:
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    model = LinearRegression().fit(features, [0.01, 0.02, 0.03])
    model_path = tmp_path / "models" / "baseline" / "123" / "model.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(model, model_path)

    active_file = tmp_path / "registry" / "active_model.yaml"
    write_active(
        ActiveModelRef(
            model_type="baseline",
            model_id="baseline",
            version="123",
            artifact_path=model_path,
        ),
        active_file=active_file,
    )

    out = run(
        BatchPredictConfig(
            features_path=features_path,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "predictions",
            runs_dir=tmp_path / "runs",
            active_file=active_file,
            require_published_model_contract=False,
        )
    )

    predictions = pd.read_parquet(out)
    assert predictions["model_name"].unique().tolist() == ["baseline"]
    assert predictions["is_active"].all()
    assert "active" not in predictions["model_name"].tolist()


def test_out_of_contract_prediction_scale_is_quarantined(tmp_path: Path) -> None:
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    # This model is intentionally trained to return price-scale values rather
    # than decimal returns, so the production contract must reject it.
    model = LinearRegression().fit(features, [1.0, 2.0, 3.0])
    model_dir = tmp_path / "models" / "baseline" / "123"
    model_dir.mkdir(parents=True)
    joblib.dump(model, model_dir / "model.joblib")

    try:
        run(
            BatchPredictConfig(
                features_path=features_path,
                models_dir=tmp_path / "models",
                output_dir=tmp_path / "predictions",
                runs_dir=tmp_path / "runs",
                require_published_model_contract=False,
            )
        )
    except RuntimeError as exc:
        assert "prediction scale violates" in str(exc)
    else:  # pragma: no cover - makes a contract regression explicit
        raise AssertionError("out-of-contract prediction scale was accepted")


def test_global_active_failure_does_not_publish_a_shadows_only_run(tmp_path: Path) -> None:
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    bad_active = LinearRegression().fit(features, [1.0, 2.0, 3.0])
    baseline_path = tmp_path / "models" / "baseline" / "123" / "model.joblib"
    baseline_path.parent.mkdir(parents=True)
    joblib.dump(bad_active, baseline_path)

    healthy_shadow = LinearRegression().fit(features, [0.01, 0.02, 0.03])
    pretrained_path = tmp_path / "models" / "pretrained" / "healthy_shadow.joblib"
    pretrained_path.parent.mkdir(parents=True)
    joblib.dump(healthy_shadow, pretrained_path)

    active_file = tmp_path / "registry" / "active_model.yaml"
    write_active(
        ActiveModelRef(
            model_type="baseline",
            model_id="baseline",
            version="123",
            artifact_path=baseline_path,
        ),
        active_file=active_file,
    )

    with pytest.raises(RuntimeError, match="Active registry model failed inference safety checks"):
        run(
            BatchPredictConfig(
                features_path=features_path,
                models_dir=tmp_path / "models",
                output_dir=tmp_path / "predictions",
                runs_dir=tmp_path / "runs",
                active_file=active_file,
                require_published_model_contract=False,
            )
        )


def test_regime_specific_arima_is_executed_as_a_shadow_model(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "log_return_1_x": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "close_x": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "close_y": [90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
            "regime": ["bullish"] * 6,
        }
    )
    features_path = tmp_path / "regimes.parquet"
    df.to_parquet(features_path, index=False)

    arima_path = tmp_path / "models" / "experts" / "bullish" / "latest.arima.json"
    arima_path.parent.mkdir(parents=True)
    arima_path.write_text(
        json.dumps(
            {
                "model_type": "arima",
                "regime": "bullish",
                "training_regime": "bullish",
                "order": {"p": 0, "d": 0, "q": 0},
                "trend": "c",
                "refit_interval": 1,
                "min_train_size": 2,
                "target_col": "log_return_1_x",
                "target_shift": -1,
            }
        ),
        encoding="utf-8",
    )

    out = run(
        BatchPredictConfig(
            features_path=features_path,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "predictions",
            runs_dir=tmp_path / "runs",
            min_prediction_nunique=1,
            min_prediction_unique_fraction=0.0,
            require_published_model_contract=False,
        )
    )

    predictions = pd.read_parquet(out)
    assert predictions["model_name"].unique().tolist() == ["expert_arima_bullish"]
    assert len(predictions) == len(df)


def test_regime_specific_arima_history_changes_the_forecast() -> None:
    y = pd.Series([0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.04, -0.04])
    regimes = pd.Series(
        ["bullish", "bullish", "bullish", "bearish", "bearish", "bearish", "bullish", "bearish"]
    )

    bullish = _walk_forward_arima_predict(
        y,
        order=(0, 0, 0),
        trend="c",
        refit_interval=1,
        train_window=None,
        min_train_size=2,
        history_regimes=regimes,
        training_regime="bullish",
    )
    bearish = _walk_forward_arima_predict(
        y,
        order=(0, 0, 0),
        trend="c",
        refit_interval=1,
        train_window=None,
        min_train_size=2,
        history_regimes=regimes,
        training_regime="bearish",
    )

    assert not np.allclose(bullish.to_numpy(), bearish.to_numpy())

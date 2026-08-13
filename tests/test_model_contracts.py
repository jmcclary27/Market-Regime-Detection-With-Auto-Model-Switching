from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.inference.batch_predict import BatchPredictConfig, run

EXPECTED_COLS = [
    "row_id",
    "model_name",
    "model_source",
    "y_pred",
    "inference_ts",
    "features_path",
    "model_path",
    "is_active",
    "active_model_type",
    "active_model_id",
    "active_model_version",
    "active_regime",
]

ALLOWED_MODEL_SOURCES = {"baseline", "expert", "pretrained"}
ALLOWED_ACTIVE_TYPES = {"baseline", "expert", "pretrained"}


@pytest.fixture
def predictions(tmp_path: Path) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.2, 0.4, 0.1, 0.6],
        }
    )
    features_path = tmp_path / "data" / "features.parquet"
    features_path.parent.mkdir(parents=True)
    features.to_parquet(features_path, index=False)

    model_path = tmp_path / "models" / "baseline" / "1" / "model.joblib"
    model_path.parent.mkdir(parents=True)
    import joblib

    joblib.dump(LinearRegression().fit(features, [0.0, 0.1, 0.2, 0.3]), model_path)

    output_path = run(
        BatchPredictConfig(
            features_path=features_path,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "data" / "predictions",
            runs_dir=tmp_path / "data" / "runs",
            require_published_model_contract=False,
            max_abs_prediction=10.0,
        )
    )
    return pd.read_parquet(output_path)


def test_predictions_schema_exact(predictions: pd.DataFrame) -> None:
    assert list(predictions.columns) == EXPECTED_COLS


def test_predictions_basic_integrity(predictions: pd.DataFrame) -> None:
    assert len(predictions) > 0
    assert predictions["row_id"].notna().all()
    assert predictions["model_name"].notna().all()
    assert (predictions["model_name"].astype(str).str.strip() != "").all()
    assert not (set(predictions["model_source"].astype(str)) - ALLOWED_MODEL_SOURCES)
    assert pd.api.types.is_numeric_dtype(predictions["y_pred"])
    assert predictions["y_pred"].notna().all()


def test_one_prediction_per_row_id_per_model(predictions: pd.DataFrame) -> None:
    assert not predictions.duplicated(subset=["row_id", "model_name"], keep=False).any()


def test_equal_row_id_coverage_across_models(predictions: pd.DataFrame) -> None:
    models = sorted(predictions["model_name"].unique().tolist())
    assert models
    ids_by_model = {
        model: set(predictions.loc[predictions["model_name"] == model, "row_id"].tolist())
        for model in models
    }
    assert all(ids == ids_by_model[models[0]] for ids in ids_by_model.values())


def test_run_level_metadata_constant_within_file(predictions: pd.DataFrame) -> None:
    assert predictions["inference_ts"].nunique(dropna=False) == 1
    assert predictions["features_path"].nunique(dropna=False) == 1


def test_model_path_constant_per_model(predictions: pd.DataFrame) -> None:
    assert (
        predictions.groupby("model_name", sort=False)["model_path"].nunique(dropna=False) == 1
    ).all()


def test_predictions_not_all_constant_per_model(predictions: pd.DataFrame) -> None:
    assert all(group["y_pred"].nunique() > 1 for _, group in predictions.groupby("model_name"))


def test_active_pointer_consistency(predictions: pd.DataFrame) -> None:
    active = predictions[predictions["is_active"] == True]  # noqa: E712
    if active.empty:
        return
    assert (active.groupby("row_id", sort=False).size() <= 1).all()
    assert active["active_model_type"].notna().all()
    assert not (set(active["active_model_type"].astype(str).str.lower()) - ALLOWED_ACTIVE_TYPES)
    assert active["active_model_id"].notna().all()
    assert active["active_model_version"].notna().all()

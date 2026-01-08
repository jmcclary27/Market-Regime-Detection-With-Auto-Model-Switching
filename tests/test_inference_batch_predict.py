# tests/test_inference_batch_predict.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.inference.batch_predict import BatchPredictConfig, run


def test_batch_predict_produces_predictions(tmp_path: Path) -> None:
    # ---- features ----
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [0.5, 0.2, 0.1],
    })
    features_path = features_dir / "latest.parquet"
    df.to_parquet(features_path)

    # ---- model ----
    model = LinearRegression().fit(df, [1, 2, 3])

    models_dir = tmp_path / "models" / "baseline" / "123"
    models_dir.mkdir(parents=True)
    joblib.dump(model, models_dir / "model.joblib")

    # ---- run ----
    config = BatchPredictConfig(
        features_path=features_path,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "data" / "predictions",
        runs_dir=tmp_path / "data" / "runs",
    )

    out_path = run(config)

    assert out_path.exists()

    out_df = pd.read_parquet(out_path)
    assert "model_name" in out_df.columns
    assert "y_pred" in out_df.columns
    assert len(out_df) == len(df)
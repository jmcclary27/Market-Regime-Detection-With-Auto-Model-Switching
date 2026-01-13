from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

from src.inference.batch_predict import BatchPredictConfig, run


def test_batch_predict_writes_predictions(tmp_path: Path) -> None:
    # Arrange: tiny features
    data_dir = tmp_path / "data"
    (data_dir / "features").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [0.1, 0.2, 0.3],
            "f3": [5.0, 6.0, 7.0],
        }
    )
    feat_path = data_dir / "features" / "latest.parquet"
    df.to_parquet(feat_path)

    # Arrange: baseline model expecting 3 features
    model = Ridge().fit(df.to_numpy(), [0.0, 1.0, 2.0])

    models_dir = tmp_path / "models" / "baseline" / "123"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")

    # Act
    config = BatchPredictConfig(
        features_path=feat_path,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "data" / "predictions",
        runs_dir=tmp_path / "data" / "runs",
    )
    out_path = run(config)

    # Assert
    assert out_path.exists()
    latest_path = config.output_dir / "latest.parquet"
    assert latest_path.exists()

    out_df = pd.read_parquet(latest_path)
    assert set(["row_id", "model_name", "model_source", "y_pred", "inference_ts"]).issubset(
        out_df.columns
    )
    assert out_df["model_name"].nunique() >= 1
    assert len(out_df) == len(df)  # 1 model * 3 rows

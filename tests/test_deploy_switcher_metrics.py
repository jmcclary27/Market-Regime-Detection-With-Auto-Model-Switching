import json
from pathlib import Path

import pandas as pd

from src.deploy.switcher import SwitchConfig, run_switcher
from src.features.stationary import augment_pairwise_stationary_features, summarize_feature_ranges


def _write_guard_inputs(root: Path, *, candidate_model_id: str, y_pred: list[float]) -> None:
    features_path = root / "data" / "features" / "current.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame(
        {
            "close_x": [100.0, 101.0, 102.0, 103.0],
            "log_return_1_x": [0.01, 0.011, 0.012, 0.013],
            "sma_10_x": [99.4, 100.0, 100.6, 101.1],
            "close_y": [50.0, 50.4, 50.8, 51.1],
            "log_return_1_y": [0.004, 0.005, 0.0055, 0.006],
            "sma_10_y": [49.5, 49.8, 50.1, 50.4],
        }
    )
    features.to_parquet(features_path, index=False)

    augmented, stationary_cols = augment_pairwise_stationary_features(features)
    metadata = {
        "model_type": "lightgbm",
        "model_name": candidate_model_id,
        "regime": "bullish",
        "feature_columns": list(augmented.columns),
        "stationary_feature_columns": stationary_cols,
        "feature_range_stats": summarize_feature_ranges(augmented, list(augmented.columns)),
    }
    metadata_path = root / "models" / "experts" / "bullish" / "latest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "model_name": [candidate_model_id] * len(y_pred),
            "model_source": ["expert"] * len(y_pred),
            "model_path": ["models/experts/bullish/latest.joblib"] * len(y_pred),
            "features_path": [str(features_path)] * len(y_pred),
            "y_pred": y_pred,
        }
    )
    (root / "data" / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_parquet(root / "data" / "predictions" / "latest.parquet", index=False)


def test_switcher_logs_metrics_from_scorecard(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scorecards_dir = data_dir / "scorecards"
    scorecards_dir.mkdir(parents=True)
    _write_guard_inputs(
        tmp_path, candidate_model_id="expert_bullish", y_pred=[0.01, 0.015, 0.02, 0.018]
    )

    df = pd.DataFrame(
        {
            "scope": ["overall", "overall"],
            "regime": [None, None],
            "model_name": ["baseline", "expert_bullish"],
            "n": [100, 100],
            "rmse": [1.0, 0.9],
            "mae": [0.8, 0.7],
        }
    )
    df.to_parquet(scorecards_dir / "latest.parquet", index=False)

    config = SwitchConfig(metric_name="rmse", window_value=50)

    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )

    events_path = data_dir / "deployments" / "events.parquet"
    events = pd.read_parquet(events_path)
    row = events.iloc[0]

    assert row["active_metric_value"] == 1.0
    assert row["candidate_metric_value"] == 0.9
    assert row["n"] == 100
    assert row["decision"] in {"promote", "rollback", "hold", "no_action"}

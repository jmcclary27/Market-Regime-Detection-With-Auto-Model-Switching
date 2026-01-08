from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import joblib
from sklearn.linear_model import Ridge

from src.models.train import TrainConfig, run


def test_train_produces_artifacts(tmp_path: Path) -> None:
    # Arrange: create temp repo-like structure
    data_dir = tmp_path / "data"
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "regimes").mkdir(parents=True, exist_ok=True)
    (data_dir / "runs").mkdir(parents=True, exist_ok=True)

    models_dir = tmp_path / "models"
    (models_dir / "pretrained").mkdir(parents=True, exist_ok=True)

    # Copy fixture parquet files from repo into tmp_path
    repo_feat = Path("data/features/latest.parquet")
    repo_reg = Path("data/regimes/latest.parquet")
    assert repo_feat.exists(), "Missing repo fixture: data/features/latest.parquet"
    assert repo_reg.exists(), "Missing repo fixture: data/regimes/latest.parquet"

    tmp_feat = data_dir / "features" / "latest.parquet"
    tmp_reg = data_dir / "regimes" / "latest.parquet"
    shutil.copy2(repo_feat, tmp_feat)
    shutil.copy2(repo_reg, tmp_reg)

    # Load in tmp (sanity)
    tmp_feats = pd.read_parquet(tmp_feat)
    tmp_regs = pd.read_parquet(tmp_reg)

    # Build a "pretrained" expert artifact INSIDE the test env (avoids pickle/numpy issues)
    # This mirrors tools/make_pretrained_expert.py but runs in-process for hermetic tests.
    df = tmp_feats.merge(tmp_regs, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_next_return"] = df["log_return_1"].shift(-1)
    df = df.dropna(subset=["target_next_return"]).reset_index(drop=True)

    expert_df = df[df["regime"] == "bullish"].copy()
    assert len(expert_df) > 50, "Not enough bullish rows in fixture to train expert"

    feature_cols = ["close", "log_return_1", "sma_10"]
    for c in feature_cols:
        assert c in expert_df.columns, f"Missing expected feature col in fixture: {c}"

    expert_model = Ridge(alpha=1.0, random_state=42)
    expert_model.fit(
        expert_df[feature_cols].to_numpy(),
        expert_df["target_next_return"].to_numpy(),
    )

    expert_path = models_dir / "pretrained" / "expert_bullish_ridge_v0.joblib"
    joblib.dump(
        {
            "model": expert_model,
            "feature_cols": feature_cols,
            "target_col": "target_next_return",
            "timestamp_col": "timestamp",
            "regime": "bullish",
            "model_name": "expert_bullish_ridge_v0",
        },
        expert_path,
    )
    assert expert_path.exists()

    # Act: run training in tmp_path with paths overridden
    cfg = TrainConfig(
        features_path=tmp_feat,
        regimes_path=tmp_reg,
        out_dir=data_dir / "runs",
        baseline_models_dir=models_dir / "baseline",
        pretrained_expert_path=expert_path,
        experts_dir=models_dir / "experts",
        # Keep MLflow local to tmp_path (train.py should normalize this on Windows)
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="test-market-regime-auto-switch",
    )

    run_id = run(cfg)
    assert isinstance(run_id, str) and len(run_id) > 0

    # Assert: baseline model exists
    baseline_root = models_dir / "baseline"
    assert baseline_root.exists()
    baseline_models = list(baseline_root.glob("*/model.joblib"))
    assert len(baseline_models) >= 1

    # Assert: expert registered + latest pointer exists
    expert_latest = models_dir / "experts" / "bullish" / "latest.joblib"
    assert expert_latest.exists()

    expert_versioned = list((models_dir / "experts" / "bullish").glob("*/model.joblib"))
    assert len(expert_versioned) >= 1

    # Assert: metadata exists next to versioned expert
    meta_files = list((models_dir / "experts" / "bullish").glob("*/metadata.json"))
    assert len(meta_files) >= 1

    # Assert: expert bundle is loadable
    bundle = joblib.load(expert_latest)
    assert "model" in bundle
    assert "feature_cols" in bundle

    # Assert: run artifacts exist
    run_jsons = list((data_dir / "runs").glob("*.json"))
    assert len(run_jsons) >= 1
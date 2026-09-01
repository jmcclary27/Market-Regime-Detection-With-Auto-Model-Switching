from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_offline_pipeline_writes_complete_safe_hold_artifact_set(tmp_path: Path) -> None:
    """The canonical offline path must be self-contained and not promote a model."""
    repo = Path(__file__).resolve().parents[1]
    run_ts = "20260203_120000Z"

    def run_pipeline(root: Path) -> dict[str, Path]:
        root.mkdir()
        env = os.environ | {
            "PYTHONPATH": str(repo),
            "PROJECT_ROOT": str(root),
            "DATA_DIR": str(root / "data"),
            "MLFLOW_TRACKING_URI": (root / "mlruns").as_uri(),
            "MLFLOW_ALLOW_FILE_STORE": "true",
            "EVAL_WF_TRAIN": "504",
            "EVAL_WF_VAL": "126",
            "EVAL_WF_TEST": "126",
            "EVAL_WF_STEP": "126",
        }
        subprocess.check_call(
            [sys.executable, "-m", "src.pipeline.run", "--offline", "--run-ts", run_ts],
            cwd=root,
            env=env,
        )
        paths = {
            "features": root / "data/features/latest.parquet",
            "manifest": root / "data/features/latest.manifest.json",
            "regimes": root / "data/regimes/latest.parquet",
            "predictions": root / "data/predictions/latest.parquet",
            "scorecard": root / "data/scorecards/latest.parquet",
            "walkforward": root / f"data/walkforward/portfolio_metrics_{run_ts}.parquet",
            "promotion": root / f"data/walkforward/promotion_{run_ts}.json",
            "deployment_history": root / "data/deployments/events.parquet",
            "lineage": root / f"artifacts/lineage/lineage_{run_ts}.json",
            "telemetry": root / f"artifacts/pipeline_runs/pipeline_run_{run_ts}.json",
            "registry": root / "registry/active_model.yaml",
        }
        assert all(path.exists() for path in paths.values())
        return paths

    first = run_pipeline(tmp_path / "first")
    second = run_pipeline(tmp_path / "second")

    promotion = json.loads(first["promotion"].read_text(encoding="utf-8"))
    assert promotion["promoted"] is False
    assert promotion["reason"] == "no_promotable_challenger"
    assert promotion["pointer_written"] is False

    events = pd.read_parquet(first["deployment_history"])
    assert events.iloc[-1]["decision"] == "hold"
    assert events.iloc[-1]["reason"] == "no_promotable_challenger"

    telemetry = json.loads(first["telemetry"].read_text(encoding="utf-8"))
    assert telemetry["status"] == "completed"
    assert telemetry["artifacts"]["promotion_decision_json"] == str(first["promotion"])

    for name in ("features", "regimes"):
        assert first[name].read_bytes() == second[name].read_bytes()

    first_predictions = pd.read_parquet(first["predictions"])
    second_predictions = pd.read_parquet(second["predictions"])
    root_specific_columns = {"features_path", "model_path"}
    comparable_columns = [
        column for column in first_predictions.columns if column not in root_specific_columns
    ]
    pd.testing.assert_frame_equal(
        first_predictions[comparable_columns], second_predictions[comparable_columns]
    )

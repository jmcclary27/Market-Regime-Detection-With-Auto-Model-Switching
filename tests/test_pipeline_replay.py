from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_offline_demo_is_deterministic_from_empty_working_directories(tmp_path: Path) -> None:
    """The recruiter demo must be repeatable without cloud data or model state."""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ | {"PYTHONPATH": str(repo)}

    def run_demo(workdir: Path) -> pd.DataFrame:
        workdir.mkdir()
        run_env = env | {
            "PROJECT_ROOT": str(workdir),
            "DATA_DIR": str(workdir / "data"),
            "MLFLOW_TRACKING_URI": (workdir / "mlruns").as_uri(),
            "MLFLOW_ALLOW_FILE_STORE": "true",
        }
        subprocess.check_call([sys.executable, "-m", "src.demo.run"], cwd=workdir, env=run_env)
        run_ts = "20240131_160000Z"
        predictions = workdir / "data" / "predictions" / f"predictions_{run_ts}.parquet"
        required = {
            "features": workdir / "data" / "features" / "latest.parquet",
            "regimes": workdir / "data" / "regimes" / "latest.parquet",
            "predictions": predictions,
            "scorecard": workdir / "data" / "scorecards" / "latest.parquet",
            "walkforward": workdir / "data" / "walkforward" / f"portfolio_metrics_{run_ts}.parquet",
            "promotion": workdir / "data" / "walkforward" / f"promotion_{run_ts}.json",
            "deployments": workdir / "data" / "deployments" / "events.parquet",
            "lineage": workdir / "artifacts" / "lineage" / f"lineage_{run_ts}.json",
            "telemetry": workdir / "artifacts" / "pipeline_runs" / f"pipeline_run_{run_ts}.json",
            "registry": workdir / "registry" / "active_model.yaml",
            "mlflow": workdir / "mlruns",
        }
        assert all(path.exists() for path in required.values())

        promotion = json.loads(required["promotion"].read_text(encoding="utf-8"))
        assert promotion["promoted"] is False
        assert promotion["reason"] == "no_promotable_challenger"
        assert pd.read_parquet(required["deployments"]).iloc[-1]["decision"] == "hold"
        return pd.read_parquet(predictions)

    first = run_demo(tmp_path / "first")
    second = run_demo(tmp_path / "second")
    comparable_columns = [
        column for column in first.columns if column not in {"features_path", "model_path"}
    ]
    pd.testing.assert_frame_equal(first[comparable_columns], second[comparable_columns])

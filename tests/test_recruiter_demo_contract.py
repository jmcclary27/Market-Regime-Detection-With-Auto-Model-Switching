from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_recruiter_demo_uses_an_isolated_named_runtime_volume() -> None:
    """Host MLflow state must never be mounted into the Linux demo container."""
    repo = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((repo / "docker-compose.yml").read_text(encoding="utf-8"))

    market = compose["services"]["market"]
    dashboard = compose["services"]["dashboard"]

    assert market["environment"]["PROJECT_ROOT"] == "/runtime"
    assert market["environment"]["DATA_DIR"] == "/runtime/data"
    assert market["environment"]["MLFLOW_TRACKING_URI"] == "file:/runtime/mlruns"
    assert market["volumes"] == ["market_runtime:/runtime"]
    assert dashboard["environment"]["MARKET_REGIME_DASHBOARD_ROOT"] == "/runtime"
    assert dashboard["volumes"] == ["market_runtime:/runtime"]
    assert "market_runtime" in compose["volumes"]


def test_recruiter_demo_helper_resets_only_its_compose_project_volume() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts" / "run_recruiter_demo.sh").read_text(encoding="utf-8")
    entrypoint = (repo / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")

    assert 'project_name="${RECRUITER_DEMO_PROJECT:-market-regime-recruiter-demo}"' in script
    assert "down --volumes --remove-orphans" in script
    assert "run --rm market demo" in script
    assert "up -d dashboard" in script
    assert "http://localhost:8501/_stcore/health" in script
    assert 'data_dir="${DATA_DIR:-$project_root/data}"' in entrypoint
    assert 'aws s3 sync "s3://${ARTIFACT_BUCKET}/data/raw" "$data_dir/raw"' in entrypoint

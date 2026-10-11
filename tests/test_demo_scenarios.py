from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.dashboard.data import load_dashboard_snapshot
from src.demo.run import DemoPrerequisiteError, prepare_demo_models
from src.pipeline.run import PipelineConfig, _write_runtime_settings


def test_demo_model_preflight_fails_closed_without_hmm_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source-models"
    source.mkdir()

    with pytest.raises(DemoPrerequisiteError, match="HMM"):
        prepare_demo_models(source_models=source, runtime_models=tmp_path / "runtime/models")


def test_dashboard_loads_immutable_demo_scenarios_after_refresh(tmp_path: Path) -> None:
    scenarios_dir = tmp_path / "artifacts/demo/scenarios"
    scenarios_dir.mkdir(parents=True)
    for name, promoted, reason in (
        ("promotion", True, "challenger beats incumbent"),
        ("rejection", False, "challenger sharpe does not beat incumbent"),
    ):
        (scenarios_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scenario": name,
                    "status": "completed",
                    "promotion": {"promoted": promoted, "reason": reason},
                    "post_promotion_inference": {"verified_active_model_id": "expert_bullish"},
                }
            ),
            encoding="utf-8",
        )

    first = load_dashboard_snapshot(tmp_path)
    second = load_dashboard_snapshot(tmp_path)

    assert [scenario["scenario"] for scenario in first.demo_scenarios] == ["promotion", "rejection"]
    assert first.demo_scenarios == second.demo_scenarios


def test_demo_pipeline_profile_keeps_the_real_hmm_enabled(tmp_path: Path) -> None:
    settings_path = _write_runtime_settings(
        PipelineConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            run_ts="20240202_160000Z",
            mode="pipeline",
            offline=True,
            demo=True,
            demo_scenario="promotion",
        )
    )

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    assert settings["regimes"]["method"] == "hmm"
    assert settings["regimes"]["hmm"]["artifacts_dir"] == str(
        tmp_path / "models" / "regimes" / "hmm"
    )

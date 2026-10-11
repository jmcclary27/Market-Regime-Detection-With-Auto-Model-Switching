from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_demo_fails_closed_without_external_published_artifacts(tmp_path: Path) -> None:
    """A clean CI checkout must not manufacture experts for the end-to-end demo."""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ | {"PYTHONPATH": str(repo)}
    workdir = tmp_path / "empty-runtime"
    workdir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "src.demo.run"],
        cwd=workdir,
        env=env
        | {
            "PROJECT_ROOT": str(workdir),
            "DATA_DIR": str(workdir / "data"),
            "DEMO_MODEL_SOURCE_ROOT": str(tmp_path / "missing-models"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "HMM source artifacts are incomplete" in result.stderr
    assert not (workdir / "models").exists()


def test_pipeline_summary_records_offline_provenance() -> None:
    from src.pipeline.telemetry import PipelineRunRecorder

    summary = PipelineRunRecorder(
        run_ts="20260715_000000Z", mode="pipeline", replay=False, offline=True
    ).build_summary(status="completed", artifacts={})

    assert summary["offline"] is True

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_offline_demo_is_deterministic_from_empty_working_directories(tmp_path: Path) -> None:
    """The recruiter demo must be repeatable without cloud data or model state."""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ | {"PYTHONPATH": str(repo)}

    def run_demo(workdir: Path) -> str:
        workdir.mkdir()
        subprocess.check_call([sys.executable, "-m", "src.demo.run"], cwd=workdir, env=env)
        predictions = workdir / "data" / "predictions" / "predictions_20240131160000.parquet"
        assert (workdir / "data" / "demo_settings.yaml").exists()
        assert predictions.exists()
        return hashlib.sha256(predictions.read_bytes()).hexdigest()

    assert run_demo(tmp_path / "first") == run_demo(tmp_path / "second")

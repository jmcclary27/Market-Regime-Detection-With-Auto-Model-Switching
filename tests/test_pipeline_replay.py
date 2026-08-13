from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.pipeline.run import _sha256_file


@pytest.mark.integration
def test_offline_demo_is_deterministic() -> None:
    """The recruiter demo must be repeatable without cloud data or model state."""
    repo = Path(__file__).resolve().parents[1]
    subprocess.check_call(["python", "-m", "src.demo.run"], cwd=repo)
    predictions = repo / "data" / "predictions" / "predictions_20240131160000.parquet"
    first_hash = _sha256_file(predictions)

    subprocess.check_call(["python", "-m", "src.demo.run"], cwd=repo)
    assert _sha256_file(predictions) == first_hash

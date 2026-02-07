# tests/test_pipeline_replay.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.pipeline.run import (
    _sha256_file,  # ok since it's your module, or copy a tiny hasher in test
)


@pytest.mark.integration
def test_pipeline_replay_is_deterministic(tmp_path: Path) -> None:
    # Run inside repo root
    repo = Path(__file__).resolve().parents[1]

    run_ts = "20990101_000000Z"  # fixed, far future so no accidental collisions

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(repo)

    # 1) normal run
    subprocess.check_call(
        ["python", "-m", "src.pipeline.run", "--mode", "pipeline", "--run-ts", run_ts],
        cwd=repo,
        env=env,
    )

    pred = repo / "data" / "predictions" / f"predictions_{run_ts}.parquet"
    assert pred.exists()

    pred_hash = _sha256_file(pred)

    # 2) replay run
    subprocess.check_call(
        ["python", "-m", "src.pipeline.run", "--mode", "pipeline", "--replay", run_ts],
        cwd=repo,
        env=env,
    )

    # Ensure original wasn't overwritten
    assert _sha256_file(pred) == pred_hash

    # Ensure replay output exists and matches original
    replay_pred = repo / "data" / "predictions" / f"predictions_replay_{run_ts}.parquet"
    assert replay_pred.exists()
    assert _sha256_file(replay_pred) == pred_hash

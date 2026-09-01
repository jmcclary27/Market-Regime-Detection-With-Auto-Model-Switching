from __future__ import annotations

from pathlib import Path


def test_airflow_dags_invoke_only_the_canonical_pipeline_runner() -> None:
    one_off = Path("airflow/dags/pipeline_one_off.py").read_text(encoding="utf-8")
    scheduled = Path("airflow/dags/pipeline_scheduled.py").read_text(encoding="utf-8")

    for dag in (one_off, scheduled):
        assert "python -m src.pipeline.run" in dag
        assert 'task_id="run_pipeline"' in dag

    for stale_module in (
        "src.poll.run",
        "src.features.run",
        "src.regimes.run",
        "src.inference.run_batch_predict",
        "src.switcher.run",
    ):
        assert stale_module not in one_off


def test_dvc_wrapper_invokes_the_offline_canonical_pipeline() -> None:
    dvc = Path("dvc.yaml").read_text(encoding="utf-8")

    assert "offline_pipeline:" in dvc
    assert "python -m src.pipeline.run --offline --run-ts ${run_ts}" in dvc
    assert "always_changed: true" in dvc
    assert "src.deploy.switcher" not in dvc

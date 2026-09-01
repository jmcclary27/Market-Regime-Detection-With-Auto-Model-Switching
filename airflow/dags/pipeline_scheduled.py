from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}


def _cmd(extra: str = "") -> str:
    """Run the single canonical pipeline entrypoint from the baked project root."""
    return f"cd /opt/project && python -m src.pipeline.run {extra}".strip()


with DAG(
    dag_id="pipeline_scheduled",
    description="Scheduled end-to-end pipeline run",
    start_date=datetime(2025, 1, 1),
    schedule="0 * * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["market", "scheduled"],
) as dag:
    run_ts = "{{ data_interval_end.in_timezone('UTC').strftime('%Y%m%d_%H%M%SZ') }}"

    BashOperator(
        task_id="run_pipeline",
        bash_command=_cmd(f'--run-ts "{run_ts}"'),
    )

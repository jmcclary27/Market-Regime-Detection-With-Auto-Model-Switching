from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _cmd(extra: str = "") -> str:
    """Run the single canonical pipeline entrypoint from the baked project root."""
    return f"cd /opt/project && python -m src.pipeline.run {extra}".strip()


with DAG(
    dag_id="pipeline_one_off",
    description="Manual one-off canonical pipeline run; supply run_ts in conf",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["market", "oneoff"],
) as dag:
    run_ts = "{{ dag_run.conf.get('run_ts', data_interval_end.in_timezone('UTC').strftime('%Y%m%d_%H%M%SZ')) }}"

    BashOperator(
        task_id="run_pipeline",
        bash_command=_cmd(f'--run-ts "{run_ts}"'),
    )

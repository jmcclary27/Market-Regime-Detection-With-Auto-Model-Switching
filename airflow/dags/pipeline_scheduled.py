from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}


def _cmd(module: str, extra: str = "") -> str:
    return f"cd /opt/project && python -m {module} {extra}".strip()


with DAG(
    dag_id="pipeline_scheduled",
    description="Scheduled runs, run_ts derived from Airflow logical date",
    start_date=datetime(2025, 1, 1),
    schedule="0 * * * *",  # hourly, mimic cron
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["market", "scheduled"],
) as dag:
    # Use the logical date to build a deterministic run_ts.
    # Example: 2026-01-18 13:00:00 -> 20260118_130000Z
    run_ts = "{{ data_interval_end.in_timezone('UTC').strftime('%Y%m%d_%H%M%SZ') }}"

    poll = BashOperator(
        task_id="poll",
        bash_command=_cmd("src.poll.run", f'--run-ts "{run_ts}"'),
    )
    features = BashOperator(
        task_id="features",
        bash_command=_cmd("src.features.run", f'--run-ts "{run_ts}"'),
    )
    regimes = BashOperator(
        task_id="regimes",
        bash_command=_cmd("src.regimes.run", f'--run-ts "{run_ts}"'),
    )
    predict = BashOperator(
        task_id="predict",
        bash_command=_cmd("src.inference.run_batch_predict", f'--run-ts "{run_ts}"'),
    )
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=_cmd("src.eval.run_evaluator", f'--run-ts "{run_ts}"'),
    )
    switch = BashOperator(
        task_id="switch",
        bash_command=_cmd("src.switcher.run", f'--run-ts "{run_ts}"'),
    )

    poll >> features >> regimes >> predict >> evaluate >> switch

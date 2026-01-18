from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _cmd(module: str, extra: str = "") -> str:
    # Uses the project code baked into the image at /opt/project
    # and writes to /opt/project/data + /opt/project/models (mounted).
    return f"cd /opt/project && python -m {module} {extra}".strip()


with DAG(
    dag_id="pipeline_one_off",
    description="Manual one-off run, supply run_ts in conf",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["market", "oneoff"],
) as dag:
    # Expect conf like: {"run_ts": "20260118_120000Z"}
    # If you already have your own timestamp defaulting inside the module, you can omit passing run_ts.
    poll = BashOperator(
        task_id="poll",
        bash_command=_cmd("src.poll.run", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""),
    )

    features = BashOperator(
        task_id="features",
        bash_command=_cmd("src.features.run", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""),
    )

    regimes = BashOperator(
        task_id="regimes",
        bash_command=_cmd("src.regimes.run", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""),
    )

    predict = BashOperator(
        task_id="predict",
        bash_command=_cmd(
            "src.inference.run_batch_predict", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""
        ),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=_cmd(
            "src.eval.run_evaluator", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""
        ),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    switch = BashOperator(
        task_id="switch",
        bash_command=_cmd("src.switcher.run", "--run-ts \"{{ dag_run.conf.get('run_ts', '') }}\""),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    poll >> features >> regimes >> predict >> [evaluate, switch]

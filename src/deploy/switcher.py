from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass(frozen=True)
class SwitchConfig:
    window_type: str = "count"
    window_value: int = 100
    metric_name: str = "mse"
    promote_margin: float = 0.0
    rollback_margin: float = 0.0


EVENT_COLUMNS = [
    "ts",
    "event_type",
    "active_model_id_before",
    "candidate_model_id",
    "active_model_id_after",
    "window_type",
    "window_value",
    "metric_name",
    "active_metric_value",
    "candidate_metric_value",
    "decision",
    "reason",
]


def append_event(events_path: Path, event: Dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)

    row = {col: event.get(col) for col in EVENT_COLUMNS}
    df_new = pd.DataFrame([row])

    if events_path.exists():
        df_existing = pd.read_parquet(events_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(events_path, index=False)


def run_switcher(*, data_dir: Path, config: SwitchConfig) -> None:
    """
    v0 placeholder:
    - count-based canary window concept (config.window_value)
    - no switching yet
    - logs an evaluation event to an append-only parquet
    """
    events_path = data_dir / "deployments" / "events.parquet"

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": "canary_evaluated",
        "active_model_id_before": "baseline@v1",
        "candidate_model_id": "candidate@v1",
        "active_model_id_after": "baseline@v1",
        "window_type": config.window_type,
        "window_value": config.window_value,
        "metric_name": config.metric_name,
        "active_metric_value": None,
        "candidate_metric_value": None,
        "decision": "no_action",
        "reason": "v0 placeholder evaluation",
    }

    append_event(events_path, event)


def main() -> None:
    config = SwitchConfig()
    run_switcher(data_dir=Path("data"), config=config)


if __name__ == "__main__":
    main()
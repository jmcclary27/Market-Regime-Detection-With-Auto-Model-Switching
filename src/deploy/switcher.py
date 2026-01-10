from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ---------- Config ----------

@dataclass(frozen=True)
class SwitchConfig:
    window_type: str = "count"
    window_value: int = 100
    metric_name: str = "mse"
    promote_margin: float = 0.0
    rollback_margin: float = 0.0


# ---------- Event schema ----------

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


# ---------- Helpers ----------

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


def infer_model_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["model_id", "model", "model_name", "model_key", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_latest_scorecard(scorecards_dir: Path) -> Optional[pd.DataFrame]:
    path = scorecards_dir / "latest.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def extract_metric(
    df: pd.DataFrame,
    *,
    model_id: str,
    metric_name: str,
) -> Optional[float]:
    id_col = infer_model_id_col(df)
    if id_col is None:
        return None
    if metric_name not in df.columns:
        return None

    rows = df[df[id_col] == model_id]
    if rows.empty:
        return None

    val = rows.iloc[0][metric_name]
    if pd.isna(val):
        return None
    return float(val)


# ---------- Switcher ----------

def run_switcher(
    *,
    data_dir: Path,
    config: SwitchConfig,
    active_model_id: str = "baseline@v1",
    candidate_model_id: str = "candidate@v1",
) -> None:
    events_path = data_dir / "deployments" / "events.parquet"
    scorecards_dir = data_dir / "scorecards"

    scorecard = load_latest_scorecard(scorecards_dir)

    if scorecard is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": candidate_model_id,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "reason": "scorecard_missing",
            },
        )
        return

    id_col = infer_model_id_col(scorecard)
    if id_col is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": candidate_model_id,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "reason": "scorecard_missing_model_id_column",
            },
        )
        return

    active_metric = extract_metric(
        scorecard,
        model_id=active_model_id,
        metric_name=config.metric_name,
    )

    candidate_metric = extract_metric(
        scorecard,
        model_id=candidate_model_id,
        metric_name=config.metric_name,
    )

    append_event(
        events_path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": "canary_evaluated",
            "active_model_id_before": active_model_id,
            "candidate_model_id": candidate_model_id,
            "active_model_id_after": active_model_id,
            "window_type": config.window_type,
            "window_value": config.window_value,
            "metric_name": config.metric_name,
            "active_metric_value": active_metric,
            "candidate_metric_value": candidate_metric,
            "decision": "no_action",
            "reason": "metrics_logged",
        },
    )


# ---------- CLI ----------

def main() -> None:
    config = SwitchConfig()
    run_switcher(
        data_dir=Path("data"),
        config=config,
        active_model_id="baseline@v1",
        candidate_model_id="candidate@v1",
    )


if __name__ == "__main__":
    main()
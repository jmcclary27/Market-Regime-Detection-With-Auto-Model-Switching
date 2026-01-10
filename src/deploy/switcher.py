from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ---------- Config ----------

@dataclass(frozen=True)
class SwitchConfig:
    """
    Canary switcher config (v0: count-based only).

    metric_name:
      - Your scorecards use: "rmse", "mae"
      - Lower is better for both.
    """
    window_type: str = "count"
    window_value: int = 100
    metric_name: str = "rmse"
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
    "n",
    "metric_name",
    "active_metric_value",
    "candidate_metric_value",
    "decision",
    "reason",
]


# ---------- Helpers ----------

def append_event(events_path: Path, event: Dict[str, Any]) -> None:
    """
    Append a single event row to an append-only parquet log.
    Avoid pandas concat dtype warnings by enforcing schema.
    """
    events_path.parent.mkdir(parents=True, exist_ok=True)

    # Build single-row DF with enforced column order
    row = {col: event.get(col) for col in EVENT_COLUMNS}
    df_new = pd.DataFrame([row], columns=EVENT_COLUMNS)

    if not events_path.exists():
        # First write: just write schema-consistent DF
        df_new.to_parquet(events_path, index=False)
        return

    df_existing = pd.read_parquet(events_path)

    # Ensure both frames have exactly the same columns in the same order
    df_existing = df_existing.reindex(columns=EVENT_COLUMNS)
    df_new = df_new.reindex(columns=EVENT_COLUMNS)

    # Concatenate (schema-aligned)
    df = pd.concat([df_existing, df_new], ignore_index=True)

    df.to_parquet(events_path, index=False)


def infer_model_id_col(df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Your scorecards currently use 'model_name'. We also support other common names.
    """
    if df is None:
        return None

    candidates = ["model_name", "model_id", "model", "model_key", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_latest_scorecard(scorecards_dir: Path) -> Optional[pd.DataFrame]:
    path = scorecards_dir / "latest.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _prefer_overall_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer the overall aggregate row if present.
    - scope == "overall"
    - regime is null
    """
    sub = df
    if "scope" in sub.columns:
        sub = sub[sub["scope"] == "overall"]
    if "regime" in sub.columns:
        # regime None shows up as NaN in pandas
        sub = sub[sub["regime"].isna()]
    return sub


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

    sub = _prefer_overall_rows(df)

    rows = sub[sub[id_col] == model_id]
    if rows.empty:
        return None

    val = rows.iloc[0][metric_name]
    if pd.isna(val):
        return None
    return float(val)


def extract_n(df: pd.DataFrame, *, model_id: str) -> Optional[int]:
    """
    Extract sample count for the chosen (overall) row.
    """
    id_col = infer_model_id_col(df)
    if id_col is None or "n" not in df.columns:
        return None

    sub = _prefer_overall_rows(df)

    rows = sub[sub[id_col] == model_id]
    if rows.empty:
        return None

    val = rows.iloc[0]["n"]
    if pd.isna(val):
        return None
    return int(val)


# ---------- Switcher ----------

def run_switcher(
    *,
    data_dir: Path,
    config: SwitchConfig,
    active_model_id: str = "baseline",
    candidate_model_id: str = "expert_bullish",
) -> None:
    """
    v0 behavior (PR 8 Step 2):
    - Count-based window concept only (window_value is recorded, not enforced yet)
    - Reads latest scorecard from data/scorecards/latest.parquet
    - Logs active vs candidate metric into data/deployments/events.parquet
    - Does NOT promote/rollback yet (decision is always "no_action")
    """
    events_path = data_dir / "deployments" / "events.parquet"
    scorecards_dir = data_dir / "scorecards"

    scorecard = load_latest_scorecard(scorecards_dir)

    # No scorecard? Log it and return (don’t crash).
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
                "n": None,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "reason": "scorecard_missing",
            },
        )
        return

    # Scorecard exists but no recognizable model id column? Log it and return.
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
                "n": None,
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

    # This is just a convenience debug signal; uses active model row.
    n_val = extract_n(scorecard, model_id=active_model_id)

    reason = "metrics_logged"
    if active_metric is None or candidate_metric is None:
        reason = "metrics_missing_for_model_id_or_metric"

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
            "n": n_val,
            "metric_name": config.metric_name,
            "active_metric_value": active_metric,
            "candidate_metric_value": candidate_metric,
            "decision": "no_action",
            "reason": reason,
        },
    )


# ---------- CLI ----------

def main() -> None:
    config = SwitchConfig()
    run_switcher(
        data_dir=Path("data"),
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )


if __name__ == "__main__":
    main()
# src/deploy/switcher.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    # Candidate must be better than active by at least promote_margin to promote
    promote_margin: float = 0.0

    # Candidate is considered clearly worse if it is worse than active by rollback_margin
    rollback_margin: float = 0.0

    # Whether to actually update registry/active_model.yaml when promoting
    update_registry_on_promote: bool = True


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


def append_event(events_path: Path, event: dict[str, Any]) -> None:
    """
    Append a single event row to an append-only parquet log.
    Avoid pandas concat dtype warnings by enforcing schema.
    """
    events_path.parent.mkdir(parents=True, exist_ok=True)

    row = {col: event.get(col) for col in EVENT_COLUMNS}
    df_new = pd.DataFrame([row], columns=EVENT_COLUMNS)

    if not events_path.exists():
        df_new.to_parquet(events_path, index=False)
        return

    df_existing = pd.read_parquet(events_path)

    df_existing = df_existing.reindex(columns=EVENT_COLUMNS)
    df_new = df_new.reindex(columns=EVENT_COLUMNS)

    df = pd.concat([df_existing, df_new], ignore_index=True)

    # Optional: keep n as a nullable integer column when possible
    if "n" in df.columns:
        try:
            df["n"] = df["n"].astype("Int64")
        except Exception:
            pass

    df.to_parquet(events_path, index=False)


def write_active_model_yaml(registry_path: Path, model_id: str) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(f"model_id: {model_id}\n", encoding="utf-8")


def infer_model_id_col(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    candidates = ["model_name", "model_id", "model", "model_key", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_latest_scorecard(scorecards_dir: Path) -> pd.DataFrame | None:
    path = scorecards_dir / "latest.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _prefer_overall_rows(df: pd.DataFrame) -> pd.DataFrame:
    sub = df
    if "scope" in sub.columns:
        sub = sub[sub["scope"] == "overall"]
    if "regime" in sub.columns:
        sub = sub[sub["regime"].isna()]
    return sub


def extract_metric(
    df: pd.DataFrame,
    *,
    model_id: str,
    metric_name: str,
) -> float | None:
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


def extract_n(df: pd.DataFrame, *, model_id: str) -> int | None:
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


def decide(
    *,
    active_metric: float,
    candidate_metric: float,
    promote_margin: float,
    rollback_margin: float,
) -> tuple[str, str]:
    """
    Lower is better.
    Returns (decision, reason).
    decision ∈ {"promote", "rollback", "hold"}
    """
    # candidate better by margin => promote
    if candidate_metric <= active_metric - promote_margin:
        return "promote", "candidate_better_than_active"

    # candidate worse by rollback margin => rollback
    if candidate_metric >= active_metric + rollback_margin:
        return "rollback", "candidate_worse_than_active"

    return "hold", "within_margins"


# ---------- Switcher ----------


def run_switcher(
    *,
    data_dir: Path,
    config: SwitchConfig,
    active_model_id: str = "baseline",
    candidate_model_id: str = "expert_bullish",
) -> None:
    """
    Step 3 behavior:
    - Load latest scorecard
    - Extract active vs candidate metric (overall)
    - Decide: promote / rollback / hold
    - Log deployment event
    - If promote and update_registry_on_promote=True, update registry/active_model.yaml
    """
    events_path = data_dir / "deployments" / "events.parquet"
    scorecards_dir = data_dir / "scorecards"

    scorecard = load_latest_scorecard(scorecards_dir)

    if scorecard is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
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

    id_col = infer_model_id_col(scorecard)
    if id_col is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
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
    n_val = extract_n(scorecard, model_id=active_model_id)

    if active_metric is None or candidate_metric is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
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
                "reason": "metrics_missing_for_model_id_or_metric",
            },
        )
        return

    decision, reason = decide(
        active_metric=active_metric,
        candidate_metric=candidate_metric,
        promote_margin=config.promote_margin,
        rollback_margin=config.rollback_margin,
    )

    active_after = active_model_id
    event_type = "canary_evaluated"

    if decision == "promote":
        active_after = candidate_model_id
        event_type = "promoted"
        if config.update_registry_on_promote:
            project_root = data_dir.parent
            registry_path = project_root / "registry" / "active_model.yaml"
            write_active_model_yaml(registry_path, candidate_model_id)

    elif decision == "rollback":
        event_type = "rollback"

    else:  # hold
        event_type = "hold"

    append_event(
        events_path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "active_model_id_before": active_model_id,
            "candidate_model_id": candidate_model_id,
            "active_model_id_after": active_after,
            "window_type": config.window_type,
            "window_value": config.window_value,
            "n": n_val,
            "metric_name": config.metric_name,
            "active_metric_value": active_metric,
            "candidate_metric_value": candidate_metric,
            "decision": decision,
            "reason": reason,
        },
    )


def run() -> None:
    main()


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

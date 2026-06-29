from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEPLOYMENT_EVENT_COLUMNS = [
    "ts",
    "run_ts",
    "source",
    "event_type",
    "decision",
    "active_model_id_before",
    "candidate_model_id",
    "active_model_id_after",
    "window_type",
    "window_value",
    "n",
    "metric_name",
    "active_metric_value",
    "candidate_metric_value",
    "metric_delta",
    "active_max_drawdown",
    "candidate_max_drawdown",
    "promotion_guard_allowed",
    "pointer_written",
    "reason",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _coerce_int(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_deployment_event(event: dict[str, Any]) -> dict[str, Any]:
    row = {column: event.get(column) for column in DEPLOYMENT_EVENT_COLUMNS}
    row["ts"] = str(row.get("ts") or utc_now_iso())
    row["run_ts"] = None if row.get("run_ts") in (None, "") else str(row["run_ts"])
    row["source"] = str(row.get("source") or "unknown")
    row["event_type"] = str(row.get("event_type") or "unknown")
    row["decision"] = str(row.get("decision") or "unknown")
    row["active_model_id_before"] = (
        None
        if row.get("active_model_id_before") in (None, "")
        else str(row["active_model_id_before"])
    )
    row["candidate_model_id"] = (
        None if row.get("candidate_model_id") in (None, "") else str(row["candidate_model_id"])
    )
    row["active_model_id_after"] = (
        None if row.get("active_model_id_after") in (None, "") else str(row["active_model_id_after"])
    )
    row["window_type"] = None if row.get("window_type") in (None, "") else str(row["window_type"])
    row["window_value"] = _coerce_int(row.get("window_value"))
    row["n"] = _coerce_int(row.get("n"))
    row["metric_name"] = None if row.get("metric_name") in (None, "") else str(row["metric_name"])
    row["active_metric_value"] = _coerce_float(row.get("active_metric_value"))
    row["candidate_metric_value"] = _coerce_float(row.get("candidate_metric_value"))
    row["metric_delta"] = _coerce_float(row.get("metric_delta"))
    if row["metric_delta"] is None:
        active = row["active_metric_value"]
        candidate = row["candidate_metric_value"]
        if active is not None and candidate is not None:
            row["metric_delta"] = candidate - active
    row["active_max_drawdown"] = _coerce_float(row.get("active_max_drawdown"))
    row["candidate_max_drawdown"] = _coerce_float(row.get("candidate_max_drawdown"))
    if row.get("promotion_guard_allowed") is not None:
        row["promotion_guard_allowed"] = bool(row["promotion_guard_allowed"])
    if row.get("pointer_written") is not None:
        row["pointer_written"] = bool(row["pointer_written"])
    row["reason"] = None if row.get("reason") in (None, "") else str(row["reason"])
    return row


def append_deployment_event(events_path: Path, event: dict[str, Any]) -> dict[str, Any]:
    events_path.parent.mkdir(parents=True, exist_ok=True)

    row = normalize_deployment_event(event)
    df_new = pd.DataFrame([row], columns=DEPLOYMENT_EVENT_COLUMNS)

    if events_path.exists():
        df_existing = pd.read_parquet(events_path).reindex(columns=DEPLOYMENT_EVENT_COLUMNS)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    for column in ("window_value", "n"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in (
        "active_metric_value",
        "candidate_metric_value",
        "metric_delta",
        "active_max_drawdown",
        "candidate_max_drawdown",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("promotion_guard_allowed", "pointer_written"):
        if column in df.columns:
            df[column] = df[column].astype("boolean")

    df.to_parquet(events_path, index=False)
    return row

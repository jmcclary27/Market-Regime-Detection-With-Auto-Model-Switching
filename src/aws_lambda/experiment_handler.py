"""S3-triggered executor for the frozen three-portfolio paper experiment."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiment.engine import initial_state, process_daily_bar
from src.experiment.manifest import load_manifest
from src.experiment.reporting import build_dashboard_payload

from .live_sim_handler import _get_object_bytes, _parse_live_completion, parse_live_sim_s3_event


class ExperimentExecutionError(RuntimeError):
    """Raised when a completion cannot safely update the frozen experiment."""


def _json_from_s3(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        raw = _get_object_bytes(s3, bucket=bucket, key=key)
    except Exception as exc:
        code = getattr(getattr(exc, "response", {}), "get", lambda _key, _default=None: None)(
            "Error"
        )
        if isinstance(code, dict) and code.get("Code") in {"NoSuchKey", "NotFound"}:
            return None
        raise
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ExperimentExecutionError(f"Expected JSON object in {key}")
    return parsed


def _put_json(s3: Any, *, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def _event_predictions(s3: Any, *, bucket: str, completion: Any) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="experiment-predictions-") as temp_dir:
        target = Path(temp_dir) / "predictions.parquet"
        response = s3.get_object(
            Bucket=bucket,
            Key=completion.predictions.key,
            VersionId=completion.predictions.version_id,
        )
        target.write_bytes(response["Body"].read())
        return pd.read_parquet(target)


def process_record(
    s3: Any, record: Any, *, manifest_key: str, dashboard_bucket: str | None = None
) -> dict[str, Any]:
    completion = _parse_live_completion(
        _get_object_bytes(s3, bucket=record.bucket, key=record.key, version_id=record.version_id),
        record=record,
    )
    if completion is None:
        return {"status": "ignored_non_live_sim", "run_id": record.run_id}
    manifest_payload = _json_from_s3(s3, bucket=record.bucket, key=manifest_key)
    if manifest_payload is None:
        raise ExperimentExecutionError("Frozen experiment manifest is missing")
    with tempfile.TemporaryDirectory(prefix="experiment-manifest-") as temp_dir:
        path = Path(temp_dir) / "manifest.json"
        path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        manifest = load_manifest(path)
    prefix = f"experiment/{manifest.experiment_id}"
    result_key = f"{prefix}/runs/{completion.run_id}/{record.version_id}.json"
    if _json_from_s3(s3, bucket=record.bucket, key=result_key) is not None:
        return {"status": "already_completed", "run_id": completion.run_id}
    state_key = f"{prefix}/state/current.json"
    events_key = f"{prefix}/state/events.json"
    state = _json_from_s3(s3, bucket=record.bucket, key=state_key) or initial_state(manifest)
    events = _json_from_s3(s3, bucket=record.bucket, key=events_key) or {"events": []}
    if not isinstance(events.get("events"), list):
        raise ExperimentExecutionError("Experiment events projection is invalid")
    predictions = _event_predictions(s3, bucket=record.bucket, completion=completion)
    context = completion.context
    new_state, event = process_daily_bar(
        state=state,
        manifest=manifest,
        predictions=predictions,
        bar_timestamp_utc=context.bar_timestamp_utc,
        row_id=context.row_id,
        regime=context.regime,
        regime_confidence=None,
        open_price=context.open_price,
        close_price=context.close_price,
    )
    transaction = {"completion_version_id": record.version_id, "state": new_state, "event": event}
    _put_json(s3, bucket=record.bucket, key=f"{prefix}/state/transaction.json", payload=transaction)
    updated_events = [*events["events"], event]
    _put_json(s3, bucket=record.bucket, key=state_key, payload=new_state)
    _put_json(s3, bucket=record.bucket, key=events_key, payload={"events": updated_events})
    dashboard_payload = build_dashboard_payload(manifest=manifest, events=updated_events)
    _put_json(
        s3,
        bucket=record.bucket,
        key=f"{prefix}/public/latest.json",
        payload=dashboard_payload,
    )
    if dashboard_bucket:
        _put_json(s3, bucket=dashboard_bucket, key="latest.json", payload=dashboard_payload)
    _put_json(
        s3, bucket=record.bucket, key=result_key, payload={"status": "completed", "event": event}
    )
    return {"status": "completed", "run_id": completion.run_id}


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context
    manifest_key = os.environ.get("EXPERIMENT_MANIFEST_KEY", "experiment/manifest.json")
    dashboard_bucket = os.environ.get("EXPERIMENT_DASHBOARD_BUCKET")
    import boto3

    s3 = boto3.client("s3")
    results = [
        process_record(s3, record, manifest_key=manifest_key, dashboard_bucket=dashboard_bucket)
        for record in parse_live_sim_s3_event(event)
    ]
    return {"processed": len(results), "results": results}

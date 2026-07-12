"""S3-triggered, paper-only live-simulation executor.

The inference Lambda emits a completion marker only after active-plus-shadow
predictions are durable. This handler consumes live-simulation completions,
executes any signal queued by the prior bar at the current bar's open, and
queues the current active-model signal for the next received bar. State is
stored in versioned S3 objects; Terraform reserves concurrency at one so the
single paper account is mutated serially.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote_plus

import pandas as pd

from src.trading.execution import ExecutionConfig, execute_signal, log_trade
from src.trading.live_sim_shared import (
    ActivePrediction,
    is_live_eligible_prediction,
    load_loop_state,
    save_loop_state,
)
from src.trading.signals import SignalConfig, prediction_to_signal
from src.trading.state import load_account_state, log_account_snapshot, save_account_state

from .inference_contract import LiveSimContext, S3ObjectRef, request_version_token

LOG = logging.getLogger(__name__)
_COMPLETION_TOKEN_LENGTH = 24


class LiveSimContractError(ValueError):
    """Raised for malformed inference completion or live-sim state contracts."""


class LiveSimExecutionError(RuntimeError):
    """Raised when a valid live-sim completion cannot be executed safely."""


@dataclass(frozen=True)
class S3CompletionEvent:
    bucket: str
    key: str
    version_id: str
    run_id: str
    output_prefix: str


@dataclass(frozen=True)
class LiveInferenceCompletion:
    run_id: str
    request_version_id: str
    predictions: S3ObjectRef
    context: LiveSimContext


@dataclass(frozen=True)
class LiveSimStateKeys:
    transaction: str
    account: str
    loop: str
    trades: str
    equity: str


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveSimContractError(f"{field} must be a JSON object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiveSimContractError(f"{field} must be a non-empty string")
    return value.strip()


def _event_output_prefix(key: str) -> tuple[str, str]:
    """Validate a live-simulation completion key and return (run_id, prefix)."""
    if key.startswith("/") or "\\" in key:
        raise LiveSimContractError("completion key must be a normalized relative S3 key")
    parts = key.split("/")
    if (
        len(parts) != 7
        or parts[:3] != ["inference", "live-sim", "runs"]
        or parts[4] != "outputs"
        or parts[-1] != "completed.json"
    ):
        raise LiveSimContractError(
            "completion key must be inference/live-sim/runs/<run_id>/outputs/<token>/completed.json"
        )
    run_id = parts[3]
    token = parts[5]
    if not run_id or any(
        char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
        for char in run_id
    ):
        raise LiveSimContractError("completion key has an invalid run_id")
    if len(token) != _COMPLETION_TOKEN_LENGTH or any(
        char not in "0123456789abcdef" for char in token
    ):
        raise LiveSimContractError("completion key has an invalid request-version token")
    return run_id, "/".join(parts[:-1])


def parse_live_sim_s3_event(event: Any) -> list[S3CompletionEvent]:
    """Parse versioned S3 events for only the live-sim completion prefix."""
    event_data = _mapping(event, field="event")
    raw_records = event_data.get("Records")
    if not isinstance(raw_records, list) or not raw_records:
        raise LiveSimContractError("event.Records must be a non-empty list")

    records: list[S3CompletionEvent] = []
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, field=f"event.Records[{index}]")
        s3 = _mapping(record.get("s3"), field=f"event.Records[{index}].s3")
        bucket_data = _mapping(s3.get("bucket"), field=f"event.Records[{index}].s3.bucket")
        object_data = _mapping(s3.get("object"), field=f"event.Records[{index}].s3.object")
        bucket = _string(bucket_data.get("name"), field=f"event.Records[{index}].s3.bucket.name")
        key = unquote_plus(
            _string(object_data.get("key"), field=f"event.Records[{index}].s3.object.key")
        )
        version_id = _string(
            object_data.get("versionId"), field=f"event.Records[{index}].s3.object.versionId"
        )
        if version_id.lower() == "null":
            raise LiveSimContractError("S3 bucket versioning is required for live-sim events")
        run_id, output_prefix = _event_output_prefix(key)
        records.append(
            S3CompletionEvent(
                bucket=bucket,
                key=key,
                version_id=version_id,
                run_id=run_id,
                output_prefix=output_prefix,
            )
        )
    return records


def _parse_live_completion(
    payload: bytes, *, record: S3CompletionEvent
) -> LiveInferenceCompletion | None:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveSimContractError("inference completion must be UTF-8 JSON") from exc
    data = _mapping(raw, field="inference completion")
    if data.get("schema_version") != 1 or data.get("status") != "completed":
        raise LiveSimContractError("inference completion has an unsupported schema or status")
    if data.get("mode") != "live_sim":
        return None

    run_id = _string(data.get("run_id"), field="inference completion.run_id")
    if run_id != record.run_id:
        raise LiveSimContractError("inference completion run_id does not match its S3 key")

    request = _mapping(data.get("request"), field="inference completion.request")
    request_version_id = _string(
        request.get("version_id"), field="inference completion.request.version_id"
    )
    expected_output_prefix = (
        f"inference/live-sim/runs/{run_id}/outputs/{request_version_token(request_version_id)}"
    )
    if record.output_prefix != expected_output_prefix:
        raise LiveSimContractError(
            "completion output prefix does not match its immutable request-version token"
        )
    outputs = _mapping(data.get("outputs"), field="inference completion.outputs")
    predictions = S3ObjectRef.from_payload(outputs.get("predictions"), field="outputs.predictions")
    if not predictions.key.startswith(f"{record.output_prefix}/"):
        raise LiveSimContractError(
            "prediction output is not scoped to the completion output prefix"
        )

    context = LiveSimContext.from_payload(data.get("live_sim"))
    return LiveInferenceCompletion(
        run_id=run_id,
        request_version_id=request_version_id,
        predictions=predictions,
        context=context,
    )


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return str(code) if code is not None else None


def _get_object_bytes(s3: Any, *, bucket: str, key: str, version_id: str | None = None) -> bytes:
    args: dict[str, str] = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        args["VersionId"] = version_id
    response = s3.get_object(**args)
    try:
        body = response["Body"].read()
    except KeyError as exc:
        raise LiveSimExecutionError(f"S3 object has no body: s3://{bucket}/{key}") from exc
    if not isinstance(body, bytes):
        raise LiveSimExecutionError(f"S3 object body is not bytes: s3://{bucket}/{key}")
    return body


def _download_pinned(s3: Any, *, bucket: str, ref: S3ObjectRef, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, ref.key, str(destination), ExtraArgs={"VersionId": ref.version_id})
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != ref.sha256:
        raise LiveSimExecutionError(
            f"prediction digest mismatch for s3://{bucket}/{ref.key} version {ref.version_id}"
        )


def _download_optional_state(s3: Any, *, bucket: str, key: str, destination: Path) -> str | None:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        body = response["Body"].read()
    except KeyError as exc:
        raise LiveSimExecutionError(f"state object has no body: s3://{bucket}/{key}") from exc
    if not isinstance(body, bytes):
        raise LiveSimExecutionError(f"state object body is not bytes: s3://{bucket}/{key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    version_id = response.get("VersionId")
    return str(version_id) if version_id else None


def _put_file(s3: Any, *, bucket: str, key: str, path: Path, content_type: str) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as handle:
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=handle,
            ContentType=content_type,
            Metadata={"sha256": digest},
        )
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise LiveSimExecutionError("S3 bucket versioning must be enabled for live-sim state")
    return {
        "key": key,
        "version_id": version_id,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _put_json(s3: Any, *, bucket: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveSimExecutionError("live-sim result is not JSON serializable") from exc
    digest = hashlib.sha256(encoded).hexdigest()
    response = s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=encoded,
        ContentType="application/json",
        Metadata={"sha256": digest},
    )
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise LiveSimExecutionError("S3 bucket versioning must be enabled for live-sim results")
    return {"key": key, "version_id": version_id, "sha256": digest, "size_bytes": len(encoded)}


def _state_keys(strategy_id: str) -> LiveSimStateKeys:
    prefix = f"live-sim/state/{strategy_id}"
    return LiveSimStateKeys(
        transaction=f"{prefix}/state_transaction.json",
        account=f"{prefix}/account_state.json",
        loop=f"{prefix}/loop_state.json",
        trades=f"{prefix}/trades.parquet",
        equity=f"{prefix}/equity_curve.parquet",
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _active_prediction(predictions_path: Path, context: LiveSimContext) -> ActivePrediction:
    predictions = pd.read_parquet(predictions_path)
    required = {"row_id", "model_name", "y_pred", "is_active"}
    missing = required - set(predictions.columns)
    if missing:
        raise LiveSimExecutionError(f"predictions are missing required columns: {sorted(missing)}")

    active_mask = predictions["is_active"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    rows = predictions[active_mask].copy()
    row_ids = pd.to_numeric(rows["row_id"], errors="coerce")
    rows = rows[row_ids == context.row_id]
    if rows.empty:
        raise LiveSimExecutionError(
            f"no active prediction found for live-sim row_id={context.row_id}"
        )

    row = rows.sort_values(["row_id", "model_name"], kind="mergesort").iloc[-1]
    prediction = float(row["y_pred"])
    if not math.isfinite(prediction):
        raise LiveSimExecutionError(f"active prediction is not finite: {prediction!r}")
    active_model_id = row.get("active_model_id")
    if pd.isna(active_model_id):
        active_model_id = row["model_name"]
    model_path = row.get("model_path")
    return ActivePrediction(
        row_id=context.row_id,
        prediction=prediction,
        active_model_id=str(active_model_id),
        model_name=str(row["model_name"]),
        active_model_type=(
            str(row["active_model_type"])
            if "active_model_type" in row.index and pd.notna(row["active_model_type"])
            else None
        ),
        model_path=str(model_path) if pd.notna(model_path) else None,
    )


def _persist_state(
    s3: Any,
    *,
    bucket: str,
    keys: LiveSimStateKeys,
    account_path: Path,
    loop_path: Path,
    trades_path: Path,
    equity_path: Path,
) -> dict[str, dict[str, Any]]:
    refs = {
        "account": _put_file(
            s3, bucket=bucket, key=keys.account, path=account_path, content_type="application/json"
        ),
        "loop": _put_file(
            s3, bucket=bucket, key=keys.loop, path=loop_path, content_type="application/json"
        ),
    }
    if trades_path.exists():
        refs["trades"] = _put_file(
            s3,
            bucket=bucket,
            key=keys.trades,
            path=trades_path,
            content_type="application/vnd.apache.parquet",
        )
    if equity_path.exists():
        refs["equity"] = _put_file(
            s3,
            bucket=bucket,
            key=keys.equity,
            path=equity_path,
            content_type="application/vnd.apache.parquet",
        )
    return refs


def _read_transaction(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveSimExecutionError("live-sim transaction state is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LiveSimExecutionError("live-sim transaction state must be a JSON object")
    account = payload.get("account_state")
    loop = payload.get("loop_state")
    if not isinstance(account, dict) or not isinstance(loop, dict):
        raise LiveSimExecutionError(
            "live-sim transaction state is missing account_state or loop_state"
        )
    return cast(dict[str, Any], payload)


def _write_transaction(
    s3: Any,
    *,
    bucket: str,
    key: str,
    account_path: Path,
    loop_path: Path,
    record: S3CompletionEvent,
    summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        account_state = json.loads(account_path.read_text(encoding="utf-8"))
        loop_state = json.loads(loop_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveSimExecutionError("could not serialize live-sim transaction state") from exc
    if not isinstance(account_state, dict) or not isinstance(loop_state, dict):
        raise LiveSimExecutionError("local live-sim state must serialize as JSON objects")

    return _put_json(
        s3,
        bucket=bucket,
        key=key,
        payload={
            "schema_version": 1,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "account_state": account_state,
            "loop_state": loop_state,
            "last_completion": {
                "key": record.key,
                "version_id": record.version_id,
                "summary": summary,
            },
        },
    )


def _event_summary(
    *,
    status: str,
    signal: str | None,
    reason: str,
    active: ActivePrediction | None,
    trade: dict[str, Any] | None,
    pending_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "signal": signal,
        "reason": reason,
        "selected_model_id": active.active_model_id if active is not None else None,
        "selected_model_name": active.model_name if active is not None else None,
        "prediction": active.prediction if active is not None else None,
        "trade": trade,
        "pending_signal": pending_signal,
    }


def _result_from_transaction(
    *,
    record: S3CompletionEvent,
    completion: LiveInferenceCompletion,
    transaction_ref: dict[str, Any],
    transaction: dict[str, Any],
) -> dict[str, Any] | None:
    last = transaction.get("last_completion")
    if not isinstance(last, dict) or last.get("version_id") != record.version_id:
        return None
    summary = last.get("summary")
    if not isinstance(summary, dict):
        raise LiveSimExecutionError("live-sim transaction has an invalid last_completion summary")
    return {
        "schema_version": 1,
        "status": summary.get("status", "completed"),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "run_id": completion.run_id,
        "strategy_id": completion.context.strategy_id,
        "inference_completion": {"key": record.key, "version_id": record.version_id},
        "inference_request_version_id": completion.request_version_id,
        "bar": completion.context.as_dict(),
        "selected_model_id": summary.get("selected_model_id"),
        "selected_model_name": summary.get("selected_model_name"),
        "prediction": summary.get("prediction"),
        "signal": summary.get("signal"),
        "reason": summary.get("reason"),
        "trade": summary.get("trade"),
        "pending_signal": summary.get("pending_signal"),
        "state": {"transaction": transaction_ref},
    }


def _existing_result(
    s3: Any,
    *,
    bucket: str,
    key: str,
    completion_version_id: str,
) -> dict[str, Any] | None:
    try:
        raw = _get_object_bytes(s3, bucket=bucket, key=key)
    except Exception as exc:
        if _error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveSimExecutionError(
            f"existing live-sim result is invalid: s3://{bucket}/{key}"
        ) from exc
    if not isinstance(payload, dict):
        raise LiveSimExecutionError(
            f"existing live-sim result is not an object: s3://{bucket}/{key}"
        )
    source = payload.get("inference_completion")
    if not isinstance(source, dict) or source.get("version_id") != completion_version_id:
        raise LiveSimExecutionError(
            "live-sim result key collision with another inference completion"
        )
    return cast(dict[str, Any], payload)


def _result_payload(
    *,
    status: str,
    record: S3CompletionEvent,
    completion: LiveInferenceCompletion,
    state_refs: dict[str, dict[str, Any]],
    signal: str | None,
    reason: str,
    active: ActivePrediction | None,
    trade: dict[str, Any] | None,
    pending_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "run_id": completion.run_id,
        "strategy_id": completion.context.strategy_id,
        "inference_completion": {"key": record.key, "version_id": record.version_id},
        "inference_request_version_id": completion.request_version_id,
        "bar": completion.context.as_dict(),
        "selected_model_id": active.active_model_id if active is not None else None,
        "selected_model_name": active.model_name if active is not None else None,
        "prediction": active.prediction if active is not None else None,
        "signal": signal,
        "reason": reason,
        "trade": trade,
        "pending_signal": pending_signal,
        "state": state_refs,
    }


def process_live_sim_record(s3: Any, record: S3CompletionEvent) -> dict[str, Any]:
    """Process one immutable inference completion into serial paper-trading state."""
    completion_bytes = _get_object_bytes(
        s3,
        bucket=record.bucket,
        key=record.key,
        version_id=record.version_id,
    )
    completion = _parse_live_completion(completion_bytes, record=record)
    if completion is None:
        return {"status": "ignored_non_live_sim", "run_id": record.run_id}

    result_key = f"{record.output_prefix}/live_sim_result.json"
    existing = _existing_result(
        s3,
        bucket=record.bucket,
        key=result_key,
        completion_version_id=record.version_id,
    )
    if existing is not None:
        return {"status": "already_completed", "run_id": completion.run_id, "result": existing}

    work_dir = Path(tempfile.mkdtemp(prefix="market-regime-live-sim-", dir="/tmp"))
    try:
        predictions_path = work_dir / "predictions.parquet"
        transaction_path = work_dir / "state" / "state_transaction.json"
        account_path = work_dir / "state" / "account_state.json"
        loop_path = work_dir / "state" / "loop_state.json"
        trades_path = work_dir / "state" / "trades.parquet"
        equity_path = work_dir / "state" / "equity_curve.parquet"
        keys = _state_keys(completion.context.strategy_id)

        _download_pinned(
            s3,
            bucket=record.bucket,
            ref=completion.predictions,
            destination=predictions_path,
        )
        transaction_version = _download_optional_state(
            s3,
            bucket=record.bucket,
            key=keys.transaction,
            destination=transaction_path,
        )
        if transaction_path.exists():
            transaction = _read_transaction(transaction_path)
            transaction_ref = {
                "key": keys.transaction,
                "version_id": transaction_version,
                "sha256": hashlib.sha256(transaction_path.read_bytes()).hexdigest(),
                "size_bytes": transaction_path.stat().st_size,
            }
            recovered = _result_from_transaction(
                record=record,
                completion=completion,
                transaction_ref=transaction_ref,
                transaction=transaction,
            )
            if recovered is not None:
                result_ref = _put_json(s3, bucket=record.bucket, key=result_key, payload=recovered)
                return {
                    "status": "already_applied",
                    "run_id": completion.run_id,
                    "result": result_ref,
                }
            account_path.parent.mkdir(parents=True, exist_ok=True)
            account_path.write_text(
                json.dumps(transaction["account_state"], indent=2, sort_keys=True), encoding="utf-8"
            )
            loop_path.write_text(
                json.dumps(transaction["loop_state"], indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            for key, path in ((keys.account, account_path), (keys.loop, loop_path)):
                _download_optional_state(s3, bucket=record.bucket, key=key, destination=path)

        for key, path in ((keys.trades, trades_path), (keys.equity, equity_path)):
            _download_optional_state(s3, bucket=record.bucket, key=key, destination=path)

        state = load_account_state(account_path)
        loop_state = load_loop_state(loop_path)
        current_bar = _parse_timestamp(completion.context.bar_timestamp_utc)
        last_raw = loop_state.get("last_processed_bar_timestamp_utc")
        if last_raw is not None and current_bar <= _parse_timestamp(str(last_raw)):
            result = _result_payload(
                status="ignored_stale_bar",
                record=record,
                completion=completion,
                state_refs={},
                signal=None,
                reason="bar timestamp is not newer than persisted live-sim state",
                active=None,
                trade=None,
                pending_signal=None,
            )
            result_ref = _put_json(s3, bucket=record.bucket, key=result_key, payload=result)
            return {"status": result["status"], "run_id": completion.run_id, "result": result_ref}

        if last_raw is None:
            loop_state["last_processed_bar_timestamp_utc"] = completion.context.bar_timestamp_utc
            loop_state["initialized_at_utc"] = datetime.now(UTC).isoformat()
            save_account_state(state, account_path)
            save_loop_state(loop_path, loop_state)
            summary = _event_summary(
                status="initialized",
                signal="HOLD",
                reason="initialized from first closed bar without creating a trade",
                active=None,
                trade=None,
                pending_signal=None,
            )
            transaction_ref = _write_transaction(
                s3,
                bucket=record.bucket,
                key=keys.transaction,
                account_path=account_path,
                loop_path=loop_path,
                record=record,
                summary=summary,
            )
            state_refs = {"transaction": transaction_ref}
            state_refs.update(
                _persist_state(
                    s3,
                    bucket=record.bucket,
                    keys=keys,
                    account_path=account_path,
                    loop_path=loop_path,
                    trades_path=trades_path,
                    equity_path=equity_path,
                )
            )
            result = _result_payload(
                status="initialized",
                record=record,
                completion=completion,
                state_refs=state_refs,
                signal="HOLD",
                reason="initialized from first closed bar without creating a trade",
                active=None,
                trade=None,
                pending_signal=None,
            )
            result_ref = _put_json(s3, bucket=record.bucket, key=result_key, payload=result)
            return {"status": result["status"], "run_id": completion.run_id, "result": result_ref}

        trade: dict[str, Any] | None = None
        pending = loop_state.get("pending_signal")
        if isinstance(pending, dict):
            pending_signal_ts = _parse_timestamp(str(pending.get("signal_bar_timestamp_utc")))
            if pending_signal_ts < current_bar:
                executed_trade = execute_signal(
                    state=state,
                    signal=str(pending.get("signal", "HOLD")),
                    price=completion.context.open_price,
                    timestamp=completion.context.bar_timestamp_utc,
                    config=ExecutionConfig(),
                )
                executed_trade.update(
                    {
                        "fill_bar_timestamp": completion.context.bar_timestamp_utc,
                        "signal_bar_timestamp": pending.get("signal_bar_timestamp_utc"),
                        "regime": pending.get("regime"),
                        "active_model_id": pending.get("active_model_id"),
                        "prediction": pending.get("prediction"),
                        "fill_policy": "next_received_bar_open",
                    }
                )
                log_trade(trades_path, executed_trade)
                log_account_snapshot(
                    path=equity_path,
                    state=state,
                    timestamp=completion.context.bar_timestamp_utc,
                    price=completion.context.open_price,
                    regime=str(pending.get("regime")),
                    active_model_id=str(pending.get("active_model_id")),
                    prediction=float(pending.get("prediction", 0.0)),
                    signal=str(pending.get("signal", "HOLD")),
                    action_taken=str(executed_trade["action"]),
                    reason=(
                        "filled pending signal at next received bar open: "
                        f"{executed_trade['reason']}"
                    ),
                )
                trade = executed_trade
                loop_state["last_filled_signal"] = pending
            else:
                raise LiveSimExecutionError(
                    "pending signal timestamp is not earlier than current bar"
                )
        loop_state["pending_signal"] = None

        active: ActivePrediction | None = None
        signal = "HOLD"
        reason: str
        try:
            active = _active_prediction(predictions_path, completion.context)
            if not is_live_eligible_prediction(active):
                reason = f"active model is not live eligible: {active.active_model_id}"
            else:
                signal = prediction_to_signal(
                    active.prediction,
                    regime=completion.context.regime,
                    config=SignalConfig(),
                )
                reason = (
                    "queued signal for next received bar" if signal != "HOLD" else "hold signal"
                )
        except (LiveSimExecutionError, ValueError) as exc:
            reason = f"hold due to active prediction selection failure: {exc}"

        pending_signal: dict[str, Any] | None = None
        if active is not None and signal != "HOLD" and completion.context.queue_signal:
            pending_signal = {
                "signal_bar_timestamp_utc": completion.context.bar_timestamp_utc,
                "signal": signal,
                "prediction": active.prediction,
                "regime": completion.context.regime,
                "active_model_id": active.active_model_id,
                "model_name": active.model_name,
                "row_id": active.row_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            loop_state["pending_signal"] = pending_signal
        elif signal != "HOLD" and not completion.context.queue_signal:
            reason = "signal intentionally not queued by immutable live_sim context"

        state.mark_to_market(completion.context.close_price)
        log_account_snapshot(
            path=equity_path,
            state=state,
            timestamp=completion.context.bar_timestamp_utc,
            price=completion.context.close_price,
            regime=completion.context.regime,
            active_model_id=active.active_model_id if active is not None else "unavailable",
            prediction=active.prediction if active is not None else None,
            signal=signal,
            action_taken="NONE",
            reason=reason,
        )
        loop_state["last_processed_bar_timestamp_utc"] = completion.context.bar_timestamp_utc
        save_account_state(state, account_path)
        save_loop_state(loop_path, loop_state)
        summary = _event_summary(
            status="completed",
            signal=signal,
            reason=reason,
            active=active,
            trade=trade,
            pending_signal=pending_signal,
        )
        transaction_ref = _write_transaction(
            s3,
            bucket=record.bucket,
            key=keys.transaction,
            account_path=account_path,
            loop_path=loop_path,
            record=record,
            summary=summary,
        )
        state_refs = {"transaction": transaction_ref}
        state_refs.update(
            _persist_state(
                s3,
                bucket=record.bucket,
                keys=keys,
                account_path=account_path,
                loop_path=loop_path,
                trades_path=trades_path,
                equity_path=equity_path,
            )
        )
        result = _result_payload(
            status="completed",
            record=record,
            completion=completion,
            state_refs=state_refs,
            signal=signal,
            reason=reason,
            active=active,
            trade=trade,
            pending_signal=pending_signal,
        )
        result_ref = _put_json(s3, bucket=record.bucket, key=result_key, payload=result)
        return {"status": result["status"], "run_id": completion.run_id, "result": result_ref}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint for live-sim inference-completion S3 events."""
    del context
    try:
        import boto3

        s3 = boto3.client("s3")
        results = [process_live_sim_record(s3, record) for record in parse_live_sim_s3_event(event)]
    except (LiveSimContractError, LiveSimExecutionError):
        LOG.exception("Live-sim Lambda request failed")
        raise
    except Exception:
        LOG.exception("Unexpected live-sim Lambda failure")
        raise
    return {"processed": len(results), "results": results}

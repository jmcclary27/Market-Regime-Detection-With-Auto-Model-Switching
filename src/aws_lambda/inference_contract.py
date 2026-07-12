"""Validated S3 contract for the event-driven inference Lambda.

The Lambda never follows a mutable ``latest`` pointer.  An S3 request object
pins the exact versions and SHA-256 digests of its inference input and model
bundle.  Output keys are derived from the request object version, so retries
and later requests cannot silently replace a prior run's canonical result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote_plus

CONTRACT_SCHEMA_VERSION = 1
REQUEST_PREFIX = "inference/requests/"
REQUEST_SUFFIX = "/request.json"
RUNS_PREFIX = "inference/runs/"
MODEL_BUNDLE_PREFIX = "inference/model-bundles/"
LIVE_SIM_RUNS_PREFIX = "inference/live-sim/runs/"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STRATEGY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InferenceContractError(ValueError):
    """Raised when an S3 event or request does not meet the inference contract."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceContractError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceContractError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise InferenceContractError(
            "run_id must be 3-128 characters of letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )
    return run_id


def _validate_key(key: str, *, field: str) -> str:
    if key.startswith("/") or "\\" in key:
        raise InferenceContractError(f"{field} must be a normalized relative S3 key")
    parts = key.split("/")
    if not key or any(part in {"", ".", ".."} for part in parts):
        raise InferenceContractError(f"{field} must be a normalized relative S3 key")
    return key


def run_id_from_request_key(key: str) -> str:
    """Validate and extract a run id from an event-triggering request key."""
    key = _validate_key(key, field="request key")
    if not key.startswith(REQUEST_PREFIX) or not key.endswith(REQUEST_SUFFIX):
        raise InferenceContractError("request key must be inference/requests/<run_id>/request.json")

    parts = key.split("/")
    if len(parts) != 4 or parts[:2] != ["inference", "requests"] or parts[-1] != "request.json":
        raise InferenceContractError("request key must be inference/requests/<run_id>/request.json")
    return _validate_run_id(parts[2])


def request_version_token(version_id: str) -> str:
    """Produce a path-safe deterministic output namespace for an S3 object version."""
    version_id = _string(version_id, field="request object version_id")
    if version_id.lower() == "null":
        raise InferenceContractError("request object must have a non-null S3 version_id")
    return hashlib.sha256(version_id.encode("utf-8")).hexdigest()[:24]


def _utc_timestamp(value: Any, *, field: str) -> str:
    raw = _string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InferenceContractError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise InferenceContractError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise InferenceContractError(f"{field} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceContractError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise InferenceContractError(f"{field} must be a positive finite number")
    return parsed


@dataclass(frozen=True)
class S3ObjectRef:
    """A content-addressed S3 object reference from an inference request."""

    key: str
    version_id: str
    sha256: str

    @classmethod
    def from_payload(cls, payload: Any, *, field: str) -> S3ObjectRef:
        data = _mapping(payload, field=field)
        key = _validate_key(_string(data.get("key"), field=f"{field}.key"), field=f"{field}.key")
        version_id = _string(data.get("version_id"), field=f"{field}.version_id")
        if version_id.lower() == "null":
            raise InferenceContractError(f"{field}.version_id must not be null")
        sha256 = _string(data.get("sha256"), field=f"{field}.sha256").lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise InferenceContractError(f"{field}.sha256 must be a lowercase SHA-256 hex digest")
        return cls(key=key, version_id=version_id, sha256=sha256)

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "version_id": self.version_id, "sha256": self.sha256}


@dataclass(frozen=True)
class LiveSimContext:
    """The deterministic market-bar context consumed by the live-sim Lambda."""

    strategy_id: str
    bar_timestamp_utc: str
    row_id: int
    regime: str
    open_price: float
    close_price: float
    queue_signal: bool

    @classmethod
    def from_payload(cls, payload: Any) -> LiveSimContext:
        data = _mapping(payload, field="live_sim")
        strategy_id = _string(data.get("strategy_id", "default"), field="live_sim.strategy_id")
        if not _STRATEGY_ID_RE.fullmatch(strategy_id):
            raise InferenceContractError(
                "live_sim.strategy_id must contain only letters, digits, '.', '_' or '-'"
            )

        row_id = data.get("row_id")
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
            raise InferenceContractError("live_sim.row_id must be a non-negative integer")

        queue_signal = data.get("queue_signal", True)
        if not isinstance(queue_signal, bool):
            raise InferenceContractError("live_sim.queue_signal must be a boolean")

        return cls(
            strategy_id=strategy_id,
            bar_timestamp_utc=_utc_timestamp(
                data.get("bar_timestamp_utc"), field="live_sim.bar_timestamp_utc"
            ),
            row_id=row_id,
            regime=_string(data.get("regime"), field="live_sim.regime"),
            open_price=_positive_float(data.get("open_price"), field="live_sim.open_price"),
            close_price=_positive_float(data.get("close_price"), field="live_sim.close_price"),
            queue_signal=queue_signal,
        )

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "strategy_id": self.strategy_id,
            "bar_timestamp_utc": self.bar_timestamp_utc,
            "row_id": self.row_id,
            "regime": self.regime,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "queue_signal": self.queue_signal,
        }


@dataclass(frozen=True)
class InferenceRequest:
    """A validated immutable inference request."""

    run_id: str
    mode: str
    inference_ts: int
    target_col: str
    inference_input: S3ObjectRef
    model_bundle: S3ObjectRef
    live_sim: LiveSimContext | None

    @classmethod
    def from_json_bytes(cls, payload: bytes, *, request_key: str) -> InferenceRequest:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceContractError("request object must be UTF-8 JSON") from exc
        return cls.from_payload(decoded, request_key=request_key)

    @classmethod
    def from_payload(cls, payload: Any, *, request_key: str) -> InferenceRequest:
        expected_run_id = run_id_from_request_key(request_key)
        data = _mapping(payload, field="request")

        schema_version = data.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != CONTRACT_SCHEMA_VERSION:
            raise InferenceContractError(f"schema_version must equal {CONTRACT_SCHEMA_VERSION}")

        run_id = _validate_run_id(_string(data.get("run_id"), field="run_id"))
        if run_id != expected_run_id:
            raise InferenceContractError("request run_id must match its S3 request key")

        mode = _string(data.get("mode"), field="mode")
        if mode not in {"batch", "live_sim"}:
            raise InferenceContractError("mode must be either 'batch' or 'live_sim'")

        inference_ts = data.get("inference_ts")
        if isinstance(inference_ts, bool) or not isinstance(inference_ts, int) or inference_ts < 0:
            raise InferenceContractError(
                "inference_ts must be a non-negative integer Unix timestamp"
            )

        target_col = _string(data.get("target_col", "log_return_1_x"), field="target_col")
        if len(target_col) > 128:
            raise InferenceContractError("target_col must be at most 128 characters")

        inputs = _mapping(data.get("inputs"), field="inputs")
        inference_input = S3ObjectRef.from_payload(
            inputs.get("inference_input"), field="inputs.inference_input"
        )
        input_prefix = f"{RUNS_PREFIX}{run_id}/inputs/"
        if not inference_input.key.startswith(input_prefix):
            raise InferenceContractError(
                f"inputs.inference_input.key must be scoped to this run under {input_prefix}"
            )

        model_bundle = S3ObjectRef.from_payload(
            inputs.get("model_bundle"), field="inputs.model_bundle"
        )
        if not model_bundle.key.startswith(MODEL_BUNDLE_PREFIX) or not model_bundle.key.endswith(
            ".tar.gz"
        ):
            raise InferenceContractError(
                "inputs.model_bundle.key must be an inference/model-bundles/*.tar.gz object"
            )

        live_sim: LiveSimContext | None = None
        if mode == "live_sim":
            live_sim = LiveSimContext.from_payload(data.get("live_sim"))
        elif data.get("live_sim") is not None:
            raise InferenceContractError("live_sim context is allowed only when mode is 'live_sim'")

        return cls(
            run_id=run_id,
            mode=mode,
            inference_ts=inference_ts,
            target_col=target_col,
            inference_input=inference_input,
            model_bundle=model_bundle,
            live_sim=live_sim,
        )

    def output_prefix(self, *, request_version_id: str) -> str:
        runs_prefix = LIVE_SIM_RUNS_PREFIX if self.mode == "live_sim" else RUNS_PREFIX
        return f"{runs_prefix}{self.run_id}/outputs/{request_version_token(request_version_id)}"


@dataclass(frozen=True)
class S3EventRecord:
    bucket: str
    key: str
    version_id: str

    @property
    def run_id(self) -> str:
        return run_id_from_request_key(self.key)


def parse_s3_event(event: Any) -> list[S3EventRecord]:
    """Parse only versioned S3 request-object events accepted by this Lambda."""
    event_data = _mapping(event, field="event")
    raw_records = event_data.get("Records")
    if not isinstance(raw_records, list) or not raw_records:
        raise InferenceContractError("event.Records must be a non-empty list")

    records: list[S3EventRecord] = []
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, field=f"event.Records[{index}]")
        s3 = _mapping(record.get("s3"), field=f"event.Records[{index}].s3")
        bucket_data = _mapping(s3.get("bucket"), field=f"event.Records[{index}].s3.bucket")
        object_data = _mapping(s3.get("object"), field=f"event.Records[{index}].s3.object")

        bucket = _string(bucket_data.get("name"), field=f"event.Records[{index}].s3.bucket.name")
        encoded_key = _string(object_data.get("key"), field=f"event.Records[{index}].s3.object.key")
        key = unquote_plus(encoded_key)
        _ = run_id_from_request_key(key)
        version_id = _string(
            object_data.get("versionId"), field=f"event.Records[{index}].s3.object.versionId"
        )
        if version_id.lower() == "null":
            raise InferenceContractError("S3 bucket versioning is required for inference requests")
        records.append(S3EventRecord(bucket=bucket, key=key, version_id=version_id))

    return records

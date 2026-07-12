from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.aws_lambda.inference_contract import request_version_token
from src.aws_lambda.live_sim_handler import S3CompletionEvent, process_live_sim_record

BUCKET = "inference-bucket"
STRATEGY = "paper-default"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _MissingS3Object(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _FakeS3:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], list[tuple[str, bytes]]] = {}
        self._version_counter = 0
        self.fail_once_key: str | None = None

    def add(self, *, bucket: str, key: str, version_id: str, body: bytes) -> None:
        self._objects.setdefault((bucket, key), []).append((version_id, body))

    def get_object(self, *, Bucket: str, Key: str, VersionId: str | None = None) -> dict[str, Any]:
        versions = self._objects.get((Bucket, Key), [])
        if VersionId is not None:
            for version_id, body in reversed(versions):
                if version_id == VersionId:
                    return {"Body": io.BytesIO(body), "VersionId": version_id}
            raise _MissingS3Object()
        if not versions:
            raise _MissingS3Object()
        version_id, body = versions[-1]
        return {"Body": io.BytesIO(body), "VersionId": version_id}

    def download_file(
        self,
        Bucket: str,
        Key: str,
        Filename: str,
        ExtraArgs: dict[str, str],
    ) -> None:
        body = self.get_object(Bucket=Bucket, Key=Key, VersionId=ExtraArgs["VersionId"])[
            "Body"
        ].read()
        Path(Filename).write_bytes(body)

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **_: Any) -> dict[str, str]:
        if self.fail_once_key == Key:
            self.fail_once_key = None
            raise RuntimeError(f"simulated S3 failure for {Key}")
        body = Body.read() if hasattr(Body, "read") else Body
        assert isinstance(body, bytes)
        self._version_counter += 1
        version_id = f"output-{self._version_counter}"
        self.add(bucket=Bucket, key=Key, version_id=version_id, body=body)
        return {"VersionId": version_id}


def _prediction_bytes(*, prediction: float) -> bytes:
    frame = pd.DataFrame(
        {
            "row_id": [7],
            "model_name": ["expert_lightgbm_sideways"],
            "y_pred": [prediction],
            "is_active": [True],
            "active_model_id": ["expert_lightgbm_sideways"],
            "active_model_type": ["expert"],
            "model_path": ["models/experts/sideways/latest.joblib"],
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _completion_event(
    s3: _FakeS3,
    *,
    run_id: str,
    event_version: str,
    bar_timestamp_utc: str,
    open_price: float,
    close_price: float,
    prediction: float,
) -> S3CompletionEvent:
    request_version = f"request-{run_id}"
    token = request_version_token(request_version)
    output_prefix = f"inference/live-sim/runs/{run_id}/outputs/{token}"
    predictions_key = f"{output_prefix}/predictions.parquet"
    prediction_bytes = _prediction_bytes(prediction=prediction)
    s3.add(
        bucket=BUCKET,
        key=predictions_key,
        version_id=f"prediction-{run_id}",
        body=prediction_bytes,
    )
    completion_key = f"{output_prefix}/completed.json"
    completion_payload = {
        "schema_version": 1,
        "status": "completed",
        "mode": "live_sim",
        "run_id": run_id,
        "request": {"version_id": request_version},
        "outputs": {
            "predictions": {
                "key": predictions_key,
                "version_id": f"prediction-{run_id}",
                "sha256": _sha256(prediction_bytes),
            }
        },
        "live_sim": {
            "strategy_id": STRATEGY,
            "bar_timestamp_utc": bar_timestamp_utc,
            "row_id": 7,
            "regime": "bullish",
            "open_price": open_price,
            "close_price": close_price,
            "queue_signal": True,
        },
    }
    s3.add(
        bucket=BUCKET,
        key=completion_key,
        version_id=event_version,
        body=json.dumps(completion_payload).encode("utf-8"),
    )
    return S3CompletionEvent(
        bucket=BUCKET,
        key=completion_key,
        version_id=event_version,
        run_id=run_id,
        output_prefix=output_prefix,
    )


def _result_payload(s3: _FakeS3, result_ref: dict[str, Any]) -> dict[str, Any]:
    raw = s3.get_object(
        Bucket=BUCKET,
        Key=str(result_ref["key"]),
        VersionId=str(result_ref["version_id"]),
    )["Body"].read()
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_live_sim_executor_initializes_queues_fills_and_is_idempotent() -> None:
    s3 = _FakeS3()
    first = _completion_event(
        s3,
        run_id="20260712_153000Z",
        event_version="completion-one",
        bar_timestamp_utc="2026-07-12T15:30:00Z",
        open_price=100.0,
        close_price=100.2,
        prediction=0.01,
    )
    initialized = process_live_sim_record(s3, first)
    assert initialized["status"] == "initialized"

    second = _completion_event(
        s3,
        run_id="20260712_153500Z",
        event_version="completion-two",
        bar_timestamp_utc="2026-07-12T15:35:00Z",
        open_price=100.2,
        close_price=100.5,
        prediction=0.01,
    )
    queued = process_live_sim_record(s3, second)
    assert queued["status"] == "completed"
    queued_payload = _result_payload(s3, queued["result"])
    assert queued_payload["signal"] == "BUY"
    assert queued_payload["pending_signal"]["signal"] == "BUY"

    duplicate = process_live_sim_record(s3, second)
    assert duplicate["status"] == "already_completed"

    third = _completion_event(
        s3,
        run_id="20260712_154000Z",
        event_version="completion-three",
        bar_timestamp_utc="2026-07-12T15:40:00Z",
        open_price=101.0,
        close_price=101.1,
        prediction=0.0,
    )
    filled = process_live_sim_record(s3, third)
    assert filled["status"] == "completed"
    filled_payload = _result_payload(s3, filled["result"])
    assert filled_payload["trade"]["action"] == "BUY"

    trades_key = f"live-sim/state/{STRATEGY}/trades.parquet"
    trades_bytes = s3.get_object(Bucket=BUCKET, Key=trades_key)["Body"].read()
    trades = pd.read_parquet(io.BytesIO(trades_bytes))
    assert list(trades["action"]) == ["BUY"]


def test_live_sim_transaction_prevents_a_retry_from_refilling_a_pending_signal() -> None:
    s3 = _FakeS3()
    first = _completion_event(
        s3,
        run_id="20260712_160000Z",
        event_version="completion-one",
        bar_timestamp_utc="2026-07-12T16:00:00Z",
        open_price=100.0,
        close_price=100.2,
        prediction=0.01,
    )
    process_live_sim_record(s3, first)
    second = _completion_event(
        s3,
        run_id="20260712_160500Z",
        event_version="completion-two",
        bar_timestamp_utc="2026-07-12T16:05:00Z",
        open_price=100.2,
        close_price=100.5,
        prediction=0.01,
    )
    process_live_sim_record(s3, second)
    third = _completion_event(
        s3,
        run_id="20260712_161000Z",
        event_version="completion-three",
        bar_timestamp_utc="2026-07-12T16:10:00Z",
        open_price=101.0,
        close_price=101.1,
        prediction=0.0,
    )

    s3.fail_once_key = f"live-sim/state/{STRATEGY}/account_state.json"
    with pytest.raises(RuntimeError, match="simulated S3 failure"):
        process_live_sim_record(s3, third)

    recovered = process_live_sim_record(s3, third)
    assert recovered["status"] == "already_applied"
    recovered_payload = _result_payload(s3, recovered["result"])
    assert recovered_payload["trade"]["action"] == "BUY"

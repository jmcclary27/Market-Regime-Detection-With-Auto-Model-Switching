from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.aws_lambda.inference_contract import (
    InferenceContractError,
    InferenceRequest,
    S3EventRecord,
    parse_s3_event,
)
from src.aws_lambda.inference_handler import (
    LambdaInferenceError,
    _safe_extract_model_bundle,
    process_record,
)
from src.aws_lambda.model_bundle import ModelBundleError, validate_model_bundle_layout
from tools.package_lambda_model_bundle import build_bundle

RUN_ID = "20260712_153000Z"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "mode": "live_sim",
        "inference_ts": 1783860000,
        "target_col": "target",
        "inputs": {
            "inference_input": {
                "key": f"inference/runs/{RUN_ID}/inputs/regimes.parquet",
                "version_id": "input-version",
                "sha256": _digest(b"input"),
            },
            "model_bundle": {
                "key": "inference/model-bundles/models-20260712.tar.gz",
                "version_id": "bundle-version",
                "sha256": _digest(b"bundle"),
            },
        },
        "live_sim": {
            "strategy_id": "default",
            "bar_timestamp_utc": "2026-07-12T15:30:00Z",
            "row_id": 2,
            "regime": "bullish",
            "open_price": 100.0,
            "close_price": 100.5,
            "queue_signal": True,
        },
    }


def test_request_contract_derives_a_version_scoped_output_prefix() -> None:
    request_key = f"inference/requests/{RUN_ID}/request.json"
    request = InferenceRequest.from_payload(_request_payload(), request_key=request_key)

    output_prefix = request.output_prefix(request_version_id="request-version")

    assert output_prefix.startswith(f"inference/live-sim/runs/{RUN_ID}/outputs/")
    assert output_prefix != request.output_prefix(request_version_id="new-request-version")


def test_request_contract_rejects_cross_run_input_reference() -> None:
    payload = _request_payload()
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    inference_input = inputs["inference_input"]
    assert isinstance(inference_input, dict)
    inference_input["key"] = "inference/runs/other-run/inputs/regimes.parquet"

    with pytest.raises(InferenceContractError, match="scoped to this run"):
        InferenceRequest.from_payload(
            payload, request_key=f"inference/requests/{RUN_ID}/request.json"
        )


def test_s3_event_requires_a_versioned_request_object_and_decodes_key() -> None:
    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "inference-bucket"},
                    "object": {
                        "key": f"inference%2Frequests%2F{RUN_ID}%2Frequest.json",
                        "versionId": "request-version",
                    },
                }
            }
        ]
    }

    records = parse_s3_event(event)

    assert records[0].key == f"inference/requests/{RUN_ID}/request.json"
    assert records[0].run_id == RUN_ID


def test_s3_event_rejects_unversioned_request_object() -> None:
    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "inference-bucket"},
                    "object": {
                        "key": f"inference/requests/{RUN_ID}/request.json",
                        "versionId": "null",
                    },
                }
            }
        ]
    }

    with pytest.raises(InferenceContractError, match="versioning"):
        parse_s3_event(event)


def test_model_bundle_builder_includes_global_active_pointer_and_shadow_roots(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    active_artifact = models / "experts" / "sideways" / "latest.joblib"
    active_metadata = models / "experts" / "sideways" / "latest.json"
    baseline = models / "baseline" / "1" / "model.joblib"
    pretrained = models / "pretrained" / "shadow.joblib"
    for path in (active_artifact, active_metadata, baseline, pretrained):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    active_file = tmp_path / "registry" / "active_model.yaml"
    active_file.parent.mkdir()
    active_file.write_text(
        """active:
  model_type: expert
  regime: sideways
  model_id: expert_lightgbm_sideways
  version: '1'
  artifact_path: models/experts/sideways/latest.joblib
  metadata_path: models/experts/sideways/latest.json
""",
        encoding="utf-8",
    )

    output = tmp_path / "bundle.tar.gz"
    result = build_bundle(models_dir=models, active_file=active_file, output_path=output)

    assert result["sha256"] == _digest(output.read_bytes())
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
    assert "registry/active_model.yaml" in names
    assert "models/experts/sideways/latest.joblib" in names
    assert "models/baseline/1/model.joblib" in names
    assert "models/pretrained/shadow.joblib" in names


def test_model_bundle_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../../outside.txt")
        data = b"unsafe"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with pytest.raises(LambdaInferenceError, match="unsafe archive member"):
        _safe_extract_model_bundle(archive_path, tmp_path / "bundle")


def test_model_bundle_layout_rejects_active_path_outside_models(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    (bundle_root / "models").mkdir(parents=True)
    active_file = bundle_root / "registry" / "active_model.yaml"
    active_file.parent.mkdir()
    active_file.write_text(
        """active:
  model_type: expert
  regime: sideways
  model_id: unsafe
  version: '1'
  artifact_path: ../outside.joblib
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelBundleError, match="normalized project-relative"):
        validate_model_bundle_layout(bundle_root)


class _MissingS3Object(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _FakeS3:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str, str], bytes] = {}
        self._versions = 0

    def add(self, *, bucket: str, key: str, version_id: str, body: bytes) -> None:
        self._objects[(bucket, key, version_id)] = body

    def get_object(self, *, Bucket: str, Key: str, VersionId: str | None = None) -> dict[str, Any]:
        if VersionId is not None:
            body = self._objects.get((Bucket, Key, VersionId))
            if body is None:
                raise _MissingS3Object()
            return {"Body": io.BytesIO(body)}

        candidates = [
            (version, body)
            for (bucket, key, version), body in self._objects.items()
            if bucket == Bucket and key == Key
        ]
        if not candidates:
            raise _MissingS3Object()
        return {"Body": io.BytesIO(candidates[-1][1])}

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
        body = Body.read() if hasattr(Body, "read") else Body
        assert isinstance(body, bytes)
        self._versions += 1
        version_id = f"output-{self._versions}"
        self.add(bucket=Bucket, key=Key, version_id=version_id, body=body)
        return {"VersionId": version_id}


def test_process_record_uses_the_extracted_bundle_for_active_and_shadow_inference(
    tmp_path: Path,
) -> None:
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0]})
    features_path = tmp_path / "regimes.parquet"
    features.to_parquet(features_path, index=False)

    models = tmp_path / "models"
    model_path = models / "experts" / "sideways" / "latest.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(LinearRegression().fit(features, [0.001, 0.002, 0.004]), model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_contract_version": 2,
                "candidate_only": False,
                "promotion_eligible": True,
                "model_type": "lightgbm",
            }
        ),
        encoding="utf-8",
    )

    active_file = tmp_path / "registry" / "active_model.yaml"
    active_file.parent.mkdir()
    active_file.write_text(
        """active:
  model_type: expert
  regime: sideways
  model_id: expert_lightgbm_sideways
  version: '1'
  artifact_path: models/experts/sideways/latest.joblib
  metadata_path: models/experts/sideways/latest.json
""",
        encoding="utf-8",
    )
    bundle_path = tmp_path / "models.tar.gz"
    build_bundle(models_dir=models, active_file=active_file, output_path=bundle_path)

    bucket = "inference-bucket"
    input_key = f"inference/runs/{RUN_ID}/inputs/regimes.parquet"
    bundle_key = "inference/model-bundles/models.tar.gz"
    request_key = f"inference/requests/{RUN_ID}/request.json"
    input_bytes = features_path.read_bytes()
    bundle_bytes = bundle_path.read_bytes()
    request = _request_payload()
    inputs = request["inputs"]
    assert isinstance(inputs, dict)
    for name, body, key, version in (
        ("inference_input", input_bytes, input_key, "input-version"),
        ("model_bundle", bundle_bytes, bundle_key, "bundle-version"),
    ):
        ref = inputs[name]
        assert isinstance(ref, dict)
        ref["key"] = key
        ref["version_id"] = version
        ref["sha256"] = _digest(body)
    request_bytes = json.dumps(request).encode("utf-8")

    s3 = _FakeS3()
    s3.add(bucket=bucket, key=input_key, version_id="input-version", body=input_bytes)
    s3.add(bucket=bucket, key=bundle_key, version_id="bundle-version", body=bundle_bytes)
    s3.add(bucket=bucket, key=request_key, version_id="request-version", body=request_bytes)

    result = process_record(
        s3,
        S3EventRecord(bucket=bucket, key=request_key, version_id="request-version"),
    )

    assert result["status"] == "completed"
    completion = result["completion"]
    assert isinstance(completion, dict)
    completion_body = s3.get_object(
        Bucket=bucket,
        Key=completion["key"],
        VersionId=completion["version_id"],
    )["Body"].read()
    completion_payload = json.loads(completion_body)
    predictions_ref = completion_payload["outputs"]["predictions"]
    prediction_bytes = s3.get_object(
        Bucket=bucket,
        Key=predictions_ref["key"],
        VersionId=predictions_ref["version_id"],
    )["Body"].read()
    prediction_path = tmp_path / "predictions.parquet"
    prediction_path.write_bytes(prediction_bytes)
    predictions = pd.read_parquet(prediction_path)
    active_rows = predictions[predictions["is_active"].astype(bool)]
    assert set(active_rows["model_name"]) == {"expert_lightgbm_sideways"}
    assert len(active_rows) == len(features)

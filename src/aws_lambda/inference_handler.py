"""S3-triggered Lambda handler for active-plus-shadow inference.

The implementation intentionally reuses the project's inference entrypoint;
it only supplies isolated paths under Lambda's ephemeral ``/tmp`` directory.
No AWS credentials are loaded from project files: the Lambda execution role is
the only credential source at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast

from src.experiment.frozen_inference import run_frozen_stage
from src.inference.batch_predict import run_stage

from .inference_contract import (
    InferenceContractError,
    InferenceRequest,
    S3EventRecord,
    S3ObjectRef,
    parse_s3_event,
    request_version_token,
)
from .model_bundle import (
    ModelBundleError,
    is_frozen_experiment_bundle,
    validate_model_bundle_layout,
)

LOG = logging.getLogger(__name__)
_WORKING_DIRECTORY_LOCK = Lock()


class LambdaInferenceError(RuntimeError):
    """Raised for a failed pinned-artifact Lambda inference run."""


@contextmanager
def _bundle_working_directory(bundle_root: Path) -> Any:
    """Resolve the project's relative registry paths inside an extracted bundle.

    The existing lightweight registry deliberately stores project-relative
    artifact paths.  This adapter leaves that core behavior alone and scopes
    the temporary working directory only while calling its public API.
    Lambda concurrency is also reserved at one in Terraform; the lock makes
    the helper safe if a caller invokes it from multiple threads in one process.
    """
    with _WORKING_DIRECTORY_LOCK:
        previous = Path.cwd()
        os.chdir(bundle_root)
        try:
            yield
        finally:
            os.chdir(previous)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_pinned_object(s3: Any, *, bucket: str, ref: S3ObjectRef, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(
        bucket,
        ref.key,
        str(destination),
        ExtraArgs={"VersionId": ref.version_id},
    )
    actual_sha256 = _sha256_file(destination)
    if actual_sha256 != ref.sha256:
        raise LambdaInferenceError(
            f"SHA-256 mismatch for s3://{bucket}/{ref.key} version {ref.version_id}: "
            f"expected {ref.sha256}, got {actual_sha256}"
        )


def _safe_extract_model_bundle(archive_path: Path, destination: Path) -> None:
    """Extract a trusted model bundle while rejecting archive traversal and links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = (destination / member.name).resolve()
                if not member_path.is_relative_to(root):
                    raise LambdaInferenceError(
                        f"Model bundle contains an unsafe archive member: {member.name}"
                    )
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise LambdaInferenceError(
                        f"Model bundle contains an unsupported archive member: {member.name}"
                    )
            archive.extractall(destination, members=members)
    except (tarfile.TarError, OSError) as exc:
        raise LambdaInferenceError(f"Could not extract model bundle: {archive_path}") from exc

    try:
        validate_model_bundle_layout(destination)
    except ModelBundleError as exc:
        raise LambdaInferenceError(f"Unsafe or invalid model bundle: {exc}") from exc


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return str(code) if code is not None else None


def _read_existing_completion(
    s3: Any,
    *,
    bucket: str,
    completion_key: str,
    request_version_id: str,
) -> dict[str, Any] | None:
    try:
        response = s3.get_object(Bucket=bucket, Key=completion_key)
    except Exception as exc:
        if _error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise

    try:
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LambdaInferenceError(
            f"Existing completion marker is not valid JSON: s3://{bucket}/{completion_key}"
        ) from exc

    if not isinstance(payload, dict):
        raise LambdaInferenceError(
            f"Existing completion marker is not a JSON object: s3://{bucket}/{completion_key}"
        )

    request = payload.get("request")
    if not isinstance(request, dict) or request.get("version_id") != request_version_id:
        raise LambdaInferenceError(
            "Completion marker collision: output namespace is associated with a different request "
            f"version: s3://{bucket}/{completion_key}"
        )
    return cast(dict[str, Any], payload)


def _put_file(
    s3: Any,
    *,
    bucket: str,
    key: str,
    path: Path,
    content_type: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    sha256 = _sha256_file(path)
    with path.open("rb") as handle:
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=handle,
            ContentType=content_type,
            Metadata={**metadata, "sha256": sha256},
        )
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise LambdaInferenceError("S3 output bucket versioning must be enabled")
    return {
        "key": key,
        "version_id": version_id,
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
    }


def _put_json(
    s3: Any,
    *,
    bucket: str,
    key: str,
    payload: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    sha256 = hashlib.sha256(encoded).hexdigest()
    response = s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=encoded,
        ContentType="application/json",
        Metadata={**metadata, "sha256": sha256},
    )
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise LambdaInferenceError("S3 output bucket versioning must be enabled")
    return {
        "key": key,
        "version_id": version_id,
        "sha256": sha256,
        "size_bytes": len(encoded),
    }


def _request_payload(s3: Any, record: S3EventRecord) -> tuple[bytes, str]:
    response = s3.get_object(
        Bucket=record.bucket,
        Key=record.key,
        VersionId=record.version_id,
    )
    try:
        payload = response["Body"].read()
    except KeyError as exc:
        raise LambdaInferenceError("S3 request object has no body") from exc
    if not isinstance(payload, bytes):
        raise LambdaInferenceError("S3 request object body must be bytes")
    return payload, hashlib.sha256(payload).hexdigest()


def process_record(s3: Any, record: S3EventRecord) -> dict[str, Any]:
    """Process one versioned request-object event and write immutable outputs."""
    request_bytes, request_sha256 = _request_payload(s3, record)
    request = InferenceRequest.from_json_bytes(request_bytes, request_key=record.key)
    if request.run_id != record.run_id:
        raise LambdaInferenceError("validated request run_id does not match S3 event run_id")

    output_prefix = request.output_prefix(request_version_id=record.version_id)
    completion_key = f"{output_prefix}/completed.json"
    existing = _read_existing_completion(
        s3,
        bucket=record.bucket,
        completion_key=completion_key,
        request_version_id=record.version_id,
    )
    if existing is not None:
        return {
            "status": "already_completed",
            "run_id": request.run_id,
            "mode": request.mode,
            "completion": existing,
        }

    work_dir = Path(tempfile.mkdtemp(prefix="market-regime-inference-", dir="/tmp"))
    try:
        input_path = work_dir / "input" / "inference_input.parquet"
        bundle_path = work_dir / "input" / "model_bundle.tar.gz"
        bundle_root = work_dir / "bundle"
        output_dir = work_dir / "output"
        runs_dir = work_dir / "runs"

        _download_pinned_object(
            s3,
            bucket=record.bucket,
            ref=request.inference_input,
            destination=input_path,
        )
        _download_pinned_object(
            s3,
            bucket=record.bucket,
            ref=request.model_bundle,
            destination=bundle_path,
        )
        _safe_extract_model_bundle(bundle_path, bundle_root)

        recorded_features_path = (
            f"s3://{record.bucket}/{request.inference_input.key}"
            f"?versionId={request.inference_input.version_id}"
        )
        if is_frozen_experiment_bundle(bundle_root):
            prediction_path = run_frozen_stage(
                features_path=input_path,
                bundle_root=bundle_root,
                output_dir=output_dir,
                runs_dir=runs_dir,
                inference_ts=request.inference_ts,
                output_name="predictions.parquet",
                run_meta_name="inference_run.json",
                record_features_path=recorded_features_path,
            )
        else:
            with _bundle_working_directory(bundle_root):
                prediction_path = run_stage(
                    features_path=input_path,
                    active_file=Path("registry/active_model.yaml"),
                    target_col=request.target_col,
                    models_dir=Path("models"),
                    output_dir=output_dir,
                    runs_dir=runs_dir,
                    inference_ts=request.inference_ts,
                    output_name="predictions.parquet",
                    latest_name=None,
                    run_meta_name="inference_run.json",
                    record_features_path=recorded_features_path,
                )
        run_meta_path = runs_dir / "inference_run.json"
        if not run_meta_path.exists():
            raise LambdaInferenceError("Inference core did not produce its run metadata JSON")

        output_metadata = {
            "run-id": request.run_id,
            "request-token": request_version_token(record.version_id),
        }
        predictions = _put_file(
            s3,
            bucket=record.bucket,
            key=f"{output_prefix}/predictions.parquet",
            path=prediction_path,
            content_type="application/vnd.apache.parquet",
            metadata=output_metadata,
        )
        run_metadata = _put_file(
            s3,
            bucket=record.bucket,
            key=f"{output_prefix}/inference_run.json",
            path=run_meta_path,
            content_type="application/json",
            metadata=output_metadata,
        )

        completion_payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "run_id": request.run_id,
            "mode": request.mode,
            "request": {
                "key": record.key,
                "version_id": record.version_id,
                "sha256": request_sha256,
            },
            "inputs": {
                "inference_input": request.inference_input.as_dict(),
                "model_bundle": request.model_bundle.as_dict(),
            },
            "outputs": {
                "predictions": predictions,
                "inference_run": run_metadata,
            },
        }
        if request.live_sim is not None:
            completion_payload["live_sim"] = request.live_sim.as_dict()
        completion = _put_json(
            s3,
            bucket=record.bucket,
            key=completion_key,
            payload=completion_payload,
            metadata=output_metadata,
        )
        return {
            "status": "completed",
            "run_id": request.run_id,
            "mode": request.mode,
            "completion": completion,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint for S3 ``request.json`` object-created events."""
    del context
    try:
        import boto3

        records = parse_s3_event(event)
        s3 = boto3.client("s3")
        results = [process_record(s3, record) for record in records]
    except (InferenceContractError, LambdaInferenceError):
        LOG.exception("Immutable inference request failed")
        raise
    except Exception:
        LOG.exception("Unexpected Lambda inference failure")
        raise

    return {"processed": len(results), "results": results}

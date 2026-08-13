"""Scheduled Alpaca daily producer for the frozen experiment.

It is intentionally the only component that reads live market data. It writes
immutable raw/enriched inputs and uploads the request last, which makes the
existing version-pinned inference Lambda the next stage of the workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.aws_lambda.inference_handler import _safe_extract_model_bundle
from src.experiment.manifest import load_manifest
from src.features.builder import build_features
from src.features.run_features import _to_xy_wide
from src.ingestion.alpaca import fetch_daily_bars
from src.regimes.hmm import label_regimes_hmm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_optional(s3: Any, *, bucket: str, key: str, target: Path) -> bool:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", {})
        if isinstance(response, dict) and response.get("Error", {}).get("Code") in {
            "NoSuchKey",
            "NotFound",
            "404",
        }:
            return False
        raise
    target.write_bytes(response["Body"].read())
    return True


def _put_file(s3: Any, *, bucket: str, key: str, path: Path, content_type: str) -> dict[str, str]:
    with path.open("rb") as handle:
        response = s3.put_object(Bucket=bucket, Key=key, Body=handle, ContentType=content_type)
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise RuntimeError("Experiment bucket must have versioning enabled")
    return {"key": key, "version_id": version_id, "sha256": _sha256(path)}


def _secrets() -> tuple[str, str]:
    import boto3

    secret_arn = os.environ["ALPACA_SECRET_ARN"]
    raw = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)["SecretString"]
    parsed = json.loads(raw)
    return str(parsed["api_key"]), str(parsed["api_secret"])


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del event, context
    import boto3

    s3 = boto3.client("s3")
    bucket = os.environ["EXPERIMENT_BUCKET"]
    manifest_key = os.environ.get("EXPERIMENT_MANIFEST_KEY", "experiment/manifest.json")
    bundle_key = os.environ["EXPERIMENT_MODEL_BUNDLE_KEY"]
    bundle_version = os.environ["EXPERIMENT_MODEL_BUNDLE_VERSION_ID"]
    bundle_sha = os.environ["EXPERIMENT_MODEL_BUNDLE_SHA256"]
    with tempfile.TemporaryDirectory(prefix="daily-experiment-") as temp_dir:
        root = Path(temp_dir)
        manifest_path = root / "manifest.json"
        if not _get_optional(s3, bucket=bucket, key=manifest_key, target=manifest_path):
            raise RuntimeError("Frozen experiment manifest is missing")
        manifest = load_manifest(manifest_path)
        run_id = datetime.now(UTC).date().strftime("%Y%m%d") + "-daily"
        submitted_marker = root / "submitted.json"
        if _get_optional(
            s3,
            bucket=bucket,
            key=f"experiment/{manifest.experiment_id}/runs/{run_id}/submitted.json",
            target=submitted_marker,
        ):
            return {"status": "already_submitted", "run_id": run_id}
        raw_history_path = root / "raw_history.parquet"
        previous = (
            pd.read_parquet(raw_history_path)
            if _get_optional(
                s3,
                bucket=bucket,
                key=f"experiment/{manifest.experiment_id}/market/raw_history.parquet",
                target=raw_history_path,
            )
            else pd.DataFrame()
        )
        key, secret = _secrets()
        today = datetime.now(UTC).date()
        recent = fetch_daily_bars(
            ["SPY", "QQQ"],
            start=today - timedelta(days=14),
            end=today + timedelta(days=1),
            api_key=key,
            api_secret=secret,
        )
        raw = pd.concat([previous, recent], ignore_index=True).drop_duplicates(
            ["timestamp", "symbol"], keep="last"
        )
        raw = raw.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        latest_ts = raw["timestamp"].max()
        final_rows = raw[raw["timestamp"] == latest_ts]
        if set(final_rows["symbol"]) != {"SPY", "QQQ"}:
            return {"status": "skipped_no_final_bar"}
        raw.to_parquet(raw_history_path, index=False)
        run_id = latest_ts.strftime("%Y%m%d") + "-daily"
        raw_ref = _put_file(
            s3,
            bucket=bucket,
            key=f"experiment/{manifest.experiment_id}/market/raw/{run_id}.parquet",
            path=raw_history_path,
            content_type="application/vnd.apache.parquet",
        )
        _ = raw_ref
        _put_file(
            s3,
            bucket=bucket,
            key=f"experiment/{manifest.experiment_id}/market/raw_history.parquet",
            path=raw_history_path,
            content_type="application/vnd.apache.parquet",
        )
        bundle_path = root / "bundle.tar.gz"
        s3.download_file(
            bucket, bundle_key, str(bundle_path), ExtraArgs={"VersionId": bundle_version}
        )
        if _sha256(bundle_path) != bundle_sha:
            raise RuntimeError("Frozen model bundle checksum mismatch")
        bundle_root = root / "bundle"
        _safe_extract_model_bundle(bundle_path, bundle_root)
        features = _to_xy_wide(build_features(raw), symbols=("SPY", "QQQ"))
        regimes = features.join(
            label_regimes_hmm(
                features,
                cfg={
                    "regimes": {"hmm": {"artifacts_dir": str(bundle_root / "models/regimes/hmm")}}
                },
            )
        )
        current = regimes.iloc[-1]
        if str(current["regime"]) == "unknown":
            return {"status": "skipped_unknown_regime"}
        input_path = root / "regimes.parquet"
        regimes.to_parquet(input_path, index=False)
        input_ref = _put_file(
            s3,
            bucket=bucket,
            key=f"inference/runs/{run_id}/inputs/regimes.parquet",
            path=input_path,
            content_type="application/vnd.apache.parquet",
        )
        spy = final_rows[final_rows["symbol"] == "SPY"].iloc[-1]
        request = {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "live_sim",
            "inference_ts": int(datetime.now(UTC).timestamp()),
            "target_col": "log_return_1_x",
            "inputs": {
                "inference_input": input_ref,
                "model_bundle": {
                    "key": bundle_key,
                    "version_id": bundle_version,
                    "sha256": bundle_sha,
                },
            },
            "live_sim": {
                "strategy_id": manifest.experiment_id,
                "bar_timestamp_utc": pd.Timestamp(latest_ts).isoformat(),
                "row_id": int(len(regimes) - 1),
                "regime": str(current["regime"]),
                "open_price": float(spy["open"]),
                "close_price": float(spy["close"]),
                "queue_signal": True,
            },
        }
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        _put_file(
            s3,
            bucket=bucket,
            key=f"inference/requests/{run_id}/request.json",
            path=request_path,
            content_type="application/json",
        )
        submitted_marker.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        _put_file(
            s3,
            bucket=bucket,
            key=f"experiment/{manifest.experiment_id}/runs/{run_id}/submitted.json",
            path=submitted_marker,
            content_type="application/json",
        )
        return {"status": "submitted", "run_id": run_id}

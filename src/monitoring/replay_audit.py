from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.lineage import sha256_file

REPLAY_COMPARABLE_LABELS = (
    "features_parquet",
    "regimes_parquet",
    "predictions_parquet",
)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _artifact_path(lineage: dict[str, Any], project_root: Path, label: str) -> Path | None:
    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    payload = artifacts.get(label)
    if not isinstance(payload, dict):
        return None
    return _resolve_path(project_root, payload.get("path"))


def _artifact_sha(lineage: dict[str, Any], label: str) -> str | None:
    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    payload = artifacts.get(label)
    if not isinstance(payload, dict):
        return None
    value = payload.get("sha256")
    return None if value in (None, "") else str(value)


def _prediction_drift(old_path: Path, new_path: Path) -> dict[str, Any]:
    old_df = pd.read_parquet(old_path)
    new_df = pd.read_parquet(new_path)

    keys = ["row_id", "model_name", "model_source", "is_active"]
    required = keys + ["y_pred"]
    missing_old = [column for column in required if column not in old_df.columns]
    missing_new = [column for column in required if column not in new_df.columns]
    if missing_old or missing_new:
        return {
            "semantic_pass": False,
            "max_prediction_drift": None,
            "failure": {
                "kind": "missing_columns",
                "missing_old": missing_old,
                "missing_new": missing_new,
            },
        }

    old_cmp = old_df[required].sort_values(keys, kind="mergesort").reset_index(drop=True)
    new_cmp = new_df[required].sort_values(keys, kind="mergesort").reset_index(drop=True)

    if not old_cmp[keys].equals(new_cmp[keys]):
        return {
            "semantic_pass": False,
            "max_prediction_drift": None,
            "failure": {"kind": "prediction_keys_differ"},
        }

    drift = pd.to_numeric(old_cmp["y_pred"], errors="coerce") - pd.to_numeric(
        new_cmp["y_pred"], errors="coerce"
    )
    max_drift = float(drift.abs().max()) if not drift.empty else 0.0
    if pd.isna(max_drift):
        return {
            "semantic_pass": False,
            "max_prediction_drift": None,
            "failure": {"kind": "prediction_drift_nan"},
        }

    return {
        "semantic_pass": bool(max_drift <= 1e-12),
        "max_prediction_drift": max_drift,
        "failure": None if max_drift <= 1e-12 else {"kind": "prediction_drift", "value": max_drift},
    }


def default_replay_output(project_root: Path, run_ts: str) -> Path:
    return project_root / "artifacts" / "replay" / f"replay_{run_ts}.json"


def build_replay_audit(
    *,
    project_root: Path,
    run_ts: str,
    lineage: dict[str, Any],
    replay_artifacts: dict[str, Path],
) -> dict[str, Any]:
    checked_artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    exact_pass = True

    raw_path = _artifact_path(lineage, project_root, "raw_csv")
    raw_sha = _artifact_sha(lineage, "raw_csv")
    if raw_path is not None and raw_sha is not None:
        raw_exists = raw_path.exists()
        raw_actual_sha = sha256_file(raw_path) if raw_exists else None
        raw_match = bool(raw_exists and raw_actual_sha == raw_sha)
        checked_artifacts.append(
            {
                "label": "raw_csv",
                "original_path": str(raw_path),
                "replay_path": None,
                "expected_sha256": raw_sha,
                "actual_sha256": raw_actual_sha,
                "exact_match": raw_match,
            }
        )
        if not raw_match:
            exact_pass = False
            failures.append({"kind": "raw_csv_integrity", "path": str(raw_path)})

    for label in REPLAY_COMPARABLE_LABELS:
        original_path = _artifact_path(lineage, project_root, label)
        expected_sha = _artifact_sha(lineage, label)
        replay_path = replay_artifacts.get(label)
        exists = bool(
            original_path is not None
            and original_path.exists()
            and replay_path is not None
            and replay_path.exists()
        )
        actual_sha = (
            sha256_file(replay_path) if replay_path is not None and replay_path.exists() else None
        )
        exact_match = bool(exists and expected_sha is not None and actual_sha == expected_sha)
        checked_artifacts.append(
            {
                "label": label,
                "original_path": str(original_path) if original_path is not None else None,
                "replay_path": str(replay_path) if replay_path is not None else None,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "exact_match": exact_match,
            }
        )
        if not exact_match:
            exact_pass = False
            failures.append({"kind": "artifact_hash_mismatch", "label": label})

    original_predictions = _artifact_path(lineage, project_root, "predictions_parquet")
    replay_predictions = replay_artifacts.get("predictions_parquet")
    semantic_pass = False
    max_prediction_drift: float | None = None
    if (
        original_predictions is not None
        and replay_predictions is not None
        and replay_predictions.exists()
    ):
        drift_summary = _prediction_drift(original_predictions, replay_predictions)
        semantic_pass = bool(drift_summary["semantic_pass"])
        max_prediction_drift = drift_summary["max_prediction_drift"]
        if drift_summary["failure"] is not None:
            failures.append(cast(dict[str, Any], drift_summary["failure"]))
    else:
        failures.append({"kind": "missing_prediction_artifacts"})

    return {
        "run_ts": run_ts,
        "status": "passed" if exact_pass and semantic_pass else "failed",
        "exact_pass": exact_pass,
        "semantic_pass": semantic_pass,
        "max_prediction_drift": max_prediction_drift,
        "checked_artifacts": checked_artifacts,
        "failure_breakdown": failures,
    }


def write_replay_audit(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_replay_audit(path: Path) -> dict[str, Any]:
    return _read_json(path)

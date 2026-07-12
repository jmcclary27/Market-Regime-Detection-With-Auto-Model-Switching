from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_EPSILON = 1e-12
_NON_FEATURE_COLUMNS = frozenset(
    {
        "timestamp",
        "row_id",
        "target",
        "regime",
        "regime_explanation",
    }
)


@dataclass(frozen=True)
class DriftPolicy:
    """Conservative, distribution-only checks for the current regression workflow."""

    window_size: int = 252
    min_samples: int = 30
    feature_mean_shift_z_warning: float = 3.0
    prediction_mean_shift_z_warning: float = 3.0
    prediction_std_ratio_low_warning: float = 0.5
    prediction_std_ratio_high_warning: float = 2.0
    regime_total_variation_warning: float = 0.20


@dataclass(frozen=True)
class InferenceReference:
    run_ts: str
    metadata_path: Path
    features_path: Path
    predictions_path: Path


def _resolve_path(project_root: Path, raw_path: object) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _positive_count(value: object) -> int:
    if not isinstance(value, int | float | str):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _reference_run_ts(path: Path, metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("run_ts", "")).strip()
    if explicit:
        return explicit
    return path.stem.removeprefix("run_")


def find_previous_successful_inference(
    *,
    project_root: Path,
    runs_dir: Path,
    current_predictions_path: Path,
) -> InferenceReference | None:
    """Find the latest persisted, non-replay batch-inference output before this run.

    A batch-inference metadata file is only written after prediction output has been
    materialized. Requiring its output and feature-enriched regime input to exist
    keeps the comparison tied to immutable saved artifacts rather than `latest`.
    """

    if not runs_dir.exists():
        return None

    current_resolved = current_predictions_path.resolve()
    candidates: list[tuple[int, str, InferenceReference]] = []
    for metadata_path in sorted(runs_dir.glob("run_*.json")):
        if metadata_path.name.startswith("run_replay_"):
            continue
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict) or loaded.get("run_type") != "batch_inference":
            continue
        if _positive_count(loaded.get("num_prediction_rows")) == 0:
            continue
        if _positive_count(loaded.get("num_models_succeeded")) == 0:
            continue
        failed_models = loaded.get("failed_models", [])
        if not isinstance(failed_models, list) or failed_models:
            continue

        output_path = _resolve_path(project_root, loaded.get("output_path"))
        features_path = _resolve_path(project_root, loaded.get("features_path"))
        if output_path is None or features_path is None:
            continue
        if not output_path.exists() or not features_path.exists():
            continue
        if output_path.resolve() == current_resolved:
            continue

        try:
            completed_at_ns = metadata_path.stat().st_mtime_ns
        except OSError:
            continue
        reference = InferenceReference(
            run_ts=_reference_run_ts(metadata_path, loaded),
            metadata_path=metadata_path,
            features_path=features_path,
            predictions_path=output_path,
        )
        candidates.append((completed_at_ns, metadata_path.name, reference))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _validate_policy(policy: DriftPolicy) -> None:
    if policy.window_size <= 0:
        raise ValueError("window_size must be positive")
    if policy.min_samples <= 1:
        raise ValueError("min_samples must be at least two")
    if policy.min_samples > policy.window_size:
        raise ValueError("min_samples cannot exceed window_size")
    if policy.feature_mean_shift_z_warning <= 0:
        raise ValueError("feature_mean_shift_z_warning must be positive")
    if policy.prediction_mean_shift_z_warning <= 0:
        raise ValueError("prediction_mean_shift_z_warning must be positive")
    if not 0 < policy.prediction_std_ratio_low_warning < 1:
        raise ValueError("prediction_std_ratio_low_warning must be between zero and one")
    if policy.prediction_std_ratio_high_warning <= 1:
        raise ValueError("prediction_std_ratio_high_warning must be greater than one")
    if not 0 < policy.regime_total_variation_warning <= 1:
        raise ValueError("regime_total_variation_warning must be in (0, 1]")


def _latest_window(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.tail(window_size).copy()

    ordered = df.copy()
    ordered["__drift_timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.sort_values("__drift_timestamp", kind="mergesort", na_position="last")
    return ordered.drop(columns="__drift_timestamp").tail(window_size).copy()


def _finite_values(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    values = np.asarray(pd.to_numeric(frame[column], errors="coerce").to_numpy(), dtype=np.float64)
    return cast(NDArray[np.float64], values[np.isfinite(values)])


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _distribution_summary(
    reference_values: NDArray[np.float64],
    current_values: NDArray[np.float64],
    *,
    policy: DriftPolicy,
) -> dict[str, Any]:
    reference_count = int(reference_values.size)
    current_count = int(current_values.size)
    if reference_count < policy.min_samples or current_count < policy.min_samples:
        return {
            "status": "insufficient_history",
            "reference_n": reference_count,
            "current_n": current_count,
            "reference_mean": None,
            "current_mean": None,
            "reference_std": None,
            "current_std": None,
            "mean_shift_z": None,
            "std_ratio": None,
            "reference_constant": None,
            "constant_reference_changed": None,
        }

    reference_mean = float(np.mean(reference_values))
    current_mean = float(np.mean(current_values))
    reference_std = float(np.std(reference_values))
    current_std = float(np.std(current_values))
    reference_constant = bool(reference_std <= _EPSILON)
    constant_reference_changed = bool(
        reference_constant and abs(current_mean - reference_mean) > _EPSILON
    )
    mean_shift_z = (
        None
        if reference_constant
        else _finite_or_none(abs(current_mean - reference_mean) / reference_std)
    )
    std_ratio = None if reference_constant else _finite_or_none(current_std / reference_std)
    return {
        "status": "ok",
        "reference_n": reference_count,
        "current_n": current_count,
        "reference_mean": _finite_or_none(reference_mean),
        "current_mean": _finite_or_none(current_mean),
        "reference_std": _finite_or_none(reference_std),
        "current_std": _finite_or_none(current_std),
        "mean_shift_z": mean_shift_z,
        "std_ratio": std_ratio,
        "reference_constant": reference_constant,
        "constant_reference_changed": constant_reference_changed,
    }


def _mean_shift_warns(summary: dict[str, Any], threshold: float) -> bool:
    if summary.get("status") != "ok":
        return False
    if bool(summary.get("constant_reference_changed")):
        return True
    value = summary.get("mean_shift_z")
    return isinstance(value, float | int) and float(value) >= threshold


def _prediction_dispersion_warns(summary: dict[str, Any], policy: DriftPolicy) -> bool:
    if summary.get("status") != "ok":
        return False
    if bool(summary.get("reference_constant")):
        current_std = summary.get("current_std")
        return isinstance(current_std, float | int) and float(current_std) > _EPSILON
    ratio = summary.get("std_ratio")
    if not isinstance(ratio, float | int):
        return False
    return bool(
        float(ratio) <= policy.prediction_std_ratio_low_warning
        or float(ratio) >= policy.prediction_std_ratio_high_warning
    )


def _feature_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    policy: DriftPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_window = _latest_window(reference_df, policy.window_size)
    current_window = _latest_window(current_df, policy.window_size)
    columns = sorted(
        column
        for column in set(reference_window.columns).intersection(current_window.columns)
        if column not in _NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(reference_window[column])
        and pd.api.types.is_numeric_dtype(current_window[column])
    )
    if not columns:
        return (
            {
                "status": "unavailable",
                "reason": "no_shared_numeric_features",
                "reference_window_rows": int(len(reference_window)),
                "current_window_rows": int(len(current_window)),
                "features_compared": 0,
                "warning_features": [],
                "features": {},
            },
            [],
        )

    details: dict[str, dict[str, Any]] = {}
    warning_features: list[str] = []
    warnings: list[dict[str, Any]] = []
    compared_count = 0
    for column in columns:
        summary = _distribution_summary(
            _finite_values(reference_window, column),
            _finite_values(current_window, column),
            policy=policy,
        )
        details[column] = summary
        if summary["status"] == "ok":
            compared_count += 1
        if _mean_shift_warns(summary, policy.feature_mean_shift_z_warning):
            warning_features.append(column)
            warnings.append(
                {
                    "kind": "feature_mean_shift",
                    "feature": column,
                    "mean_shift_z": summary["mean_shift_z"],
                    "threshold": policy.feature_mean_shift_z_warning,
                }
            )

    return (
        {
            "status": (
                "warning"
                if warning_features
                else "ok"
                if compared_count
                else "insufficient_history"
            ),
            "reference_window_rows": int(len(reference_window)),
            "current_window_rows": int(len(current_window)),
            "features_compared": compared_count,
            "warning_features": warning_features,
            "features": details,
        },
        warnings,
    )


def _prediction_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    policy: DriftPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {"model_name", "model_source", "y_pred"}
    missing_reference = sorted(required.difference(reference_df.columns))
    missing_current = sorted(required.difference(current_df.columns))
    if missing_reference or missing_current:
        return (
            {
                "status": "unavailable",
                "reason": "missing_prediction_columns",
                "missing_reference_columns": missing_reference,
                "missing_current_columns": missing_current,
                "models": [],
                "warning_models": [],
            },
            [],
        )

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        columns = ["model_source", "model_name", "y_pred"]
        if "row_id" in frame.columns:
            columns.append("row_id")
        elif "timestamp" in frame.columns:
            columns.append("timestamp")
        prepared = frame.loc[:, columns].copy()
        prepared["model_source"] = prepared["model_source"].fillna("<missing>").astype(str)
        prepared["model_name"] = prepared["model_name"].fillna("<missing>").astype(str)
        return prepared

    def _model_window(frame: pd.DataFrame) -> pd.DataFrame:
        if "row_id" in frame.columns:
            ordered = frame.copy()
            ordered["__drift_row_id"] = pd.to_numeric(ordered["row_id"], errors="coerce")
            ordered = ordered.sort_values("__drift_row_id", kind="mergesort", na_position="last")
            return ordered.drop(columns="__drift_row_id").tail(policy.window_size).copy()
        return _latest_window(frame, policy.window_size)

    reference_prepared = _prepare(reference_df)
    current_prepared = _prepare(current_df)
    reference_groups = {
        (str(model_source), str(model_name)): _model_window(group)
        for (model_source, model_name), group in reference_prepared.groupby(
            ["model_source", "model_name"], sort=True
        )
    }
    current_groups = {
        (str(model_source), str(model_name)): _model_window(group)
        for (model_source, model_name), group in current_prepared.groupby(
            ["model_source", "model_name"], sort=True
        )
    }
    common_models = sorted(set(reference_groups).intersection(current_groups))
    if not common_models:
        return (
            {
                "status": "insufficient_history",
                "reason": "no_common_models",
                "models": [],
                "warning_models": [],
                "new_models": [
                    {"model_source": source, "model_name": name}
                    for source, name in sorted(set(current_groups).difference(reference_groups))
                ],
                "missing_models": [
                    {"model_source": source, "model_name": name}
                    for source, name in sorted(set(reference_groups).difference(current_groups))
                ],
            },
            [],
        )

    models: list[dict[str, Any]] = []
    warning_models: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    compared_count = 0
    for model_source, model_name in common_models:
        summary = _distribution_summary(
            _finite_values(reference_groups[(model_source, model_name)], "y_pred"),
            _finite_values(current_groups[(model_source, model_name)], "y_pred"),
            policy=policy,
        )
        if summary["status"] == "ok":
            compared_count += 1
        model = {"model_source": model_source, "model_name": model_name, **summary}
        models.append(model)
        model_warns = False
        if _mean_shift_warns(summary, policy.prediction_mean_shift_z_warning):
            model_warns = True
            warnings.append(
                {
                    "kind": "prediction_mean_shift",
                    "model_source": model_source,
                    "model_name": model_name,
                    "mean_shift_z": summary["mean_shift_z"],
                    "threshold": policy.prediction_mean_shift_z_warning,
                }
            )
        if _prediction_dispersion_warns(summary, policy):
            model_warns = True
            warnings.append(
                {
                    "kind": "prediction_dispersion_shift",
                    "model_source": model_source,
                    "model_name": model_name,
                    "std_ratio": summary["std_ratio"],
                    "low_threshold": policy.prediction_std_ratio_low_warning,
                    "high_threshold": policy.prediction_std_ratio_high_warning,
                }
            )
        if model_warns:
            warning_models.append({"model_source": model_source, "model_name": model_name})

    return (
        {
            "status": (
                "warning" if warning_models else "ok" if compared_count else "insufficient_history"
            ),
            "models": models,
            "warning_models": warning_models,
            "new_models": [
                {"model_source": source, "model_name": name}
                for source, name in sorted(set(current_groups).difference(reference_groups))
            ],
            "missing_models": [
                {"model_source": source, "model_name": name}
                for source, name in sorted(set(reference_groups).difference(current_groups))
            ],
        },
        warnings,
    )


def _regime_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    policy: DriftPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if "regime" not in reference_df.columns or "regime" not in current_df.columns:
        return (
            {
                "status": "unavailable",
                "reason": "missing_regime_column",
                "total_variation_distance": None,
                "reference_distribution": {},
                "current_distribution": {},
            },
            [],
        )

    reference = _latest_window(reference_df, policy.window_size)["regime"].dropna().astype(str)
    current = _latest_window(current_df, policy.window_size)["regime"].dropna().astype(str)
    if len(reference) < policy.min_samples or len(current) < policy.min_samples:
        return (
            {
                "status": "insufficient_history",
                "reference_n": int(len(reference)),
                "current_n": int(len(current)),
                "total_variation_distance": None,
                "reference_distribution": {},
                "current_distribution": {},
            },
            [],
        )

    categories = sorted(set(reference).union(current))
    reference_distribution = {
        category: float((reference == category).mean()) for category in categories
    }
    current_distribution = {
        category: float((current == category).mean()) for category in categories
    }
    total_variation_distance = float(
        0.5
        * sum(
            abs(reference_distribution[category] - current_distribution[category])
            for category in categories
        )
    )
    warning = total_variation_distance >= policy.regime_total_variation_warning
    summary = {
        "status": "warning" if warning else "ok",
        "reference_n": int(len(reference)),
        "current_n": int(len(current)),
        "total_variation_distance": total_variation_distance,
        "threshold": policy.regime_total_variation_warning,
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
    }
    warnings = (
        [
            {
                "kind": "regime_distribution_shift",
                "total_variation_distance": total_variation_distance,
                "threshold": policy.regime_total_variation_warning,
            }
        ]
        if warning
        else []
    )
    return summary, warnings


def _read_frame(path: Path, *, label: str, errors: list[dict[str, str]]) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append({"component": label, "error": type(exc).__name__, "path": str(path)})
        return None


def _unavailable_component(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def build_drift_snapshot(
    *,
    project_root: Path,
    run_ts: str,
    current_features_path: Path,
    current_regimes_path: Path,
    current_predictions_path: Path,
    runs_dir: Path | None = None,
    policy: DriftPolicy | None = None,
) -> dict[str, Any]:
    """Build a deterministic comparison against the prior successful inference run.

    This deliberately observes distributions only. A warning requests review, but
    never blocks inference, promotes a model, or changes the active registry.
    """

    policy = policy or DriftPolicy()
    _validate_policy(policy)
    root = project_root.resolve()
    reference = find_previous_successful_inference(
        project_root=root,
        runs_dir=runs_dir or root / "data" / "runs",
        current_predictions_path=current_predictions_path,
    )
    policy_payload = {
        **asdict(policy),
        "reference_selection": "previous_successful_batch_inference",
        "action_on_warning": "review_only",
    }
    if reference is None:
        return {
            "schema_version": 1,
            "run_ts": run_ts,
            "status": "insufficient_history",
            "action": "none",
            "policy": policy_payload,
            "reference": None,
            "feature_drift": {
                "status": "insufficient_history",
                "reason": "no_prior_successful_run",
            },
            "prediction_drift": {
                "status": "insufficient_history",
                "reason": "no_prior_successful_run",
            },
            "regime_distribution_drift": {
                "status": "insufficient_history",
                "reason": "no_prior_successful_run",
            },
            "warnings": [],
            "errors": [],
        }

    errors: list[dict[str, str]] = []
    reference_features = _read_frame(
        reference.features_path, label="reference_features", errors=errors
    )
    current_features = _read_frame(current_features_path, label="current_features", errors=errors)
    reference_regimes = _read_frame(
        reference.features_path, label="reference_regimes", errors=errors
    )
    current_regimes = _read_frame(current_regimes_path, label="current_regimes", errors=errors)
    reference_predictions = _read_frame(
        reference.predictions_path, label="reference_predictions", errors=errors
    )
    current_predictions = _read_frame(
        current_predictions_path, label="current_predictions", errors=errors
    )

    warnings: list[dict[str, Any]] = []
    if reference_features is not None and current_features is not None:
        feature_drift, feature_warnings = _feature_summary(
            reference_features, current_features, policy=policy
        )
        warnings.extend(feature_warnings)
    else:
        feature_drift = _unavailable_component("feature_artifact_unreadable")

    if reference_predictions is not None and current_predictions is not None:
        prediction_drift, prediction_warnings = _prediction_summary(
            reference_predictions, current_predictions, policy=policy
        )
        warnings.extend(prediction_warnings)
    else:
        prediction_drift = _unavailable_component("prediction_artifact_unreadable")

    if reference_regimes is not None and current_regimes is not None:
        regime_drift, regime_warnings = _regime_summary(
            reference_regimes, current_regimes, policy=policy
        )
        warnings.extend(regime_warnings)
    else:
        regime_drift = _unavailable_component("regime_artifact_unreadable")

    component_statuses = [
        str(feature_drift["status"]),
        str(prediction_drift["status"]),
        str(regime_drift["status"]),
    ]
    if warnings:
        status = "warning"
        action = "review_only"
    elif errors or "unavailable" in component_statuses:
        status = "unavailable"
        action = "none"
    elif all(component_status == "insufficient_history" for component_status in component_statuses):
        status = "insufficient_history"
        action = "none"
    else:
        status = "ok"
        action = "none"

    return {
        "schema_version": 1,
        "run_ts": run_ts,
        "status": status,
        "action": action,
        "policy": policy_payload,
        "reference": {
            "run_ts": reference.run_ts,
            "run_metadata_path": _display_path(root, reference.metadata_path),
            "features_or_regimes_path": _display_path(root, reference.features_path),
            "predictions_path": _display_path(root, reference.predictions_path),
        },
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "regime_distribution_drift": regime_drift,
        "warnings": warnings,
        "errors": errors,
    }


def default_drift_output(project_root: Path, run_ts: str) -> Path:
    return project_root / "artifacts" / "drift" / f"drift_{run_ts}.json"


def default_latest_drift_output(project_root: Path) -> Path:
    return project_root / "artifacts" / "drift" / "latest.json"


def write_drift_snapshot(
    path: Path,
    snapshot: dict[str, Any],
    *,
    latest_path: Path | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    if latest_path is not None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(payload, encoding="utf-8")
    return path

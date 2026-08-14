"""Validation shared by Lambda bundle creation and consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.registry.registry import RegistryError, read_active


class ModelBundleError(ValueError):
    """Raised when a model bundle cannot safely satisfy the registry contract."""


def is_frozen_experiment_bundle(bundle_root: Path) -> bool:
    """Return whether a bundle opts into the isolated frozen-experiment layout."""
    return (bundle_root / "experiment_bundle.json").is_file()


def _frozen_descriptor(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "experiment_bundle.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError("frozen experiment bundle descriptor is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ModelBundleError("frozen experiment bundle descriptor is invalid")
    return payload


def validate_frozen_experiment_bundle_layout(bundle_root: Path) -> None:
    """Validate exact frozen artifacts without accepting a mutable registry pointer."""
    descriptor = _frozen_descriptor(bundle_root)
    static_id = descriptor.get("static_model_id")
    regime_ids = descriptor.get("regime_model_ids")
    if not isinstance(static_id, str) or not static_id or not isinstance(regime_ids, dict):
        raise ModelBundleError("frozen experiment bundle is missing model identities")
    if set(regime_ids) != {"bullish", "sideways", "bearish"}:
        raise ModelBundleError("frozen experiment bundle must contain all three regime model ids")
    roots = [(str(static_id), bundle_root / "models" / "static" / str(static_id))]
    roots.extend(
        (str(model_id), bundle_root / "models" / "experts" / regime / str(model_id))
        for regime, model_id in sorted(regime_ids.items())
    )
    for model_id, root in roots:
        model = root / "model.joblib"
        metadata = root / "metadata.json"
        if not model.is_file() or not metadata.is_file():
            raise ModelBundleError(f"frozen model artifacts are incomplete for {model_id}")
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelBundleError(f"frozen model metadata is unreadable for {model_id}") from exc
        if payload.get("model_id") != model_id or bool(payload.get("candidate_only", True)):
            raise ModelBundleError(f"frozen model metadata is invalid for {model_id}")
    hmm = bundle_root / "models" / "regimes" / "hmm" / "latest"
    if any(
        not (hmm / name).is_file()
        for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json")
    ):
        raise ModelBundleError("frozen experiment bundle has incomplete HMM artifacts")
    if not (bundle_root / "features" / "feature_manifest.json").is_file():
        raise ModelBundleError("frozen experiment bundle is missing its feature manifest")


def _resolve_model_path(*, bundle_root: Path, raw_path: Path, field: str) -> Path:
    if raw_path.is_absolute() or any(part in {"", ".", ".."} for part in raw_path.parts):
        raise ModelBundleError(f"{field} must be a normalized project-relative path")
    if not raw_path.as_posix().startswith("models/"):
        raise ModelBundleError(f"{field} must be scoped under models/")

    root = bundle_root.resolve()
    resolved = (bundle_root / raw_path).resolve()
    if not resolved.is_relative_to(root):
        raise ModelBundleError(f"{field} escapes the extracted model bundle")
    if not resolved.is_file():
        raise ModelBundleError(f"{field} is not present in the model bundle: {raw_path}")
    return resolved


def validate_model_bundle_layout(bundle_root: Path) -> None:
    """Require a safe active pointer and a model root in an extracted bundle."""
    if is_frozen_experiment_bundle(bundle_root):
        validate_frozen_experiment_bundle_layout(bundle_root)
        return
    models_root = bundle_root / "models"
    active_file = bundle_root / "registry" / "active_model.yaml"
    if not models_root.is_dir() or not active_file.is_file():
        raise ModelBundleError("bundle must contain models/ and registry/active_model.yaml")

    try:
        active = read_active(active_file)
    except RegistryError as exc:
        raise ModelBundleError(f"invalid active registry pointer: {exc}") from exc

    _resolve_model_path(
        bundle_root=bundle_root,
        raw_path=active.artifact_path,
        field="active.artifact_path",
    )
    if active.metadata_path is not None:
        _resolve_model_path(
            bundle_root=bundle_root,
            raw_path=active.metadata_path,
            field="active.metadata_path",
        )

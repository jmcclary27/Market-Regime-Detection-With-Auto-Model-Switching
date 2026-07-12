"""Validation shared by Lambda bundle creation and consumption."""

from __future__ import annotations

from pathlib import Path

from src.registry.registry import RegistryError, read_active


class ModelBundleError(ValueError):
    """Raised when a model bundle cannot safely satisfy the registry contract."""


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

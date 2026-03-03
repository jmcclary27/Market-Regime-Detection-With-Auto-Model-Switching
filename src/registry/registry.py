# src/registry/registry.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import json
import yaml

REGISTRY_DIR = Path("registry")
ACTIVE_FILE = REGISTRY_DIR / "active_model.yaml"


class RegistryError(RuntimeError):
    """Raised when the registry pointer is missing/invalid or cannot be loaded."""


@dataclass(frozen=True)
class ActiveModelRef:
    model_type: str  # "baseline" | "expert" | "pretrained"
    model_id: str
    version: str  # keep as str for safety (yaml may parse ints)
    artifact_path: Path
    regime: str | None = None
    metadata_path: Path | None = None
    updated_at: str | None = None


def _now_iso_z() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_active_payload(payload: dict[str, Any]) -> ActiveModelRef:
    if not isinstance(payload, dict):
        raise RegistryError("active_model.yaml root must be a mapping/dict.")

    if "active" not in payload or not isinstance(payload["active"], dict):
        raise RegistryError("active_model.yaml must contain top-level key 'active' as a mapping.")

    active = payload["active"]

    required = ["model_type", "model_id", "version", "artifact_path"]
    missing = [k for k in required if k not in active]
    if missing:
        raise RegistryError(f"active is missing required keys: {missing}")

    model_type = str(active["model_type"]).strip().lower()
    if model_type not in {"baseline", "expert", "pretrained"}:
        raise RegistryError("active.model_type must be one of: baseline, expert, pretrained")

    regime = active.get("regime", None)
    if model_type == "expert":
        if regime is None or str(regime).strip() == "":
            raise RegistryError("active.regime is required when model_type is 'expert'")
        regime = str(regime).strip()
    else:
        # For baseline/pretrained, ignore regime if present
        regime = None

    artifact_path = Path(str(active["artifact_path"]))
    metadata_path = active.get("metadata_path", None)
    metadata_path = Path(str(metadata_path)) if metadata_path not in (None, "") else None

    updated_at = payload.get("updated_at", None)
    if updated_at not in (None, ""):
        updated_at = str(updated_at)

    return ActiveModelRef(
        model_type=model_type,
        model_id=str(active["model_id"]).strip(),
        version=str(active["version"]).strip(),
        artifact_path=artifact_path,
        regime=regime,
        metadata_path=metadata_path,
        updated_at=updated_at,
    )


def read_active(active_file: Path = ACTIVE_FILE) -> ActiveModelRef:
    """
    Read and validate registry/active_model.yaml, returning a strongly-typed ref.
    """
    if not active_file.exists():
        raise RegistryError(f"Active model pointer not found: {active_file}")

    with active_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    return _validate_active_payload(payload)


def write_active(ref: ActiveModelRef, active_file: Path = ACTIVE_FILE) -> None:
    """
    Write registry/active_model.yaml from an ActiveModelRef.
    """
    active_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "active": {
            "model_type": ref.model_type,
            "regime": ref.regime,
            "model_id": ref.model_id,
            "version": ref.version,
            "artifact_path": ref.artifact_path.as_posix(),
        },
        "updated_at": ref.updated_at or _now_iso_z(),
    }

    if ref.metadata_path is not None:
        payload["active"]["metadata_path"] = ref.metadata_path.as_posix()

    # Remove null keys (like regime when not expert)
    if payload["active"].get("regime") is None:
        payload["active"].pop("regime", None)

    with active_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_active_model(
    active_file: Path = ACTIVE_FILE,
) -> tuple[Any, dict[str, Any] | None, ActiveModelRef]:
    ref = read_active(active_file=active_file)

    if not ref.artifact_path.exists():
        raise RegistryError(f"Active model artifact not found: {ref.artifact_path}")

    # NEW: support json artifacts (ARIMA meta)
    if ref.artifact_path.suffix.lower() == ".json":
        data = json.loads(ref.artifact_path.read_text(encoding="utf-8"))
        model = cast(dict[str, Any], data)
    else:
        model = joblib.load(ref.artifact_path)

    metadata: dict[str, Any] | None = None
    if ref.metadata_path is not None:
        if not ref.metadata_path.exists():
            raise RegistryError(f"Active model metadata not found: {ref.metadata_path}")
        metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
        metadata = cast(dict[str, Any], metadata)

    return model, metadata, ref

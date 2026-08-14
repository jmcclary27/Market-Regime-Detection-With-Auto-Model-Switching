# src/registry/registry.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import yaml

REGISTRY_DIR = Path("registry")
ACTIVE_FILE = REGISTRY_DIR / "active_model.yaml"
HISTORY_FILE = REGISTRY_DIR / "history.parquet"
REGISTRY_HISTORY_COLUMNS = [
    "ts",
    "event_type",
    "source",
    "run_ts",
    "reason",
    "previous_model_type",
    "previous_model_id",
    "previous_version",
    "previous_artifact_path",
    "previous_regime",
    "new_model_type",
    "new_model_id",
    "new_version",
    "new_artifact_path",
    "new_regime",
]


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


def _ref_identity(ref: ActiveModelRef | None) -> tuple[str | None, ...]:
    if ref is None:
        return (None, None, None, None, None, None)
    return (
        ref.model_type,
        ref.model_id,
        ref.version,
        ref.artifact_path.as_posix(),
        ref.regime,
        ref.metadata_path.as_posix() if ref.metadata_path is not None else None,
    )


def append_registry_history(history_file: Path, event: dict[str, Any]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    row = {column: event.get(column) for column in REGISTRY_HISTORY_COLUMNS}
    df_new = pd.DataFrame([row], columns=REGISTRY_HISTORY_COLUMNS)

    if history_file.exists():
        df_existing = pd.read_parquet(history_file).reindex(columns=REGISTRY_HISTORY_COLUMNS)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(history_file, index=False)


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


def write_active(
    ref: ActiveModelRef,
    active_file: Path = ACTIVE_FILE,
    *,
    history_file: Path | None = None,
    event_context: dict[str, Any] | None = None,
) -> bool:
    """
    Write registry/active_model.yaml from an ActiveModelRef.
    """
    previous_ref: ActiveModelRef | None = None
    if active_file.exists():
        try:
            previous_ref = read_active(active_file=active_file)
        except RegistryError:
            previous_ref = None

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

    pointer_changed = _ref_identity(previous_ref) != _ref_identity(ref)
    if pointer_changed:
        target_history = history_file or active_file.parent / "history.parquet"
        context = event_context or {}
        append_registry_history(
            target_history,
            {
                "ts": context.get("ts") or _now_iso_z(),
                "event_type": context.get("event_type") or "pointer_update",
                "source": context.get("source") or "registry.write_active",
                "run_ts": context.get("run_ts"),
                "reason": context.get("reason"),
                "previous_model_type": previous_ref.model_type
                if previous_ref is not None
                else None,
                "previous_model_id": previous_ref.model_id if previous_ref is not None else None,
                "previous_version": previous_ref.version if previous_ref is not None else None,
                "previous_artifact_path": (
                    previous_ref.artifact_path.as_posix() if previous_ref is not None else None
                ),
                "previous_regime": previous_ref.regime if previous_ref is not None else None,
                "new_model_type": ref.model_type,
                "new_model_id": ref.model_id,
                "new_version": ref.version,
                "new_artifact_path": ref.artifact_path.as_posix(),
                "new_regime": ref.regime,
            },
        )

    return pointer_changed


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

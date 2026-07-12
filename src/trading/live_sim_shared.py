"""Small dependency-free contracts shared by local and Lambda paper execution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActivePrediction:
    row_id: int
    prediction: float
    active_model_id: str
    model_name: str
    active_model_type: str | None = None
    model_path: str | None = None


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def load_loop_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Loop state must be a JSON object: {path}")
    return data


def save_loop_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    _atomic_write_json(path, state)


def is_live_eligible_prediction(active: ActivePrediction) -> bool:
    """Accept any globally active model that passed batch-inference safety gates."""
    model_id = active.active_model_id.lower().strip()
    return bool(model_id and model_id not in {"unknown", "unavailable"}) and bool(
        np.isfinite(active.prediction)
    )

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the YAML settings configuration.

    Resolution order:
    1. Explicit `path` argument, if provided
    2. SETTINGS_YAML environment variable, if set
    3. settings.yaml located next to this file
    """
    if path is not None:
        config_path = Path(path)
    elif env_path := os.getenv("SETTINGS_YAML"):
        config_path = Path(env_path)
    else:
        config_path = Path(__file__).resolve().parent / "settings.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # yaml.safe_load returns Any; we want a dict at runtime
    return cast(dict[str, Any], data)

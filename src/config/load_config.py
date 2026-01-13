from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path = "src/config/settings.yaml") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # yaml.safe_load returns Any; we want a dict at runtime
    return cast(dict[str, Any], data)

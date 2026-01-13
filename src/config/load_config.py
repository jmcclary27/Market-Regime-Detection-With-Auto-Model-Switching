from pathlib import Path

import yaml


def load_config(path: str | Path = "src/config/settings.yaml") -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

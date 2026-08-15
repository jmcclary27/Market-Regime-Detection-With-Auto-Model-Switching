"""Create an offline, deterministic end-to-end demo run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config.load_config import load_config
from src.features.run_features import run as run_features
from src.inference.batch_predict import run_stage
from src.models.train import Config as TrainConfig
from src.models.train import run as train_baseline
from src.regimes.run_regime_detection import run as run_regimes
from src.registry.registry import ActiveModelRef, write_active

DEMO_TIMESTAMP = "20240131_160000Z"
DEMO_SYMBOLS = ("SPY", "QQQ")
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def build_demo_bars(rows: int = 420) -> pd.DataFrame:
    """Return deterministic, autocorrelated synthetic daily bars for two symbols."""
    if rows < 250:
        raise ValueError("rows must be at least 250 to train the demo baseline")

    dates = pd.date_range("2022-01-03", periods=rows, freq="B", tz="UTC")
    frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(DEMO_SYMBOLS):
        steps = np.arange(rows, dtype=float)
        innovations = 0.0007 * np.sin(steps / (11.0 + index))
        returns = 0.0012 * np.sin(steps / (5.0 + index)) + innovations
        returns[1:] += 0.45 * returns[:-1]
        close = (100.0 + index * 50.0) * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "close": close,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "volume": 1_000_000 + index * 100_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _write_demo_config(path: Path) -> Path:
    """Write an offline configuration that never needs unpublished HMM artifacts."""
    config: dict[str, Any] = load_config(SETTINGS_PATH)
    regimes = dict(config.get("regimes", {}))
    regimes["method"] = "rules"
    config["regimes"] = regimes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run() -> dict[str, Path]:
    """Generate data, publish a local baseline, and run active-model inference."""
    raw_path = Path("data/raw/demo_bars.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    build_demo_bars().to_csv(raw_path, index=False)
    config_path = _write_demo_config(Path("data/demo_settings.yaml"))

    features_path, manifest_path = run_features(
        input_path=raw_path,
        timestamp=DEMO_TIMESTAMP,
        config_path=SETTINGS_PATH,
    )
    regimes_path = run_regimes(
        input_path=features_path,
        timestamp=DEMO_TIMESTAMP,
        config_path=config_path,
    )

    train_baseline(
        TrainConfig(
            features_path=features_path,
            baseline_models_dir=Path("models/baseline"),
            publish_latest=True,
            experiment_name="market-regime-demo",
        )
    )
    active_path = Path("registry/active_model.yaml")
    write_active(
        ActiveModelRef(
            model_type="baseline",
            model_id="baseline_ridge",
            version="demo",
            artifact_path=Path("models/baseline/latest.joblib"),
            metadata_path=Path("models/baseline/latest.json"),
            updated_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        ),
        active_file=active_path,
        event_context={"source": "demo", "run_ts": DEMO_TIMESTAMP, "reason": "demo bootstrap"},
    )
    predictions_path = run_stage(
        features_path=regimes_path,
        inference_ts=20240131160000,
        include_discovered_models=False,
    )

    outputs = {
        "raw": raw_path,
        "features": features_path,
        "manifest": manifest_path,
        "regimes": regimes_path,
        "registry": active_path,
        "predictions": predictions_path,
    }
    print("Offline demo completed:")
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return outputs


def main() -> None:
    run()


if __name__ == "__main__":
    main()

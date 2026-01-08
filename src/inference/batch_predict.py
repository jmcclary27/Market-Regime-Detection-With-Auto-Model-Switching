# src/inference/batch_predict.py
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


@dataclass
class BatchPredictConfig:
    features_path: Path
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/predictions")
    runs_dir: Path = Path("data/runs")


def _latest_timestamp_dir(parent: Path) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"No timestamped dirs found in {parent}")
    return max(candidates, key=lambda p: int(p.name))


def discover_models(models_dir: Path) -> List[Dict]:
    models = []

    # ---- baseline ----
    baseline_dir = models_dir / "baseline"
    if baseline_dir.exists():
        latest_dir = _latest_timestamp_dir(baseline_dir)
        model_path = latest_dir / "model.joblib"
        models.append({
            "model_name": "baseline",
            "model_source": "baseline",
            "model_path": model_path,
        })

    # ---- experts ----
    experts_dir = models_dir / "experts"
    if experts_dir.exists():
        for regime_dir in experts_dir.iterdir():
            latest_model = regime_dir / "latest.joblib"
            if latest_model.exists():
                models.append({
                    "model_name": f"expert_{regime_dir.name}",
                    "model_source": "expert",
                    "model_path": latest_model,
                })

    # ---- pretrained ----
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        for model_path in pretrained_dir.glob("*.joblib"):
            models.append({
                "model_name": model_path.stem,
                "model_source": "pretrained",
                "model_path": model_path,
            })

    if not models:
        raise RuntimeError("No models discovered for inference")

    return models


def run(config: BatchPredictConfig) -> Path:
    ts = int(time.time())

    # Load features
    df = pd.read_parquet(config.features_path)

    # Drop target if it exists
    X = df.drop(columns=["target"], errors="ignore")

    models = discover_models(config.models_dir)

    rows = []
    for spec in models:
        model = joblib.load(spec["model_path"])
        preds = model.predict(X)

        for row_id, y_pred in zip(X.index, preds):
            rows.append({
                "row_id": int(row_id),
                "model_name": spec["model_name"],
                "model_source": spec["model_source"],
                "y_pred": float(y_pred),
                "inference_ts": ts,
                "features_path": str(config.features_path),
                "model_path": str(spec["model_path"]),
            })

    out_df = pd.DataFrame(rows)

    # Write outputs
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"predictions_{ts}.parquet"
    out_df.to_parquet(output_path, index=False)

    latest_path = config.output_dir / "latest.parquet"
    shutil.copyfile(output_path, latest_path)

    # Write run metadata
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "run_type": "batch_inference",
        "timestamp": ts,
        "features_path": str(config.features_path),
        "models": [m["model_name"] for m in models],
        "output_path": str(output_path),
        "num_rows": len(out_df),
    }

    with open(config.runs_dir / f"run_{ts}.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/features/latest.parquet"),
    )
    args = parser.parse_args()

    config = BatchPredictConfig(features_path=args.features_path)
    run(config)


if __name__ == "__main__":
    main()
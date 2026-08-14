"""Inference adapter for an immutable frozen-experiment model bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.aws_lambda.model_bundle import ModelBundleError, validate_frozen_experiment_bundle_layout


def run_frozen_stage(
    *,
    features_path: Path,
    bundle_root: Path,
    output_dir: Path,
    runs_dir: Path,
    inference_ts: int,
    output_name: str,
    run_meta_name: str,
    record_features_path: str,
) -> Path:
    """Run exactly the models named by a validated frozen bundle."""
    try:
        validate_frozen_experiment_bundle_layout(bundle_root)
        descriptor = json.loads(
            (bundle_root / "experiment_bundle.json").read_text(encoding="utf-8")
        )
        frame = pd.read_parquet(features_path)
    except (OSError, ValueError, json.JSONDecodeError, ModelBundleError) as exc:
        raise RuntimeError("frozen experiment inference inputs are invalid") from exc
    if not isinstance(descriptor, dict):
        raise RuntimeError("frozen experiment descriptor is invalid")
    specifications: list[tuple[str, Path, str]] = [
        (
            str(descriptor["static_model_id"]),
            bundle_root / "models" / "static" / str(descriptor["static_model_id"]),
            "baseline",
        )
    ]
    regime_ids = descriptor["regime_model_ids"]
    assert isinstance(regime_ids, dict)
    specifications.extend(
        (
            str(model_id),
            bundle_root / "models" / "experts" / str(regime) / str(model_id),
            "expert",
        )
        for regime, model_id in sorted(regime_ids.items())
    )
    rows: list[dict[str, Any]] = []
    for model_id, root, source in specifications:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        columns = metadata.get("feature_columns")
        if not isinstance(columns, list) or any(not isinstance(column, str) for column in columns):
            raise RuntimeError(f"frozen model {model_id} has invalid feature columns")
        if any(column not in frame.columns for column in columns):
            raise RuntimeError(f"frozen model {model_id} is incompatible with inference input")
        x = frame.loc[:, columns]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise RuntimeError("frozen experiment inference input contains non-finite features")
        bundle = joblib.load(root / "model.joblib")
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        values = np.asarray(model.predict(x), dtype=float)
        if len(values) != len(frame) or not np.isfinite(values).all():
            raise RuntimeError(f"frozen model {model_id} returned invalid predictions")
        rows.extend(
            {
                "row_id": index,
                "model_name": model_id,
                "model_source": source,
                "y_pred": float(value),
                "inference_ts": inference_ts,
                "features_path": record_features_path,
                "model_path": root.relative_to(bundle_root).as_posix() + "/model.joblib",
                "is_active": model_id == str(descriptor["static_model_id"]),
                "active_model_type": "frozen_static",
                "active_model_id": str(descriptor["static_model_id"]),
                "active_model_version": str(metadata.get("version", "")),
                "active_regime": None,
            }
            for index, value in enumerate(values)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name
    pd.DataFrame(rows).sort_values(["row_id", "model_name"], kind="mergesort").to_parquet(
        path, index=False
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / run_meta_name).write_text(
        json.dumps(
            {
                "run_type": "frozen_experiment_inference",
                "models_executed": [item[0] for item in specifications],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path

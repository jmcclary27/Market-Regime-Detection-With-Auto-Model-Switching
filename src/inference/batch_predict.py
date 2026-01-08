# src/inference/batch_predict.py
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


@dataclass
class BatchPredictConfig:
    features_path: Path = Path("data/features/latest.parquet")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/predictions")
    runs_dir: Path = Path("data/runs")
    # if your training uses a different target name, change this
    target_col: str = "target"


def _latest_timestamp_dir(parent: Path) -> Path:
    """Return newest directory with numeric name under parent."""
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"No timestamped dirs found in {parent}")
    return max(candidates, key=lambda p: int(p.name))


def discover_models(models_dir: Path) -> List[Dict[str, str]]:
    """
    Discover models according to your repo's contract:

    - models/baseline/<ts>/model.joblib                  -> model_name="baseline"
    - models/experts/<regime>/latest.joblib              -> model_name=f"expert_<regime>"
    - models/pretrained/*.joblib                         -> model_name=stem
    """
    models: List[Dict[str, str]] = []

    # ---- baseline ----
    baseline_root = models_dir / "baseline"
    if baseline_root.exists():
        latest_dir = _latest_timestamp_dir(baseline_root)
        model_path = latest_dir / "model.joblib"
        if model_path.exists():
            models.append(
                {
                    "model_name": "baseline",
                    "model_source": "baseline",
                    "model_path": str(model_path),
                }
            )

    # ---- experts ----
    experts_root = models_dir / "experts"
    if experts_root.exists():
        for regime_dir in experts_root.iterdir():
            if not regime_dir.is_dir():
                continue
            latest_model = regime_dir / "latest.joblib"
            if latest_model.exists():
                models.append(
                    {
                        "model_name": f"expert_{regime_dir.name}",
                        "model_source": "expert",
                        "model_path": str(latest_model),
                    }
                )

    # ---- pretrained ----
    pretrained_root = models_dir / "pretrained"
    if pretrained_root.exists():
        for model_path in pretrained_root.glob("*.joblib"):
            models.append(
                {
                    "model_name": model_path.stem,
                    "model_source": "pretrained",
                    "model_path": str(model_path),
                }
            )

    if not models:
        raise RuntimeError(
            f"No models discovered. Looked under: {models_dir}/baseline, {models_dir}/experts, {models_dir}/pretrained"
        )

    # deterministic ordering for easier debugging
    models.sort(key=lambda m: (m["model_source"], m["model_name"], m["model_path"]))
    return models


def _unwrap_model(obj: Any) -> Any:
    """
    Your joblib artifacts may be either:
      - a model/pipeline with .predict
      - a dict wrapper like {"model": estimator, "metadata": ...}

    Return the predict()-able estimator/pipeline.
    """
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in ("model", "estimator", "pipeline", "clf", "regressor"):
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

    raise TypeError(
        f"Loaded object of type {type(obj)} does not have .predict and is not a recognized wrapper dict."
    )


def _align_features_if_possible(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    If sklearn model exposes feature_names_in_, align X to those columns.
    If not available, return X unchanged.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return X

    names = list(names)
    missing = [c for c in names if c not in X.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns for model: {missing}")

    return X.loc[:, names]


def _safe_pred_value(v: Any) -> Any:
    """
    Store predictions safely without assuming numeric type.
    - numeric -> float
    - Timestamp/datetime -> ISO string
    - anything else -> string
    """
    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    # Handle numpy scalar types gracefully
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass

    # numeric
    try:
        return float(v)
    except Exception:
        pass

    # datetime-ish that isn't pd.Timestamp
    try:
        ts = pd.Timestamp(v)
        # if conversion succeeded and isn't NaT, store as ISO
        if ts is not pd.NaT:
            return ts.isoformat()
    except Exception:
        pass

    return str(v)


def run(config: BatchPredictConfig) -> Path:
    ts = int(time.time())

    if not config.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {config.features_path}")

    # ---- Load features ----
    df = pd.read_parquet(config.features_path)

    # Build X by dropping target if present
    X = df.drop(columns=[config.target_col], errors="ignore").copy()

    # row_id: stable integer id
    # If index is integer-like, use it; otherwise reset.
    if pd.api.types.is_integer_dtype(X.index):
        row_ids = X.index.astype(int)
    else:
        X = X.reset_index(drop=True)
        row_ids = X.index.astype(int)

    models = discover_models(config.models_dir)

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for spec in models:
        model_path = Path(spec["model_path"])
        try:
            loaded = joblib.load(model_path)
            model = _unwrap_model(loaded)

            X_aligned = _align_features_if_possible(model, X)
            preds = model.predict(X_aligned)

            for rid, pred in zip(row_ids, preds):
                rows.append(
                    {
                        "row_id": int(rid),
                        "model_name": spec["model_name"],
                        "model_source": spec["model_source"],
                        "y_pred": _safe_pred_value(pred),
                        "inference_ts": ts,
                        "features_path": str(config.features_path),
                        "model_path": str(model_path),
                    }
                )

        except Exception as e:
            failed.append(
                {
                    "model_name": spec["model_name"],
                    "model_source": spec["model_source"],
                    "model_path": str(model_path),
                    "error": repr(e),
                }
            )

    if not rows:
        preview = failed[:5]
        raise RuntimeError(
            "Batch inference produced no predictions because all models failed to run. "
            f"First failures: {preview}"
        )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["row_id", "model_name"], kind="mergesort")
        .reset_index(drop=True)
    )

    # ---- Write outputs ----
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"predictions_{ts}.parquet"
    out_df.to_parquet(output_path, index=False)

    latest_path = config.output_dir / "latest.parquet"
    shutil.copyfile(output_path, latest_path)

    # ---- Write run metadata ----
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    run_meta: Dict[str, Any] = {
        "run_type": "batch_inference",
        "timestamp": ts,
        "features_path": str(config.features_path),
        "models_discovered": models,
        "output_path": str(output_path),
        "latest_path": str(latest_path),
        "num_prediction_rows": int(len(out_df)),
        "num_models_succeeded": int(out_df["model_name"].nunique()),
        "failed_models": failed,
    }

    with open(config.runs_dir / f"run_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference across all models (shadow predictions)."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/features/latest.parquet"),
        help="Path to features parquet. Default: data/features/latest.parquet",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column name to drop from features before inference (if present). Default: target",
    )
    args = parser.parse_args()

    config = BatchPredictConfig(
        features_path=args.features_path,
        target_col=args.target_col,
    )
    out_path = run(config)
    print(f"Wrote predictions: {out_path}")
    print(f"Updated latest: {config.output_dir / 'latest.parquet'}")


if __name__ == "__main__":
    main()
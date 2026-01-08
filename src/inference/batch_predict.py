# src/inference/batch_predict.py
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


@dataclass
class BatchPredictConfig:
    features_path: Path = Path("data/features/latest.parquet")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/predictions")
    runs_dir: Path = Path("data/runs")
    target_col: str = "target"


def _latest_timestamp_dir(parent: Path) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"No timestamped dirs found in {parent}")
    return max(candidates, key=lambda p: int(p.name))


def discover_models(models_dir: Path) -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []

    # baseline: models/baseline/<ts>/model.joblib
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

    # experts: models/experts/<regime>/latest.joblib
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

    # pretrained: models/pretrained/*.joblib
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

    # deterministic order
    models.sort(key=lambda m: (m["model_source"], m["model_name"], m["model_path"]))
    return models


def _unwrap_model(obj: Any) -> Any:
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in ("model", "estimator", "pipeline", "clf", "regressor"):
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

    raise TypeError(
        f"Loaded object of type {type(obj)} does not have .predict and is not a recognized wrapper dict."
    )


def _align_X_for_model(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align X to what the model expects.

    Priority:
    1) If sklearn model exposes feature_names_in_, use those columns in that order.
    2) Otherwise, fall back to expected feature count. If X has extra cols, take first n.
    """
    # 1) Feature-name alignment (best)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise RuntimeError(f"Missing required feature columns for model: {missing}")
        return X.loc[:, names]

    # 2) Count-based fallback (works when model was trained on numpy arrays)
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is None:
        # Some pipelines may not expose this; just return X and let sklearn error if needed
        return X

    n_expected = int(n_expected)
    if X.shape[1] < n_expected:
        raise RuntimeError(
            f"X has {X.shape[1]} features but model expects {n_expected}. "
            "Not enough features to run inference."
        )

    if X.shape[1] > n_expected:
        # Deterministic: take first n columns in current order
        return X.iloc[:, :n_expected]

    return X


def _make_numeric_X(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convert datetime columns to int64 nanoseconds.
    Drop any remaining non-numeric columns (object/string).
    Return: (X_numeric, converted_cols, dropped_cols)
    """
    X = df.copy()
    converted: List[str] = []
    dropped: List[str] = []

    # Convert datetime64 columns to int64
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64")
            converted.append(col)

    # Drop anything still non-numeric (object, string, mixed)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    if non_numeric:
        X = X.drop(columns=non_numeric)
        dropped.extend(non_numeric)

    # Convert bool -> int (safe for sklearn)
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    for c in bool_cols:
        X[c] = X[c].astype(int)

    return X, converted, dropped


def _safe_pred_value(v: Any) -> Any:
    # Keep datetimes as ISO strings if they ever appear
    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    # numpy scalars -> python scalars
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass

    # normal numeric predictions
    try:
        return float(v)
    except Exception:
        pass

    # fallback
    return str(v)


def run(config: BatchPredictConfig) -> Path:
    ts = int(time.time())

    if not config.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {config.features_path}")

    df = pd.read_parquet(config.features_path)

    # Build X by dropping target if present
    X_raw = df.drop(columns=[config.target_col], errors="ignore").copy()

    # If index is not integer-like, reset so row_id is stable ints
    if not pd.api.types.is_integer_dtype(X_raw.index):
        X_raw = X_raw.reset_index(drop=True)

    row_ids = X_raw.index.astype(int)

    # Make X numeric (fixes your Timestamp -> float crash)
    X, converted_cols, dropped_cols = _make_numeric_X(X_raw)

    if X.shape[1] == 0:
        raise RuntimeError(
            "After preprocessing, X has 0 usable numeric feature columns. "
            "Your features parquet may be mostly timestamps/strings. "
            f"Dropped columns: {dropped_cols}"
        )

    models = discover_models(config.models_dir)

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for spec in models:
        model_path = Path(spec["model_path"])
        try:
            loaded = joblib.load(model_path)
            model = _unwrap_model(loaded)

            X_aligned = _align_X_for_model(model, X)
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"predictions_{ts}.parquet"
    out_df.to_parquet(output_path, index=False)

    latest_path = config.output_dir / "latest.parquet"
    shutil.copyfile(output_path, latest_path)

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
        "feature_preprocessing": {
            "converted_datetime_cols": converted_cols,
            "dropped_non_numeric_cols": dropped_cols,
            "final_num_features": int(X.shape[1]),
        },
    }

    with open(config.runs_dir / f"run_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference across all models (shadow predictions).")
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
        help="Target column to drop if present. Default: target",
    )
    args = parser.parse_args()

    config = BatchPredictConfig(features_path=args.features_path, target_col=args.target_col)
    out_path = run(config)
    print(f"Wrote predictions: {out_path}")
    print(f"Updated latest: {config.output_dir / 'latest.parquet'}")


if __name__ == "__main__":
    main()
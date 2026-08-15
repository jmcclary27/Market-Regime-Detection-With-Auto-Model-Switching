# src/models/train.py
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

LOG = logging.getLogger("train_baseline_features_only")


@dataclass(frozen=True)
class Config:
    features_path: Path = Path("data/features/latest.parquet")

    timestamp_col: str = "timestamp"
    return_col: str = "log_return_1"
    target_col: str = "target_next_return"

    train_frac: float = 0.7
    val_frac: float = 0.15

    ridge_alpha: float = 1.0
    imputer_strategy: str = "median"
    random_state: int = 42

    # Candidate artifacts live outside the directories scanned by inference until an
    # explicit review/publish step opts into replacing the latest pointer.
    baseline_models_dir: Path = Path("models/candidates/baseline")
    min_rows: int = 200
    min_train_rows: int = 100
    min_test_rows: int = 25
    publish_latest: bool = False

    tracking_uri: str | None = None
    experiment_name: str = "market-regime-auto-switch"


# Backward-compatible exports for older tests / callers
TrainConfig = Config


class SplitMetrics(TypedDict):
    mae: float
    rmse: float


def _normalize_mlflow_uri(uri: str) -> str:
    u = uri.strip()
    if "://" in u or u.startswith("file:"):
        return u
    return Path(u).resolve().as_uri()


def _resolve_col_df(df: pd.DataFrame, base: str) -> str:
    """
    Resolve base / base_x / base_y in a dataframe.

    Priority:
      1) base
      2) base_x
      3) base_y

    This keeps training compatible with both long (single symbol) and wide (x/y) schemas.
    """
    if base in df.columns:
        return base
    if f"{base}_x" in df.columns:
        return f"{base}_x"
    if f"{base}_y" in df.columns:
        return f"{base}_y"
    raise KeyError(f"Missing required column '{base}' (or suffixed). cols={list(df.columns)}")


def time_split(
    df: pd.DataFrame, ts_col: str, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise ValueError(f"Not enough rows to train, n={n}")

    if n < 10:
        n_train = min(max(2, int(round(n * train_frac))), n - 1)
        train = df.iloc[:n_train].copy()
        val = df.iloc[0:0].copy()
        test = df.iloc[n_train:].copy()
        LOG.warning(
            "Small dataset (n=%d), using train/test only: n_train=%d, n_test=%d",
            n,
            len(train),
            len(test),
        )
        return train, val, test

    n_train = max(2, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_train = max(2, n_train - 1)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    if len(test) == 0:
        raise ValueError("Split produced empty test set.")
    return train, val, test


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _finite_nunique(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _target_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        raise ValueError("Training target has no finite values.")
    quantiles = finite.quantile([0.01, 0.5, 0.99])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "p01": float(quantiles.loc[0.01]),
        "p50": float(quantiles.loc[0.5]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(finite.max()),
    }


def _zero_return_quality_gate(
    val_metrics: SplitMetrics | None,
    y_val: np.ndarray,
    test_metrics: SplitMetrics,
    y_test: np.ndarray,
) -> dict[str, Any]:
    zero_return_val_rmse = float(np.sqrt(np.mean(np.square(y_val))))
    zero_return_test_rmse = float(np.sqrt(np.mean(np.square(y_test))))
    val_rmse = float(val_metrics["rmse"]) if val_metrics is not None else float("nan")
    test_rmse = float(test_metrics["rmse"])
    reasons: list[str] = []
    if not np.isfinite(val_rmse) or val_rmse > zero_return_val_rmse:
        reasons.append("validation_rmse_exceeds_zero_return_baseline")
    if test_rmse > zero_return_test_rmse:
        reasons.append("test_rmse_exceeds_zero_return_baseline")
    return {
        "promotion_eligible": not reasons,
        "val_rmse": val_rmse,
        "zero_return_val_rmse": zero_return_val_rmse,
        "test_rmse": test_rmse,
        "zero_return_test_rmse": zero_return_test_rmse,
        "reason": "; ".join(reasons) if reasons else None,
        "reasons": reasons,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temp_path)
    temp_path.replace(path)


def evaluate(
    model: Any, df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> SplitMetrics:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": rmse(y, pred)}


def run(cfg: TrainConfig) -> Path:
    """
    Train a validated baseline candidate.

    New artifacts are versioned and candidate-only by default. Updating
    ``latest.joblib`` remains an explicit publish action so a retraining run
    cannot replace the production baseline accidentally.
    """
    if cfg.min_rows < 3:
        raise ValueError("min_rows must be >= 3")
    if cfg.min_train_rows < 2:
        raise ValueError("min_train_rows must be >= 2")
    if cfg.min_test_rows < 1:
        raise ValueError("min_test_rows must be >= 1")

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(_normalize_mlflow_uri(cfg.tracking_uri))
    # The project deliberately uses local file-backed MLflow tracking for offline runs.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_experiment(cfg.experiment_name)

    feats = pd.read_parquet(cfg.features_path)

    if cfg.timestamp_col not in feats.columns:
        raise KeyError(
            f"Missing required column '{cfg.timestamp_col}' in features. cols={list(feats.columns)}"
        )

    resolved_return_col = _resolve_col_df(feats, cfg.return_col)

    feats = feats.sort_values(cfg.timestamp_col).reset_index(drop=True)
    feats[cfg.target_col] = feats[resolved_return_col].shift(-1)
    target_numeric = pd.to_numeric(feats[cfg.target_col], errors="coerce")
    feats = feats.loc[np.isfinite(target_numeric)].reset_index(drop=True)
    if len(feats) < cfg.min_rows:
        raise ValueError(
            "Refusing to train a baseline candidate from an undersized dataset: "
            f"n_rows={len(feats)} < min_rows={cfg.min_rows}. No artifact was written."
        )

    # Features-only numeric columns
    # The observed return is the training target's source and is deliberately
    # removed at inference time by the shared batch-prediction contract.
    exclude = {cfg.timestamp_col, "symbol", cfg.target_col, resolved_return_col}
    feature_cols = [
        c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found. cols={list(feats.columns)}")

    train_df, val_df, test_df = time_split(feats, cfg.timestamp_col, cfg.train_frac, cfg.val_frac)
    if len(train_df) < cfg.min_train_rows:
        raise ValueError(
            f"Baseline training split has {len(train_df)} rows, below "
            f"min_train_rows={cfg.min_train_rows}. No artifact was written."
        )
    if len(test_df) < cfg.min_test_rows:
        raise ValueError(
            f"Baseline test split has {len(test_df)} rows, below "
            f"min_test_rows={cfg.min_test_rows}. No artifact was written."
        )

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[cfg.target_col].to_numpy()
    nan_frac = float(np.isnan(X_train).mean()) if X_train.size else 0.0

    model: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.imputer_strategy)),
            ("ridge", Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state)),
        ]
    )
    model.fit(X_train, y_train)

    val_metrics: SplitMetrics | None = (
        None if len(val_df) == 0 else evaluate(model, val_df, feature_cols, cfg.target_col)
    )
    test_metrics: SplitMetrics = evaluate(model, test_df, feature_cols, cfg.target_col)
    val_pred_nunique: int | None = None
    if len(val_df) > 0:
        val_pred_nunique = _finite_nunique(model.predict(val_df[feature_cols].to_numpy()))
        if val_pred_nunique <= 1:
            raise ValueError(
                "Refusing to save baseline candidate because validation predictions collapsed "
                f"to a constant (n_unique={val_pred_nunique})."
            )
    test_pred_nunique = _finite_nunique(model.predict(test_df[feature_cols].to_numpy()))
    if test_pred_nunique <= 1:
        raise ValueError(
            "Refusing to save baseline candidate because test predictions collapsed "
            f"to a constant (n_unique={test_pred_nunique})."
        )

    quality_gate = _zero_return_quality_gate(
        val_metrics,
        val_df[cfg.target_col].to_numpy(dtype=float),
        test_metrics,
        test_df[cfg.target_col].to_numpy(dtype=float),
    )
    promotion_eligible = bool(quality_gate["promotion_eligible"])

    metrics: dict[str, Any] = {
        "val": val_metrics,
        "test": test_metrics,
        "n_rows": int(len(feats)),
        "n_features": int(len(feature_cols)),
        "X_train_nan_frac": float(nan_frac),
        "feature_cols": feature_cols,
        "return_col": cfg.return_col,
        "resolved_return_col": resolved_return_col,
        "target_summary": _target_summary(feats[cfg.target_col]),
        "val_pred_nunique": val_pred_nunique,
        "test_pred_nunique": test_pred_nunique,
        "quality_gate": quality_gate,
    }

    # Keep a numeric version directory for the existing inference discovery contract.
    run_ts = time.time_ns()
    out_dir = cfg.baseline_models_dir / str(run_ts)
    out_dir.mkdir(parents=True, exist_ok=False)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"

    artifact: dict[str, Any] = {
        "artifact_contract_version": 2,
        "candidate_only": not (cfg.publish_latest and promotion_eligible),
        "promotion_eligible": promotion_eligible,
        "quality_gate": quality_gate,
        "model": model,
        "feature_cols": feature_cols,
        "target_col": cfg.target_col,
        "timestamp_col": cfg.timestamp_col,
        "return_col": cfg.return_col,
        "resolved_return_col": resolved_return_col,
        "runtime_versions": _runtime_versions(),
    }
    metadata: dict[str, Any] = {
        "artifact_contract_version": 2,
        "model_type": "ridge",
        "model_name": "baseline_ridge",
        "candidate_only": not (cfg.publish_latest and promotion_eligible),
        "publish_requested": cfg.publish_latest,
        "promotion_eligible": promotion_eligible,
        "created_at_unix_ns": run_ts,
        "features_path": str(cfg.features_path),
        "feature_cols": feature_cols,
        "target_col": cfg.target_col,
        "return_col": cfg.return_col,
        "resolved_return_col": resolved_return_col,
        "metrics": metrics,
        "params": {
            "ridge_alpha": cfg.ridge_alpha,
            "imputer_strategy": cfg.imputer_strategy,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
        },
        "runtime_versions": _runtime_versions(),
    }
    _atomic_joblib_dump(artifact, model_path)
    _atomic_write_text(meta_path, json.dumps(metadata, indent=2, sort_keys=True))

    if cfg.publish_latest and promotion_eligible:
        _atomic_joblib_dump(artifact, cfg.baseline_models_dir / "latest.joblib")
        _atomic_write_text(
            cfg.baseline_models_dir / "latest.json", json.dumps(metadata, indent=2, sort_keys=True)
        )

    # Mypy-friendly locals for nested indexing
    test_m = cast(SplitMetrics, metrics["test"])
    val_m = cast(SplitMetrics | None, metrics["val"])

    with mlflow.start_run(run_name=f"baseline_features_only_{run_ts}") as _r:
        mlflow.log_param("features_path", str(cfg.features_path))
        mlflow.log_param("ridge_alpha", float(cfg.ridge_alpha))
        mlflow.log_param("n_features", int(len(feature_cols)))
        mlflow.log_param("return_col", str(cfg.return_col))
        mlflow.log_param("resolved_return_col", str(resolved_return_col))
        mlflow.log_param(
            "candidate_only", str(not (cfg.publish_latest and promotion_eligible)).lower()
        )
        mlflow.log_param("promotion_eligible", str(promotion_eligible).lower())
        mlflow.log_metric("X_train_nan_frac", float(nan_frac))
        mlflow.log_metric("test_mae", float(test_m["mae"]))
        mlflow.log_metric("test_rmse", float(test_m["rmse"]))
        mlflow.log_metric("test_pred_nunique", float(test_pred_nunique))
        mlflow.log_metric("zero_return_test_rmse", float(quality_gate["zero_return_test_rmse"]))
        if val_m is not None:
            mlflow.log_metric("val_mae", float(val_m["mae"]))
            mlflow.log_metric("val_rmse", float(val_m["rmse"]))
            mlflow.log_metric("zero_return_val_rmse", float(quality_gate["zero_return_val_rmse"]))
        if val_pred_nunique is not None:
            mlflow.log_metric("val_pred_nunique", float(val_pred_nunique))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

    print("Wrote validated baseline candidate:", model_path)
    if cfg.publish_latest and promotion_eligible:
        print("Published baseline latest pointer under:", cfg.baseline_models_dir)
    if cfg.publish_latest and not promotion_eligible:
        raise ValueError(
            "Baseline candidate failed the zero-return test gate; latest pointers were not updated. "
            f"model_rmse={quality_gate['test_rmse']:.8f} "
            f"zero_return_rmse={quality_gate['zero_return_test_rmse']:.8f}"
        )
    print("Resolved return col:", resolved_return_col)
    print("Feature cols:", feature_cols)
    return model_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s, %(message)s"
    )

    p = argparse.ArgumentParser()
    p.add_argument("--features-path", default=str(Config.features_path))
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment-name", default=Config.experiment_name)
    p.add_argument("--ridge-alpha", type=float, default=Config.ridge_alpha)
    p.add_argument("--output-dir", default=str(Config.baseline_models_dir))
    p.add_argument("--min-rows", type=int, default=Config.min_rows)
    p.add_argument("--min-train-rows", type=int, default=Config.min_train_rows)
    p.add_argument("--min-test-rows", type=int, default=Config.min_test_rows)
    p.add_argument(
        "--publish-latest",
        action="store_true",
        help="Explicitly update latest.joblib/latest.json after candidate validation succeeds.",
    )
    args = p.parse_args()

    cfg = Config(
        features_path=Path(args.features_path),
        tracking_uri=args.tracking_uri,
        experiment_name=str(args.experiment_name),
        ridge_alpha=float(args.ridge_alpha),
        baseline_models_dir=Path(args.output_dir),
        min_rows=int(args.min_rows),
        min_train_rows=int(args.min_train_rows),
        min_test_rows=int(args.min_test_rows),
        publish_latest=bool(args.publish_latest),
    )

    run(cfg)


if __name__ == "__main__":
    main()

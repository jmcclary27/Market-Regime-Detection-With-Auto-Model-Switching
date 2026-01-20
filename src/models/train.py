from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
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

    baseline_models_dir: Path = Path("models/baseline")

    tracking_uri: str | None = None
    experiment_name: str = "market-regime-auto-switch"


def _normalize_mlflow_uri(uri: str) -> str:
    u = uri.strip()
    if "://" in u or u.startswith("file:"):
        return u
    return Path(u).resolve().as_uri()


def time_split(df: pd.DataFrame, ts_col: str, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise ValueError(f"Not enough rows to train, n={n}")

    if n < 10:
        n_train = min(max(2, int(round(n * train_frac))), n - 1)
        train = df.iloc[:n_train].copy()
        val = df.iloc[0:0].copy()
        test = df.iloc[n_train:].copy()
        LOG.warning("Small dataset (n=%d), using train/test only: n_train=%d, n_test=%d", n, len(train), len(test))
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


def evaluate(model: Any, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict[str, float]:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": rmse(y, pred)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s, %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--features-path", default=str(Config.features_path))
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment-name", default=Config.experiment_name)
    p.add_argument("--ridge-alpha", type=float, default=Config.ridge_alpha)
    args = p.parse_args()

    cfg = Config(
        features_path=Path(args.features_path),
        tracking_uri=args.tracking_uri,
        experiment_name=str(args.experiment_name),
        ridge_alpha=float(args.ridge_alpha),
    )

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(_normalize_mlflow_uri(cfg.tracking_uri))
    mlflow.set_experiment(cfg.experiment_name)

    feats = pd.read_parquet(cfg.features_path)
    for col in (cfg.timestamp_col, cfg.return_col):
        if col not in feats.columns:
            raise KeyError(f"Missing required column '{col}' in features. cols={list(feats.columns)}")

    feats = feats.sort_values(cfg.timestamp_col).reset_index(drop=True)
    feats[cfg.target_col] = feats[cfg.return_col].shift(-1)
    feats = feats.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    # Features-only numeric columns
    exclude = {cfg.timestamp_col, "symbol", cfg.target_col}
    feature_cols = [c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found. cols={list(feats.columns)}")

    train_df, val_df, test_df = time_split(feats, cfg.timestamp_col, cfg.train_frac, cfg.val_frac)

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

    metrics = {
        "val": None if len(val_df) == 0 else evaluate(model, val_df, feature_cols, cfg.target_col),
        "test": evaluate(model, test_df, feature_cols, cfg.target_col),
        "n_rows": int(len(feats)),
        "n_features": int(len(feature_cols)),
        "X_train_nan_frac": float(nan_frac),
        "feature_cols": feature_cols,
    }

    run_ts = int(time.time())
    out_dir = cfg.baseline_models_dir / str(run_ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": cfg.target_col,
            "timestamp_col": cfg.timestamp_col,
        },
        model_path,
    )
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # update latest pointers
    cfg.baseline_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(joblib.load(model_path), cfg.baseline_models_dir / "latest.joblib")
    (cfg.baseline_models_dir / "latest.json").write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    with mlflow.start_run(run_name=f"baseline_features_only_{run_ts}") as r:
        mlflow.log_param("features_path", str(cfg.features_path))
        mlflow.log_param("ridge_alpha", float(cfg.ridge_alpha))
        mlflow.log_param("n_features", int(len(feature_cols)))
        mlflow.log_metric("X_train_nan_frac", float(nan_frac))
        mlflow.log_metric("test_mae", float(metrics["test"]["mae"]))
        mlflow.log_metric("test_rmse", float(metrics["test"]["rmse"]))
        if metrics["val"] is not None:
            mlflow.log_metric("val_mae", float(metrics["val"]["mae"]))
            mlflow.log_metric("val_rmse", float(metrics["val"]["rmse"]))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

    print("Wrote baseline:", model_path)
    print("Feature cols:", feature_cols)


if __name__ == "__main__":
    main()

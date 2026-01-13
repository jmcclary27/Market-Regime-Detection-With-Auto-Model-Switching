from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class TrainConfig:
    # Inputs
    features_path: Path = Path("data/features/test-run.parquet")
    regimes_path: Path = Path("data/regimes/test-run.parquet")

    # Outputs
    out_dir: Path = Path("data/runs")
    baseline_models_dir: Path = Path("models/baseline")

    # Pretrained expert (loaded, not trained in this script)
    pretrained_expert_path: Path = Path("models/pretrained/expert_bullish_ridge_v0.joblib")
    experts_dir: Path = Path("models/experts")  # writes to models/experts/<regime>/<run_ts>/

    # MLflow
    tracking_uri: str | None = None  # None means default local ./mlruns
    experiment_name: str = "market-regime-auto-switch"

    # Columns
    target_col: str = "target_next_return"
    timestamp_col: str = "timestamp"
    return_col: str = "log_return_1"

    # Split
    train_frac: float = 0.7
    val_frac: float = 0.15

    # Baseline model config
    ridge_alpha: float = 1.0
    random_state: int = 42


def _normalize_mlflow_uri(uri: str) -> str:
    """
    MLflow expects a URI with a scheme (file:, sqlite:, http:, ...).
    On Windows, a raw path like C:\\tmp\\mlruns is parsed with scheme 'c' and fails.
    Convert raw filesystem paths to file:// URIs.
    """
    u = uri.strip()

    # Already a URI-like value
    if "://" in u or u.startswith("file:"):
        return u

    # Treat as filesystem path
    return Path(u).resolve().as_uri()


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file, {path}")
    return pd.read_parquet(path)


def build_training_frame(cfg: TrainConfig) -> pd.DataFrame:
    feats = _load_parquet(cfg.features_path)
    regs = _load_parquet(cfg.regimes_path)

    if cfg.timestamp_col not in feats.columns or cfg.timestamp_col not in regs.columns:
        raise KeyError(
            f"Expected '{cfg.timestamp_col}' in both features and regimes. "
            f"features_cols={list(feats.columns)}, regimes_cols={list(regs.columns)}"
        )

    df = feats.merge(regs, on=cfg.timestamp_col, how="inner")

    # Clean up duplicate symbol columns if merge created suffixes
    if "symbol_x" in df.columns and "symbol_y" in df.columns:
        df = df.drop(columns=["symbol_y"]).rename(columns={"symbol_x": "symbol"})
    elif "symbol_y" in df.columns and "symbol_x" not in df.columns:
        df = df.drop(columns=["symbol_y"])

    if cfg.return_col not in df.columns:
        raise KeyError(
            f"Expected '{cfg.return_col}' column in joined frame to build target. "
            f"cols={list(df.columns)}"
        )

    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    df[cfg.target_col] = df[cfg.return_col].shift(-1)
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)
    return df


def time_split(
    df: pd.DataFrame, timestamp_col: str, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def select_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(model: Any, feature_cols: list[str], df: pd.DataFrame, target_col: str) -> dict[str, float]:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
    }


def register_expert_artifact(
    experts_root: Path,
    regime: str,
    run_ts: int,
    src_model_path: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Copies the pretrained expert into a versioned folder and updates latest pointers.
    Returns: (versioned_model_path, latest_model_path)
    """
    dest_dir = experts_root / regime / str(run_ts)
    dest_dir.mkdir(parents=True, exist_ok=True)

    versioned_model_path = dest_dir / "model.joblib"
    versioned_meta_path = dest_dir / "metadata.json"

    shutil.copy2(src_model_path, versioned_model_path)
    versioned_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_model_path = experts_root / regime / "latest.joblib"
    latest_meta_path = experts_root / regime / "latest.json"
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(versioned_model_path, latest_model_path)
    shutil.copy2(versioned_meta_path, latest_meta_path)

    return versioned_model_path, latest_model_path


def run(cfg: TrainConfig) -> str:
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(_normalize_mlflow_uri(cfg.tracking_uri))
    mlflow.set_experiment(cfg.experiment_name)

    df = build_training_frame(cfg)

    run_ts = int(time.time())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "features_path": str(cfg.features_path),
        "regimes_path": str(cfg.regimes_path),
        "target_col": cfg.target_col,
        "timestamp_col": cfg.timestamp_col,
        "return_col": cfg.return_col,
        "regime_value_counts": df["regime"].value_counts().to_dict() if "regime" in df.columns else None,
    }

    train_df, val_df, test_df = time_split(df, cfg.timestamp_col, cfg.train_frac, cfg.val_frac)

    exclude = {
        cfg.target_col,
        cfg.timestamp_col,
        "regime_explanation",
        "regime",
        "symbol",
        "symbol_x",
        "symbol_y",
    }
    baseline_feature_cols = select_feature_columns(df, exclude)
    if not baseline_feature_cols:
        raise ValueError(
            "No numeric feature columns found for baseline. "
            f"df_cols={list(df.columns)} exclude={sorted(exclude)}"
        )

    # Train baseline
    X_train = train_df[baseline_feature_cols].to_numpy()
    y_train = train_df[cfg.target_col].to_numpy()
    baseline_model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state)
    baseline_model.fit(X_train, y_train)

    baseline_val = evaluate_model(baseline_model, baseline_feature_cols, val_df, cfg.target_col)
    baseline_test = evaluate_model(baseline_model, baseline_feature_cols, test_df, cfg.target_col)

    # Save baseline artifact
    baseline_dir = cfg.baseline_models_dir / str(run_ts)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_model_path = baseline_dir / "model.joblib"
    joblib.dump(
        {
            "model": baseline_model,
            "feature_cols": baseline_feature_cols,
            "target_col": cfg.target_col,
            "timestamp_col": cfg.timestamp_col,
        },
        baseline_model_path,
    )

    # Load pretrained expert
    if not cfg.pretrained_expert_path.exists():
        raise FileNotFoundError(
            f"Pretrained expert not found at {cfg.pretrained_expert_path}. "
            "Run: python tools/make_pretrained_expert.py"
        )

    expert_bundle_any = joblib.load(cfg.pretrained_expert_path)
    expert_bundle = cast(dict[str, Any], expert_bundle_any)

    expert_model = expert_bundle["model"]
    expert_feature_cols_any = expert_bundle["feature_cols"]
    expert_feature_cols = cast(list[str], expert_feature_cols_any)

    expert_regime = str(expert_bundle.get("regime", "unknown"))
    expert_name = str(expert_bundle.get("model_name", cfg.pretrained_expert_path.stem))

    missing = [c for c in expert_feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Pretrained expert expects feature columns that are missing from current data. "
            f"missing={missing}, available_cols={list(df.columns)}"
        )

    expert_val = evaluate_model(expert_model, expert_feature_cols, val_df, cfg.target_col)
    expert_test = evaluate_model(expert_model, expert_feature_cols, test_df, cfg.target_col)

    val_reg_df = val_df[val_df["regime"] == expert_regime].copy() if "regime" in val_df.columns else val_df.iloc[0:0]
    test_reg_df = (
        test_df[test_df["regime"] == expert_regime].copy() if "regime" in test_df.columns else test_df.iloc[0:0]
    )

    expert_val_reg = evaluate_model(expert_model, expert_feature_cols, val_reg_df, cfg.target_col) if len(val_reg_df) else None
    expert_test_reg = (
        evaluate_model(expert_model, expert_feature_cols, test_reg_df, cfg.target_col) if len(test_reg_df) else None
    )

    run_meta_path = cfg.out_dir / f"run_pr4_dataset_{run_ts}.json"
    run_meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    split_summary_path = cfg.out_dir / f"split_summary_{run_ts}.json"
    split_summary_path.write_text(
        json.dumps(
            {
                "n_rows": int(len(df)),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "baseline_feature_cols": baseline_feature_cols,
                "expert_feature_cols": expert_feature_cols,
                "expert_name": expert_name,
                "expert_regime": expert_regime,
                "n_val_expert_regime": int(len(val_reg_df)),
                "n_test_expert_regime": int(len(test_reg_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name=f"pr4_baseline_plus_expert_{run_ts}") as active_run:
        # Params
        mlflow.log_param("features_path", str(cfg.features_path))
        mlflow.log_param("regimes_path", str(cfg.regimes_path))
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("timestamp_col", cfg.timestamp_col)
        mlflow.log_param("return_col", cfg.return_col)

        mlflow.log_param("train_frac", float(cfg.train_frac))
        mlflow.log_param("val_frac", float(cfg.val_frac))
        mlflow.log_param("n_train", int(len(train_df)))
        mlflow.log_param("n_val", int(len(val_df)))
        mlflow.log_param("n_test", int(len(test_df)))

        # Baseline
        mlflow.log_param("baseline_model_name", "baseline_ridge")
        mlflow.log_param("baseline_ridge_alpha", float(cfg.ridge_alpha))
        mlflow.log_param("baseline_n_features", int(len(baseline_feature_cols)))

        # Expert
        mlflow.log_param("expert_model_name", expert_name)
        mlflow.log_param("expert_model_type", "pretrained_loaded")
        mlflow.log_param("expert_regime", expert_regime)
        mlflow.log_param("expert_n_features", int(len(expert_feature_cols)))
        mlflow.log_param("expert_artifact_path", str(cfg.pretrained_expert_path))

        # Dataset metrics
        mlflow.log_metric("n_rows", float(summary["rows"]))
        mlflow.log_metric("n_cols", float(summary["cols"]))

        # Baseline metrics
        mlflow.log_metric("baseline_val_mae", baseline_val["mae"])
        mlflow.log_metric("baseline_val_rmse", baseline_val["rmse"])
        mlflow.log_metric("baseline_test_mae", baseline_test["mae"])
        mlflow.log_metric("baseline_test_rmse", baseline_test["rmse"])

        # Expert metrics
        mlflow.log_metric("expert_val_mae", expert_val["mae"])
        mlflow.log_metric("expert_val_rmse", expert_val["rmse"])
        mlflow.log_metric("expert_test_mae", expert_test["mae"])
        mlflow.log_metric("expert_test_rmse", expert_test["rmse"])

        if expert_val_reg is not None:
            mlflow.log_metric("expert_val_regime_mae", expert_val_reg["mae"])
            mlflow.log_metric("expert_val_regime_rmse", expert_val_reg["rmse"])
        if expert_test_reg is not None:
            mlflow.log_metric("expert_test_regime_mae", expert_test_reg["mae"])
            mlflow.log_metric("expert_test_regime_rmse", expert_test_reg["rmse"])

        # Register expert artifact + latest pointers
        expert_metadata: dict[str, Any] = {
            "model_name": expert_name,
            "regime": expert_regime,
            "source_path": str(cfg.pretrained_expert_path),
            "mlflow_run_id": str(active_run.info.run_id),
            "created_at_unix": run_ts,
            "feature_cols": expert_feature_cols,
            "metrics": {
                "val": expert_val,
                "test": expert_test,
                "val_regime_only": expert_val_reg,
                "test_regime_only": expert_test_reg,
            },
        }
        versioned_expert_path, latest_expert_path = register_expert_artifact(
            experts_root=cfg.experts_dir,
            regime=expert_regime,
            run_ts=run_ts,
            src_model_path=cfg.pretrained_expert_path,
            metadata=expert_metadata,
        )

        # Artifacts
        mlflow.log_artifact(str(run_meta_path))
        mlflow.log_artifact(str(split_summary_path))
        mlflow.log_artifact(str(baseline_model_path))
        mlflow.log_artifact(str(cfg.pretrained_expert_path))
        mlflow.log_artifact(str(versioned_expert_path))
        mlflow.log_artifact(str(latest_expert_path))

        return str(active_run.info.run_id)


def main() -> None:
    cfg = TrainConfig()
    run_id = run(cfg)
    print("Done. MLflow run_id:", run_id)


if __name__ == "__main__":
    main()

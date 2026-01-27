# tools/train_lightgbm_expert.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.lightgbm  # important: avoid scoping bugs
import mlflow.sklearn  # fallback
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class TrainConfig:
    features_path: str
    target_col: str
    target_expr: Optional[str]
    target_shift: int
    group_col: Optional[str]
    vol_window: Optional[int]

    model_name: str
    experiment_name: str
    run_name: str
    output_dir: str

    id_cols: List[str]
    drop_cols: List[str]
    time_col: Optional[str]

    train_frac: float
    val_frac: float
    test_frac: float

    early_stopping_rounds: int
    num_boost_round: int
    seed: int

    params_json: Optional[str]
    mlflow_tracking_uri: Optional[str]


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a LightGBM expert and log to MLflow.")

    p.add_argument("--features-path", required=True, help="Path to features data, parquet or csv.")

    # Target options:
    p.add_argument(
        "--target-col",
        required=True,
        help="Target column name. If --target-expr is set, this will be the name of the created target.",
    )
    p.add_argument(
        "--target-expr",
        default=None,
        help="Optional pandas eval expression to create the target, e.g. 'log_return_1_x - log_return_1_y'.",
    )
    p.add_argument(
        "--target-shift",
        type=int,
        default=0,
        help="Shift applied to the target. Use -1 to predict the next period.",
    )
    p.add_argument(
        "--group-col",
        default=None,
        help="Optional grouping column for target shifting (e.g. 'symbol'). If not set, shift is global.",
    )
    p.add_argument(
        "--vol-window",
        type=int,
        default=None,
        help="Optional rolling window size for volatility normalization (e.g. 20).",
    )

    p.add_argument("--model-name", default="lightgbm_expert", help="Logical name for this expert.")
    p.add_argument("--experiment-name", default="market-regime", help="MLflow experiment name.")
    p.add_argument("--run-name", default="", help="Optional MLflow run name.")
    p.add_argument("--output-dir", default="artifacts/lightgbm", help="Local output folder.")

    p.add_argument(
        "--id-cols",
        default="timestamp",
        help="Comma-separated columns to exclude from training (kept only as identifiers).",
    )
    p.add_argument(
        "--drop-cols",
        default="regime,regime_explanation",
        help="Comma-separated columns to drop if present.",
    )
    p.add_argument(
        "--time-col",
        default="timestamp",
        help="Optional time column for ordering before split. Default 'timestamp'.",
    )

    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)

    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--num-boost-round", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--params-json",
        default=None,
        help="JSON string or path to JSON file with LightGBM params.",
    )
    p.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")

    args = p.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"{args.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    return TrainConfig(
        features_path=args.features_path,
        target_col=args.target_col,
        target_expr=args.target_expr,
        target_shift=args.target_shift,
        group_col=args.group_col,
        vol_window=args.vol_window,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        run_name=run_name,
        output_dir=args.output_dir,
        id_cols=id_cols,
        drop_cols=drop_cols,
        time_col=args.time_col.strip() if args.time_col else None,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        early_stopping_rounds=args.early_stopping_rounds,
        num_boost_round=args.num_boost_round,
        seed=args.seed,
        params_json=args.params_json,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )


def _load_params(params_json: Optional[str]) -> Dict[str, Any]:
    if not params_json:
        return {}

    maybe_path = Path(params_json)
    if maybe_path.exists() and maybe_path.is_file():
        return json.loads(maybe_path.read_text(encoding="utf-8"))

    return json.loads(params_json)


def _read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"features path not found: {path}")

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type: {p.suffix}, use parquet or csv")


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _time_ordered_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"train, val, test fractions must sum to 1.0, got {total}")

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"split too small, n={n}, train={n_train}, val={n_val}, test={n_test}"
        )

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _make_xy(
    df: pd.DataFrame, target_col: str, exclude_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"target col not found: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col] + [c for c in exclude_cols if c in df.columns], errors="ignore")

    # Keep numeric columns only (simple + robust)
    X = X.select_dtypes(include=[np.number]).copy()

    # Drop all-null or constant columns
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    X = X[keep]

    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }


def _build_target(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    out = df.copy()

    # Step 1: build raw target
    if cfg.target_expr:
        try:
            out[cfg.target_col] = out.eval(cfg.target_expr, engine="python")
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate --target-expr='{cfg.target_expr}'. "
                f"Available columns: {sorted(out.columns)}"
            ) from e
    else:
        if cfg.target_col not in out.columns:
            raise KeyError(f"target col not found: {cfg.target_col}")

    # Step 2: volatility normalization (optional, based on CURRENT/PREV info only)
    if cfg.vol_window is not None:
        if cfg.vol_window <= 1:
            raise ValueError("--vol-window must be >= 2")

        vol_col = "__vol__"

        if cfg.group_col:
            if cfg.group_col not in out.columns:
                raise KeyError(
                    f"--group-col '{cfg.group_col}' not found in df columns: {sorted(out.columns)}"
                )
            out[vol_col] = (
                out.groupby(cfg.group_col, sort=False)[cfg.target_col]
                .rolling(cfg.vol_window)
                .std()
                .reset_index(level=0, drop=True)
            )
        else:
            out[vol_col] = out[cfg.target_col].rolling(cfg.vol_window).std()

        # Avoid divide-by-zero / NaNs from early window
        out[cfg.target_col] = out[cfg.target_col] / out[vol_col]
        out.loc[out[vol_col].isna() | (out[vol_col] == 0.0), cfg.target_col] = np.nan

        out = out.drop(columns=[vol_col])

    # Step 3: shift target to next period if requested
    if cfg.target_shift != 0:
        if cfg.group_col:
            out[cfg.target_col] = (
                out.groupby(cfg.group_col, sort=False)[cfg.target_col].shift(cfg.target_shift)
            )
        else:
            out[cfg.target_col] = out[cfg.target_col].shift(cfg.target_shift)

    return out


def main() -> None:
    cfg = _parse_args()

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    mlflow.set_experiment(cfg.experiment_name)

    df = _read_df(cfg.features_path)

    # Drop junk columns if present
    df = df.drop(columns=[c for c in cfg.drop_cols if c in df.columns], errors="ignore")

    # Time ordering if requested and present
    if cfg.time_col and cfg.time_col in df.columns:
        dt = _safe_to_datetime(df[cfg.time_col])
        df = df.assign(_dt=dt).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Build / shift target
    df = _build_target(df, cfg)

    # Remove missing targets
    df = df[df[cfg.target_col].notna()].reset_index(drop=True)

    # Split
    train_df, val_df, test_df = _time_ordered_split(df, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    exclude_cols = list(cfg.id_cols)

    X_train, y_train = _make_xy(train_df, cfg.target_col, exclude_cols)
    X_val, y_val = _make_xy(val_df, cfg.target_col, exclude_cols)
    X_test, y_test = _make_xy(test_df, cfg.target_col, exclude_cols)

    if X_train.shape[1] == 0:
        raise ValueError("no numeric training features after filtering, check your feature file")

    # Align columns across splits
    cols = list(X_train.columns)
    X_val = X_val.reindex(columns=cols)
    X_test = X_test.reindex(columns=cols)

    user_params = _load_params(cfg.params_json)

    # Sensible defaults for tabular finance features
    params: Dict[str, Any] = {
        "n_estimators": cfg.num_boost_round,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": cfg.seed,
        "n_jobs": -1,
    }
    params.update(user_params)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "features_path": cfg.features_path,
                "target_col": cfg.target_col,
                "target_expr": cfg.target_expr or "",
                "target_shift": cfg.target_shift,
                "group_col": cfg.group_col or "",
                "vol_window": int(cfg.vol_window) if cfg.vol_window is not None else 0,
                "train_frac": cfg.train_frac,
                "val_frac": cfg.val_frac,
                "test_frac": cfg.test_frac,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                "num_boost_round": cfg.num_boost_round,
                "seed": cfg.seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_features": int(X_train.shape[1]),
            }
        )
        mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})

        model = LGBMRegressor(**params)

        # Early stopping is version-dependent, keep it robust
        try:
            from lightgbm import early_stopping  # type: ignore

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
        except Exception:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
            )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = _metrics(y_val.to_numpy(), val_pred)
        test_metrics = _metrics(y_test.to_numpy(), test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Save feature list
        (out_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "feature_columns.json"))

        # Log model to MLflow (keep your current behavior)
        try:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                input_example=X_train.head(5),
            )
        except Exception:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=X_train.head(5),
            )

        mlflow.set_tag("expert_type", "lightgbm")
        mlflow.set_tag("expert_name", cfg.model_name)
        mlflow.set_tag("run_id", run.info.run_id)

    print("done")


if __name__ == "__main__":
    main()

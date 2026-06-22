# tools/train_lightgbm_expert.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.lightgbm  # important: avoid scoping bugs
import mlflow.sklearn  # fallback
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.stationary import (
    augment_pairwise_stationary_features,
    summarize_feature_ranges,
)


@dataclass(frozen=True)
class TrainConfig:
    features_path: str
    regimes_path: str | None
    target_col: str
    target_expr: str | None
    target_shift: int
    group_col: str | None
    vol_window: int | None
    min_regime_rows: int

    # NEW: which regime this expert is for (controls output path + naming)
    regime: str

    model_name: str
    experiment_name: str
    run_name: str
    output_dir: str

    id_cols: list[str]
    drop_cols: list[str]
    time_col: str | None

    train_frac: float
    val_frac: float
    test_frac: float

    early_stopping_rounds: int
    num_boost_round: int
    seed: int

    params_json: str | None
    mlflow_tracking_uri: str | None


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a LightGBM expert and log to MLflow.")

    p.add_argument("--features-path", required=True, help="Path to features data, parquet or csv.")
    p.add_argument(
        "--regimes-path",
        default=None,
        help="Optional regimes parquet/csv to merge when --features-path lacks a regime column.",
    )

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

    # NEW: regime controls output folder under models/experts/<regime>/
    p.add_argument(
        "--regime",
        required=True,
        choices=["bullish", "bearish", "sideways"],
        help="Which regime expert this model is for. Writes to models/experts/<regime>/",
    )
    p.add_argument(
        "--min-regime-rows",
        type=int,
        default=100,
        help="Minimum required rows for the selected regime after target prep.",
    )

    p.add_argument("--model-name", default="lightgbm_expert", help="Logical name for this expert.")
    p.add_argument("--experiment-name", default="market-regime", help="MLflow experiment name.")
    p.add_argument("--run-name", default="", help="Optional MLflow run name.")

    # IMPORTANT: now this is the ROOT experts directory (not a per-run leaf dir)
    p.add_argument(
        "--output-dir",
        default="models/experts",
        help="Root output folder. Model will be written under models/experts/<regime>/<timestamp>/ and latest.joblib",
    )

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
        regimes_path=args.regimes_path,
        target_col=args.target_col,
        target_expr=args.target_expr,
        target_shift=args.target_shift,
        group_col=args.group_col,
        vol_window=args.vol_window,
        min_regime_rows=args.min_regime_rows,
        regime=args.regime,
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


def _load_params(params_json: str | None) -> dict[str, Any]:
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


def _ensure_regime_columns(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if "regime" in df.columns:
        return df

    if not cfg.regimes_path:
        raise KeyError(
            "Input data does not contain a 'regime' column. "
            "Pass --regimes-path or train directly from data/regimes/latest.parquet."
        )

    join_col = cfg.time_col or "timestamp"
    if join_col not in df.columns:
        raise KeyError(
            f"Cannot merge regimes because join column '{join_col}' is missing from features. "
            f"Available columns: {sorted(df.columns)}"
        )

    reg_df = _read_df(cfg.regimes_path)
    if join_col not in reg_df.columns:
        raise KeyError(
            f"Cannot merge regimes because join column '{join_col}' is missing from regimes file. "
            f"Available columns: {sorted(reg_df.columns)}"
        )
    if "regime" not in reg_df.columns:
        raise KeyError(
            f"Regimes file must contain a 'regime' column. Available columns: {sorted(reg_df.columns)}"
        )

    reg_cols = [join_col, "regime"]
    if "regime_explanation" in reg_df.columns:
        reg_cols.append("regime_explanation")

    reg_df = reg_df.loc[:, reg_cols].drop_duplicates(subset=[join_col], keep="last")
    merged = df.merge(reg_df, on=join_col, how="left")
    if "regime" not in merged.columns:
        raise KeyError("Failed to merge regime labels into the training dataframe.")
    return merged


def _time_ordered_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"train, val, test fractions must sum to 1.0, got {total}")

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"split too small, n={n}, train={n_train}, val={n_val}, test={n_test}")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _make_xy(
    df: pd.DataFrame, target_col: str, exclude_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"target col not found: {target_col}")

    y = df[target_col]
    X = df.drop(
        columns=[target_col] + [c for c in exclude_cols if c in df.columns], errors="ignore"
    )

    # Keep numeric columns only (simple + robust)
    X = X.select_dtypes(include=[np.number]).copy()

    # Drop all-null or constant columns
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    X = X[keep]

    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
            out[cfg.target_col] = out.groupby(cfg.group_col, sort=False)[cfg.target_col].shift(
                cfg.target_shift
            )
        else:
            out[cfg.target_col] = out[cfg.target_col].shift(cfg.target_shift)

    return out


def _finite_nunique(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def _preserve_legacy_arima_latest(latest_dir: Path) -> None:
    legacy = latest_dir / "latest.json"
    new_path = latest_dir / "latest.arima.json"
    if not legacy.exists() or new_path.exists():
        return

    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
    except Exception:
        return

    if isinstance(payload, dict) and str(payload.get("model_type", "")).lower() == "arima":
        new_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cfg: TrainConfig) -> Path:
    if cfg.min_regime_rows <= 0:
        raise ValueError("--min-regime-rows must be >= 1")

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    mlflow.set_experiment(cfg.experiment_name)

    df = _read_df(cfg.features_path)
    df = _ensure_regime_columns(df, cfg)

    # Time ordering if requested and present
    if cfg.time_col and cfg.time_col in df.columns:
        dt = _safe_to_datetime(df[cfg.time_col])
        df = df.assign(_dt=dt).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Add stationary / relative pairwise features before target shifting so the
    # raw columns remain available for feature engineering.
    df, stationary_cols = augment_pairwise_stationary_features(df)

    # Build / shift target
    df = _build_target(df, cfg)

    # Remove missing targets
    df = df[df[cfg.target_col].notna()].reset_index(drop=True)
    if "regime" not in df.columns:
        raise KeyError("Training dataframe is missing required 'regime' column after target prep.")

    regime_series = df["regime"].astype("string").str.lower().str.strip()
    df = df.loc[regime_series == cfg.regime].reset_index(drop=True)
    if len(df) < cfg.min_regime_rows:
        raise ValueError(
            f"Not enough rows for regime '{cfg.regime}' after target prep: "
            f"{len(df)} < {cfg.min_regime_rows}"
        )

    # Split
    train_df, val_df, test_df = _time_ordered_split(df, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    exclude_cols = list(dict.fromkeys([*cfg.id_cols, *cfg.drop_cols]))

    X_train, y_train = _make_xy(train_df, cfg.target_col, exclude_cols)
    X_val, y_val = _make_xy(val_df, cfg.target_col, exclude_cols)
    X_test, y_test = _make_xy(test_df, cfg.target_col, exclude_cols)

    if X_train.shape[1] == 0:
        raise ValueError("no numeric training features after filtering, check your feature file")

    # Align columns across splits
    cols = list(X_train.columns)
    X_val = X_val.reindex(columns=cols)
    X_test = X_test.reindex(columns=cols)

    feature_range_stats = summarize_feature_ranges(X_train, cols)

    user_params = _load_params(cfg.params_json)

    # Sensible defaults for tabular finance features
    params: dict[str, Any] = {
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

    # ----------------------------
    # NEW: output layout
    # models/experts/<regime>/<timestamp>/
    # models/experts/<regime>/latest.joblib
    # ----------------------------
    ts_slug = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg.output_dir)
    out_dir = root / cfg.regime / ts_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = root / cfg.regime
    latest_dir.mkdir(parents=True, exist_ok=True)
    _preserve_legacy_arima_latest(latest_dir)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "regime": cfg.regime,
                "regimes_path": cfg.regimes_path or "",
                "features_path": cfg.features_path,
                "target_col": cfg.target_col,
                "target_expr": cfg.target_expr or "",
                "target_shift": cfg.target_shift,
                "group_col": cfg.group_col or "",
                "vol_window": int(cfg.vol_window) if cfg.vol_window is not None else 0,
                "min_regime_rows": cfg.min_regime_rows,
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
        val_pred_nunique = _finite_nunique(np.asarray(val_pred, dtype=float))
        test_pred_nunique = _finite_nunique(np.asarray(test_pred, dtype=float))
        if val_pred_nunique <= 1 or test_pred_nunique <= 1:
            raise ValueError(
                "Refusing to save LightGBM expert because predictions collapsed to a constant. "
                f"val_pred_nunique={val_pred_nunique} test_pred_nunique={test_pred_nunique}"
            )

        val_metrics = _metrics(y_val.to_numpy(), val_pred)
        test_metrics = _metrics(y_test.to_numpy(), test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("val_pred_nunique", float(val_pred_nunique))
        mlflow.log_metric("test_pred_nunique", float(test_pred_nunique))

        # Save feature list (local)
        feature_cols_path = out_dir / "feature_columns.json"
        feature_cols_path.write_text(json.dumps(cols, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(feature_cols_path))

        # Convenience copy for "latest"
        (latest_dir / "feature_columns.json").write_text(
            json.dumps(cols, indent=2), encoding="utf-8"
        )

        # NEW: Save joblib artifacts for your pipeline
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)

        latest_model_path = latest_dir / "latest.joblib"
        joblib.dump(model, latest_model_path)

        metadata = {
            "model_type": "lightgbm",
            "model_name": cfg.model_name,
            "regime": cfg.regime,
            "features_path": cfg.features_path,
            "regimes_path": cfg.regimes_path,
            "target_col": cfg.target_col,
            "target_expr": cfg.target_expr,
            "target_shift": cfg.target_shift,
            "group_col": cfg.group_col,
            "vol_window": cfg.vol_window,
            "min_regime_rows": cfg.min_regime_rows,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "test_frac": cfg.test_frac,
            "n_rows_used": int(len(df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_features": int(len(cols)),
            "feature_columns": cols,
            "stationary_feature_columns": stationary_cols,
            "feature_range_stats": feature_range_stats,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_pred_nunique": int(val_pred_nunique),
            "test_pred_nunique": int(test_pred_nunique),
            "run_id": run.info.run_id,
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path))
        (latest_dir / "latest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        mlflow.set_tag("expert_regime", cfg.regime)
        mlflow.set_tag("run_id", run.info.run_id)

    print(f"Wrote: {out_dir}")
    print(f"Wrote: {latest_dir}")
    print("done")
    return out_dir


def main() -> None:
    cfg = _parse_args()
    run(cfg)


if __name__ == "__main__":
    main()

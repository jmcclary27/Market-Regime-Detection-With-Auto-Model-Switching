# tools/train_arima_expert.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class TrainConfig:
    features_path: str
    target_col: str
    target_expr: str | None
    target_shift: int
    group_col: str | None
    vol_window: int | None

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

    # ARIMA specific
    p: int
    d: int
    q: int
    trend: str
    refit_interval: int
    train_window: int | None
    min_train_size: int

    seed: int
    mlflow_tracking_uri: str | None


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train an ARIMA expert and log to MLflow.")

    p.add_argument("--features-path", required=True)

    p.add_argument("--target-col", required=True)
    p.add_argument("--target-expr", default=None)
    p.add_argument("--target-shift", type=int, default=0)
    p.add_argument("--group-col", default=None)
    p.add_argument("--vol-window", type=int, default=None)

    p.add_argument("--regime", required=True, choices=["bullish", "bearish", "sideways"])

    p.add_argument("--model-name", default="arima_expert")
    p.add_argument("--experiment-name", default="market-regime")
    p.add_argument("--run-name", default="")

    p.add_argument("--output-dir", default="models/experts")

    p.add_argument("--id-cols", default="timestamp")
    p.add_argument("--drop-cols", default="regime,regime_explanation")
    p.add_argument("--time-col", default="timestamp")

    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)

    # ARIMA
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--d", type=int, default=0)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--trend", default="c", choices=["n", "c", "t", "ct"])
    p.add_argument(
        "--refit-interval", type=int, default=10, help="Refit every N steps during walk-forward."
    )
    p.add_argument(
        "--train-window", type=int, default=None, help="Optional rolling window length, e.g. 252."
    )
    p.add_argument(
        "--min-train-size", type=int, default=60, help="Minimum points required before first fit."
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow-tracking-uri", default=None)

    args = p.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"{args.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    if args.refit_interval <= 0:
        raise ValueError("--refit-interval must be >= 1")
    if args.min_train_size <= 10:
        raise ValueError("--min-train-size should be >= 10")

    return TrainConfig(
        features_path=args.features_path,
        target_col=args.target_col,
        target_expr=args.target_expr,
        target_shift=args.target_shift,
        group_col=args.group_col,
        vol_window=args.vol_window,
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
        p=args.p,
        d=args.d,
        q=args.q,
        trend=args.trend,
        refit_interval=args.refit_interval,
        train_window=args.train_window,
        min_train_size=args.min_train_size,
        seed=args.seed,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )


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

    if cfg.vol_window is not None:
        if cfg.vol_window <= 1:
            raise ValueError("--vol-window must be >= 2")

        vol_col = "__vol__"
        if cfg.group_col:
            if cfg.group_col not in out.columns:
                raise KeyError(f"--group-col '{cfg.group_col}' not found in df columns")
            out[vol_col] = (
                out.groupby(cfg.group_col, sort=False)[cfg.target_col]
                .rolling(cfg.vol_window)
                .std()
                .reset_index(level=0, drop=True)
            )
        else:
            out[vol_col] = out[cfg.target_col].rolling(cfg.vol_window).std()

        out[cfg.target_col] = out[cfg.target_col] / out[vol_col]
        out.loc[out[vol_col].isna() | (out[vol_col] == 0.0), cfg.target_col] = np.nan
        out = out.drop(columns=[vol_col])

    if cfg.target_shift != 0:
        if cfg.group_col:
            out[cfg.target_col] = out.groupby(cfg.group_col, sort=False)[cfg.target_col].shift(
                cfg.target_shift
            )
        else:
            out[cfg.target_col] = out[cfg.target_col].shift(cfg.target_shift)

    return out


def _walk_forward_arima(
    y_train: np.ndarray,
    y_future: np.ndarray,
    order: tuple[int, int, int],
    trend: str,
    refit_interval: int,
    train_window: int | None,
    min_train_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Leakage-safe 1-step-ahead walk-forward predictions over y_future,
    conditioning only on information available up to each step.
    """
    history: list[float] = list(map(float, y_train))
    preds: list[float] = []
    n_refits = 0
    n_fail = 0
    last_fit_i = -(10**9)
    last_result: Any | None = None

    for i, y_true in enumerate(y_future):
        # training slice for this step
        if train_window is not None and train_window > 0:
            hist_used = history[-train_window:]
        else:
            hist_used = history

        if len(hist_used) < min_train_size:
            preds.append(float("nan"))
            history.append(float(y_true))
            continue

        need_refit = (i - last_fit_i) >= refit_interval or last_result is None

        if need_refit:
            try:
                model = ARIMA(np.asarray(hist_used, dtype=float), order=order, trend=trend)
                last_result = model.fit()
                last_fit_i = i
                n_refits += 1
            except Exception:
                last_result = None
                n_fail += 1

        if last_result is None:
            preds.append(float("nan"))
        else:
            try:
                fc = last_result.forecast(steps=1)
                preds.append(float(fc[0]))
            except Exception:
                preds.append(float("nan"))
                n_fail += 1
                last_result = None

        history.append(float(y_true))

    pred_arr = np.asarray(preds, dtype=float)

    diag = {
        "n_refits": float(n_refits),
        "n_fit_failures": float(n_fail),
        "nan_rate": float(np.mean(~np.isfinite(pred_arr))) if len(pred_arr) else float("nan"),
    }
    return pred_arr, diag


def main() -> None:
    cfg = _parse_args()

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    mlflow.set_experiment(cfg.experiment_name)

    df = _read_df(cfg.features_path)
    df = df.drop(columns=[c for c in cfg.drop_cols if c in df.columns], errors="ignore")

    if cfg.time_col and cfg.time_col in df.columns:
        dt = _safe_to_datetime(df[cfg.time_col])
        df = df.assign(_dt=dt).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = _build_target(df, cfg)
    df = df[df[cfg.target_col].notna()].reset_index(drop=True)

    train_df, val_df, test_df = _time_ordered_split(df, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    y_train = train_df[cfg.target_col].to_numpy(dtype=float)
    y_val = val_df[cfg.target_col].to_numpy(dtype=float)
    y_test = test_df[cfg.target_col].to_numpy(dtype=float)

    order = (cfg.p, cfg.d, cfg.q)

    ts_slug = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg.output_dir)
    out_dir = root / cfg.regime / ts_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = root / cfg.regime
    latest_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "regime": cfg.regime,
                "features_path": cfg.features_path,
                "target_col": cfg.target_col,
                "target_expr": cfg.target_expr or "",
                "target_shift": cfg.target_shift,
                "group_col": cfg.group_col or "",
                "vol_window": int(cfg.vol_window) if cfg.vol_window is not None else 0,
                "train_frac": cfg.train_frac,
                "val_frac": cfg.val_frac,
                "test_frac": cfg.test_frac,
                "seed": cfg.seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "arima_p": cfg.p,
                "arima_d": cfg.d,
                "arima_q": cfg.q,
                "arima_trend": cfg.trend,
                "refit_interval": cfg.refit_interval,
                "train_window": int(cfg.train_window) if cfg.train_window is not None else 0,
                "min_train_size": cfg.min_train_size,
            }
        )

        # Walk-forward on val
        val_pred, val_diag = _walk_forward_arima(
            y_train=y_train,
            y_future=y_val,
            order=order,
            trend=cfg.trend,
            refit_interval=cfg.refit_interval,
            train_window=cfg.train_window,
            min_train_size=cfg.min_train_size,
        )

        # Walk-forward on test, conditioning on train+val history, not test future
        y_train_plus_val = np.concatenate([y_train, y_val]).astype(float)
        test_pred, test_diag = _walk_forward_arima(
            y_train=y_train_plus_val,
            y_future=y_test,
            order=order,
            trend=cfg.trend,
            refit_interval=cfg.refit_interval,
            train_window=cfg.train_window,
            min_train_size=cfg.min_train_size,
        )

        # Drop NaNs for metrics
        def _finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            m = np.isfinite(y_true) & np.isfinite(y_pred)
            if not np.any(m):
                return {
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                }
            return _metrics(y_true[m], y_pred[m])

        val_metrics = _finite_metrics(y_val, val_pred)
        test_metrics = _finite_metrics(y_test, test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.log_metrics({f"val_{k}": v for k, v in val_diag.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_diag.items()})

        meta = {
            "model_type": "arima",
            "model_name": cfg.model_name,
            "regime": cfg.regime,
            "order": {"p": cfg.p, "d": cfg.d, "q": cfg.q},
            "trend": cfg.trend,
            "refit_interval": cfg.refit_interval,
            "train_window": cfg.train_window,
            "min_train_size": cfg.min_train_size,
            "target_col": cfg.target_col,
            "target_expr": cfg.target_expr,
            "target_shift": cfg.target_shift,
            "group_col": cfg.group_col,
            "vol_window": cfg.vol_window,
            "run_id": run.info.run_id,
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }

        meta_path = out_dir / "model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_path))

        latest_meta_path = latest_dir / "latest.json"
        latest_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        mlflow.set_tag("expert_type", "arima")
        mlflow.set_tag("expert_name", cfg.model_name)
        mlflow.set_tag("expert_regime", cfg.regime)
        mlflow.set_tag("run_id", run.info.run_id)

    print(f"Wrote: {out_dir}")
    print(f"Wrote: {latest_dir}")
    print("done")


if __name__ == "__main__":
    main()

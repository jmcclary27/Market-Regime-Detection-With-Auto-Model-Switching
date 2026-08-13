from __future__ import annotations

import argparse
import json
import platform
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import statsmodels
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

REGIMES = {"bullish", "bearish", "sideways"}


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

    regime: str

    model_name: str
    experiment_name: str
    run_name: str
    output_dir: str
    publish_latest: bool
    update_legacy_pointer: bool

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
    p = argparse.ArgumentParser(
        description="Train a regime-specific ARIMA candidate and log to MLflow."
    )

    p.add_argument("--features-path", required=True)
    p.add_argument(
        "--regimes-path",
        default=None,
        help="Regime labels to merge when --features-path does not already contain 'regime'.",
    )

    p.add_argument("--target-col", required=True)
    p.add_argument("--target-expr", default=None)
    p.add_argument("--target-shift", type=int, default=0)
    p.add_argument("--group-col", default=None)
    p.add_argument("--vol-window", type=int, default=None)
    p.add_argument(
        "--min-regime-rows",
        type=int,
        default=180,
        help="Minimum labelled rows required after target construction and regime filtering.",
    )

    p.add_argument("--regime", required=True, choices=sorted(REGIMES))

    p.add_argument("--model-name", default="arima_expert")
    p.add_argument("--experiment-name", default="market-regime")
    p.add_argument("--run-name", default="")
    p.add_argument(
        "--output-dir",
        default="models/candidates/arima",
        help="Candidate artifact root. This is intentionally not scanned by live inference.",
    )
    p.add_argument(
        "--publish-latest",
        action="store_true",
        help=(
            "Explicitly publish canonical metadata at "
            "<output-dir>/<regime>/arima/<model_id>.json after all validation checks pass."
        ),
    )
    p.add_argument(
        "--update-legacy-pointer",
        action="store_true",
        help=(
            "With --publish-latest only, also update the legacy single-expert "
            "<output-dir>/<regime>/latest.arima.json pointer."
        ),
    )

    p.add_argument("--id-cols", default="timestamp")
    p.add_argument("--drop-cols", default="regime,regime_explanation")
    p.add_argument("--time-col", default="timestamp")

    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)

    p.add_argument("--p", type=int, default=1)
    p.add_argument("--d", type=int, default=0)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--trend", default="c", choices=["n", "c", "t", "ct"])
    p.add_argument(
        "--refit-interval",
        type=int,
        default=21,
        help="Refit every N regime observations during walk-forward (default: 21).",
    )
    p.add_argument(
        "--train-window",
        type=int,
        default=504,
        help="Bound each ARIMA refit to this many regime observations (default: 504).",
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
        run_name = f"{args.model_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    if args.refit_interval <= 0:
        raise ValueError("--refit-interval must be >= 1")
    if args.min_train_size <= 10:
        raise ValueError("--min-train-size should be >= 10")
    if args.min_regime_rows <= 0:
        raise ValueError("--min-regime-rows must be >= 1")
    if args.update_legacy_pointer and not args.publish_latest:
        raise ValueError("--update-legacy-pointer requires --publish-latest")

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
        publish_latest=bool(args.publish_latest),
        update_legacy_pointer=bool(args.update_legacy_pointer),
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


def _normalize_mlflow_uri(uri: str) -> str:
    normalized = uri.strip()
    if "://" in normalized or normalized.startswith("file:"):
        return normalized
    return Path(normalized).resolve().as_uri()


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _ensure_regime_column(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Attach one regime label per timestamp without re-merging feature columns."""
    if "regime" in df.columns:
        return df.copy()

    if not cfg.regimes_path:
        raise KeyError(
            "Training ARIMA experts requires a 'regime' column. Pass a regime-labelled "
            "--features-path or provide --regimes-path."
        )

    join_col = cfg.time_col or "timestamp"
    if join_col not in df.columns:
        raise KeyError(
            f"Cannot merge regimes: features are missing '{join_col}'. "
            f"Available columns: {sorted(df.columns)}"
        )

    regimes_df = _read_df(cfg.regimes_path)
    required = {join_col, "regime"}
    missing = sorted(required - set(regimes_df.columns))
    if missing:
        raise KeyError(
            f"Regimes file is missing required columns {missing}. "
            f"Available columns: {sorted(regimes_df.columns)}"
        )

    labels = regimes_df.loc[:, [join_col, "regime"]].drop_duplicates(subset=[join_col], keep="last")
    merged = df.merge(labels, on=join_col, how="left", validate="many_to_one")
    if "regime" not in merged.columns:
        raise KeyError("Failed to merge regime labels into the ARIMA training frame.")
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
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate --target-expr='{cfg.target_expr}'. "
                f"Available columns: {sorted(out.columns)}"
            ) from exc
    elif cfg.target_col not in out.columns:
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


def _select_regime_rows(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Filter after building the next-step target, so the forecast horizon stays one period."""
    if "regime" not in df.columns:
        raise KeyError("Training dataframe is missing required 'regime' column.")

    requested = cfg.regime.strip().lower()
    if requested not in REGIMES:
        raise ValueError(f"Unsupported regime '{cfg.regime}'. Expected one of {sorted(REGIMES)}")

    normalized = df["regime"].astype("string").str.strip().str.lower()
    return df.loc[normalized == requested].reset_index(drop=True)


def _model_id(cfg: TrainConfig) -> str:
    """Build a stable, path-safe model ID for multiple ARIMA experts per regime."""
    configured_name = cfg.model_name.strip().lower()
    if not configured_name:
        raise ValueError("model_name must not be blank")
    slug = re.sub(r"[^a-z0-9]+", "_", configured_name).strip("_")
    if not slug:
        raise ValueError(f"model_name '{cfg.model_name}' does not contain a usable identifier")
    regime = cfg.regime.strip().lower()
    return f"expert_arima_{regime}_{slug}"


def _walk_forward_arima(
    y_train: np.ndarray,
    y_future: np.ndarray,
    order: tuple[int, int, int],
    trend: str,
    refit_interval: int,
    train_window: int | None,
    min_train_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Leakage-safe one-step walk-forward predictions over ``y_future``."""
    history: list[float] = list(map(float, y_train))
    preds: list[float] = []
    n_refits = 0
    n_fail = 0
    last_fit_i = -(10**9)
    last_result: Any | None = None

    for i, y_true in enumerate(y_future):
        hist_used = history[-train_window:] if train_window and train_window > 0 else history

        if len(hist_used) < min_train_size:
            preds.append(float("nan"))
            history.append(float(y_true))
            continue

        need_refit = (i - last_fit_i) >= refit_interval or last_result is None
        if need_refit:
            try:
                import warnings

                from statsmodels.tools.sm_exceptions import ConvergenceWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    last_result = ARIMA(
                        np.asarray(hist_used, dtype=float), order=order, trend=trend
                    ).fit()
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

        # A one-step forecast must condition on the realized observation before
        # the next prediction. Without this state update, every point between
        # scheduled refits receives the same stale forecast.
        if last_result is not None:
            try:
                last_result = last_result.append(np.asarray([y_true], dtype=float), refit=False)
            except Exception:
                last_result = None
                n_fail += 1

        history.append(float(y_true))

    pred_arr = np.asarray(preds, dtype=float)
    diag = {
        "n_refits": float(n_refits),
        "n_fit_failures": float(n_fail),
        "nan_rate": float(np.mean(~np.isfinite(pred_arr))) if len(pred_arr) else float("nan"),
    }
    return pred_arr, diag


def _finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        raise ValueError("ARIMA produced no finite walk-forward predictions.")
    return _metrics(y_true[mask], y_pred[mask])


def _finite_nunique(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    return int(np.unique(finite).size) if finite.size else 0


def _zero_return_quality_gate(
    y_val: np.ndarray,
    val_metrics: dict[str, float],
    y_test: np.ndarray,
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    zero_return_val_rmse = float(np.sqrt(np.mean(np.square(y_val))))
    zero_return_test_rmse = float(np.sqrt(np.mean(np.square(y_test))))
    val_rmse = float(val_metrics["rmse"])
    test_rmse = float(test_metrics["rmse"])
    reasons: list[str] = []
    if val_rmse > zero_return_val_rmse:
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


def _timestamp_bounds(df: pd.DataFrame, time_col: str | None) -> dict[str, str | None]:
    if not time_col or time_col not in df.columns or df.empty:
        return {"start": None, "end": None}
    times = _safe_to_datetime(df[time_col])
    start = times.min()
    end = times.max()
    return {
        "start": start.isoformat() if pd.notna(start) else None,
        "end": end.isoformat() if pd.notna(end) else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def run(cfg: TrainConfig) -> Path:
    """Train and validate a regime-filtered ARIMA candidate without publishing it by default."""
    if cfg.min_regime_rows <= 0:
        raise ValueError("min_regime_rows must be >= 1")
    if cfg.refit_interval <= 0:
        raise ValueError("refit_interval must be >= 1")
    if cfg.min_train_size <= 10:
        raise ValueError("min_train_size should be >= 10")
    if cfg.update_legacy_pointer and not cfg.publish_latest:
        raise ValueError("update_legacy_pointer requires publish_latest=True")

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(_normalize_mlflow_uri(cfg.mlflow_tracking_uri))
    mlflow.set_experiment(cfg.experiment_name)

    df = _ensure_regime_column(_read_df(cfg.features_path), cfg)
    source_rows = len(df)

    if cfg.time_col and cfg.time_col in df.columns:
        df = (
            df.assign(_dt=_safe_to_datetime(df[cfg.time_col]))
            .sort_values("_dt")
            .drop(columns=["_dt"])
            .reset_index(drop=True)
        )
    else:
        df = df.reset_index(drop=True)

    # Build the next-period target on the full time series *before* selecting a regime.
    # Filtering first would turn target_shift=-1 into "next matching regime", which is wrong.
    df = _build_target(df, cfg)
    df = df.loc[np.isfinite(pd.to_numeric(df[cfg.target_col], errors="coerce"))].reset_index(
        drop=True
    )
    target_ready_rows = len(df)
    regime_df = _select_regime_rows(df, cfg)

    if len(regime_df) < cfg.min_regime_rows:
        raise ValueError(
            f"Not enough labelled rows for ARIMA regime '{cfg.regime}' after target construction: "
            f"{len(regime_df)} < {cfg.min_regime_rows}. "
            "Regenerate or correct regime labels before retraining; no artifact was written."
        )

    train_df, val_df, test_df = _time_ordered_split(
        regime_df, cfg.train_frac, cfg.val_frac, cfg.test_frac
    )
    if len(train_df) < cfg.min_train_size:
        raise ValueError(
            f"ARIMA training split for '{cfg.regime}' has {len(train_df)} rows, below "
            f"min_train_size={cfg.min_train_size}. No artifact was written."
        )

    y_train = train_df[cfg.target_col].to_numpy(dtype=float)
    y_val = val_df[cfg.target_col].to_numpy(dtype=float)
    y_test = test_df[cfg.target_col].to_numpy(dtype=float)
    order = (cfg.p, cfg.d, cfg.q)

    model_id = _model_id(cfg)
    ts_slug = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    root = Path(cfg.output_dir)
    out_dir = root / cfg.regime / "arima" / model_id / ts_slug
    canonical_metadata_path = root / cfg.regime / "arima" / f"{model_id}.json"

    with mlflow.start_run(run_name=cfg.run_name) as run_info:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "model_id": model_id,
                "regime": cfg.regime,
                "features_path": cfg.features_path,
                "regimes_path": cfg.regimes_path or "",
                "target_col": cfg.target_col,
                "target_expr": cfg.target_expr or "",
                "target_shift": cfg.target_shift,
                "group_col": cfg.group_col or "",
                "vol_window": int(cfg.vol_window) if cfg.vol_window is not None else 0,
                "min_regime_rows": cfg.min_regime_rows,
                "train_frac": cfg.train_frac,
                "val_frac": cfg.val_frac,
                "test_frac": cfg.test_frac,
                "seed": cfg.seed,
                "n_source_rows": source_rows,
                "n_target_ready_rows": target_ready_rows,
                "n_regime_rows": len(regime_df),
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
                "publish_latest": cfg.publish_latest,
                "update_legacy_pointer": cfg.update_legacy_pointer,
            }
        )

        val_pred, val_diag = _walk_forward_arima(
            y_train=y_train,
            y_future=y_val,
            order=order,
            trend=cfg.trend,
            refit_interval=cfg.refit_interval,
            train_window=cfg.train_window,
            min_train_size=cfg.min_train_size,
        )
        test_pred, test_diag = _walk_forward_arima(
            y_train=np.concatenate([y_train, y_val]).astype(float),
            y_future=y_test,
            order=order,
            trend=cfg.trend,
            refit_interval=cfg.refit_interval,
            train_window=cfg.train_window,
            min_train_size=cfg.min_train_size,
        )

        val_metrics = _finite_metrics(y_val, val_pred)
        test_metrics = _finite_metrics(y_test, test_pred)
        val_pred_nunique = _finite_nunique(val_pred)
        test_pred_nunique = _finite_nunique(test_pred)
        if val_pred_nunique <= 1 or test_pred_nunique <= 1:
            raise ValueError(
                "Refusing to save ARIMA candidate because walk-forward predictions collapsed "
                f"to a constant (val={val_pred_nunique}, test={test_pred_nunique})."
            )
        if val_diag["nan_rate"] > 0.10 or test_diag["nan_rate"] > 0.10:
            raise ValueError(
                "Refusing to save ARIMA candidate because too many walk-forward predictions "
                f"were non-finite (val_nan_rate={val_diag['nan_rate']:.3f}, "
                f"test_nan_rate={test_diag['nan_rate']:.3f})."
            )

        quality_gate = _zero_return_quality_gate(y_val, val_metrics, y_test, test_metrics)
        promotion_eligible = bool(quality_gate["promotion_eligible"])

        mlflow.log_metrics({f"val_{key}": value for key, value in val_metrics.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics({f"val_{key}": value for key, value in val_diag.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_diag.items()})
        mlflow.log_metric("val_pred_nunique", float(val_pred_nunique))
        mlflow.log_metric("test_pred_nunique", float(test_pred_nunique))
        mlflow.log_metric("zero_return_test_rmse", float(quality_gate["zero_return_test_rmse"]))

        meta: dict[str, Any] = {
            "artifact_contract_version": 2,
            "model_type": "arima",
            "model_name": cfg.model_name,
            "model_id": model_id,
            "regime": cfg.regime,
            "training_regime": cfg.regime,
            "regime_column": "regime",
            "regime_filter_applied": True,
            "regime_history_policy": "filter_to_training_regime",
            "candidate_only": not (cfg.publish_latest and promotion_eligible),
            "publish_requested": cfg.publish_latest,
            "promotion_eligible": promotion_eligible,
            # Publishing makes an artifact eligible for a later registry decision;
            # it never changes the registry by itself.
            "shadow_only": not (cfg.publish_latest and promotion_eligible),
            "order": {"p": cfg.p, "d": cfg.d, "q": cfg.q},
            "trend": cfg.trend,
            "refit_interval": cfg.refit_interval,
            "train_window": cfg.train_window,
            "min_train_size": cfg.min_train_size,
            "target_col": cfg.target_col,
            "target_expr": cfg.target_expr,
            "target_shift": cfg.target_shift,
            "target_alignment": "current_features_to_next_period_target"
            if cfg.target_shift == -1
            else "configured_target_shift",
            "group_col": cfg.group_col,
            "vol_window": cfg.vol_window,
            "features_path": cfg.features_path,
            "regimes_path": cfg.regimes_path,
            "canonical_metadata_path": canonical_metadata_path.as_posix(),
            "source_rows": int(source_rows),
            "target_ready_rows": int(target_ready_rows),
            "n_rows_used": int(len(regime_df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "time_bounds": _timestamp_bounds(regime_df, cfg.time_col),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_diagnostics": val_diag,
            "test_diagnostics": test_diag,
            "val_pred_nunique": int(val_pred_nunique),
            "test_pred_nunique": int(test_pred_nunique),
            "quality_gate": quality_gate,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "statsmodels": statsmodels.__version__,
            },
            "run_id": run_info.info.run_id,
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        meta_path = out_dir / "model_meta.json"
        out_dir.mkdir(parents=True, exist_ok=False)
        _write_json(meta_path, meta)
        mlflow.log_artifact(str(meta_path))
        mlflow.set_tag("expert_type", "arima")
        mlflow.set_tag("expert_name", cfg.model_name)
        mlflow.set_tag("expert_regime", cfg.regime)
        mlflow.set_tag("regime_filter_applied", "true")
        mlflow.set_tag(
            "candidate_only", str(not (cfg.publish_latest and promotion_eligible)).lower()
        )
        mlflow.set_tag("promotion_eligible", str(promotion_eligible).lower())
        mlflow.set_tag("run_id", run_info.info.run_id)

    if cfg.publish_latest and promotion_eligible:
        _write_json(canonical_metadata_path, meta)
    if cfg.update_legacy_pointer and promotion_eligible:
        _write_json(root / cfg.regime / "latest.arima.json", meta)

    print(f"Wrote validated ARIMA candidate: {out_dir}")
    if cfg.publish_latest and promotion_eligible:
        print(f"Published canonical ARIMA metadata: {canonical_metadata_path}")
    if cfg.update_legacy_pointer and promotion_eligible:
        print(f"Updated legacy ARIMA pointer: {root / cfg.regime / 'latest.arima.json'}")
    if cfg.publish_latest and not promotion_eligible:
        raise ValueError(
            "ARIMA candidate failed the zero-return test gate; no canonical or legacy pointer "
            "was changed. "
            f"model_rmse={quality_gate['test_rmse']:.8f} "
            f"zero_return_rmse={quality_gate['zero_return_test_rmse']:.8f}"
        )
    return out_dir


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

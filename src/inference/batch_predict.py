# src/inference/batch_predict.py
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.arima.model import ARIMA

from src.registry.registry import ACTIVE_FILE, RegistryError, load_active_model


@dataclass
class BatchPredictConfig:
    features_path: Path = Path("data/regimes/latest.parquet")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/predictions")
    runs_dir: Path = Path("data/runs")
    target_col: str = "target"
    active_file: Path | None = None
    inference_ts: int | None = None


def _latest_timestamp_dir(parent: Path) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"No timestamped dirs found in {parent}")
    return max(candidates, key=lambda p: int(p.name))


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def _walk_forward_arima_predict(
    y: pd.Series,
    *,
    order: tuple[int, int, int],
    trend: str,
    refit_interval: int,
    train_window: int | None,
    min_train_size: int,
    fallback: float = 0.0,
) -> pd.Series:
    """
    Leakage-safe 1-step predictions for each row i using only y[:i] history.
    Returns a Series aligned to y.index.

    NOTE: This predicts "next step" return aligned to the current row index,
    meaning pred[i] is the forecast for i (given history up to i-1).
    That matches your earlier evaluation logic using y_true = log_return_1_x.shift(-1)
    IF your model was trained with target_shift=-1 and you do NOT shift y here.
    """
    y_arr = y.to_numpy(dtype=np.float64)

    # Fill with fallback first, then overwrite with forecasts when available.
    preds: NDArray[np.float64] = np.full(
        shape=(len(y_arr),),
        fill_value=np.float64(fallback),
        dtype=np.float64,
    )

    history: list[float] = []
    last_fit_i = -(10**9)
    last_result: Any | None = None

    for i in range(len(y_arr)):
        yi = float(y_arr[i])

        # predict at i using history up to i-1
        if i == 0:
            history.append(yi)
            continue

        hist_used = history[-train_window:] if (train_window and train_window > 0) else history
        # Keep only finite history for fitting
        hist_used = [x for x in hist_used if np.isfinite(x)]

        if len(hist_used) < int(min_train_size):
            history.append(yi)
            continue

        need_refit = (i - last_fit_i) >= int(refit_interval) or last_result is None
        if need_refit:
            try:
                import warnings
                from statsmodels.tools.sm_exceptions import ConvergenceWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    last_result = ARIMA(
                        np.asarray(hist_used, dtype=np.float64), order=order, trend=trend
                    ).fit()
                    last_fit_i = i
            except Exception:
                last_result = None

        if last_result is not None:
            try:
                fc = last_result.forecast(steps=1)
                val = float(fc[0])
                if np.isfinite(val):
                    preds[i] = np.float64(val)
            except Exception:
                last_result = None

        history.append(yi)

    return pd.Series(preds, index=y.index, name="y_pred")


def discover_models(models_dir: Path) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []

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

    # experts: models/experts/<regime>/latest.joblib OR latest.json
    experts_root = models_dir / "experts"
    if experts_root.exists():
        for regime_dir in experts_root.iterdir():
            if not regime_dir.is_dir():
                continue

            # sklearn-style expert
            latest_model = regime_dir / "latest.joblib"
            if latest_model.exists():
                models.append(
                    {
                        "model_name": f"expert_lightgbm_{regime_dir.name}",
                        "model_source": "expert",
                        "model_path": str(latest_model),
                        "expert_kind": "sklearn",
                        "regime": regime_dir.name,
                    }
                )

            # ARIMA-style expert
            latest_json = regime_dir / "latest.json"
            if latest_json.exists():
                models.append(
                    {
                        "model_name": f"expert_arima_{regime_dir.name}",
                        "model_source": "expert",
                        "model_path": str(latest_json),
                        "expert_kind": "arima",
                        "regime": regime_dir.name,
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
            "No models discovered. Looked under: "
            f"{models_dir}/baseline, {models_dir}/experts, {models_dir}/pretrained"
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
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        name_list = list(names)
        missing = [c for c in name_list if c not in X.columns]
        if missing:
            raise RuntimeError(f"Missing required feature columns for model: {missing}")
        return X.loc[:, name_list]

    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is None:
        return X

    n = int(n_expected)
    if X.shape[1] < n:
        raise RuntimeError(
            f"X has {X.shape[1]} features but model expects {n}. "
            "Not enough features to run inference."
        )

    if X.shape[1] > n:
        return X.iloc[:, :n]

    return X


def _make_numeric_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Drop datetime columns and non-numeric columns.
    Return: (X_numeric, converted_cols, dropped_cols)
    """
    X = df.copy()
    converted: list[str] = []
    dropped: list[str] = []

    dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    if dt_cols:
        X = X.drop(columns=dt_cols)
        dropped.extend(dt_cols)

    non_numeric = [
        c
        for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    if non_numeric:
        X = X.drop(columns=non_numeric)
        dropped.extend(non_numeric)

    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    for c in bool_cols:
        X[c] = X[c].astype(int)

    return X, converted, dropped


def _safe_pred_value(v: Any) -> float:
    # unwrap numpy scalars
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass

    if isinstance(v, (pd.Timestamp, np.datetime64)):
        raise ValueError(f"prediction is datetime-like, refusing: {v!r}")

    f = float(v)

    if not np.isfinite(f):
        raise ValueError(f"prediction is not finite: {f!r}")

    if abs(f) > 1e6:
        raise ValueError(f"prediction magnitude insane, likely timestamp leak: {f!r}")

    return f


def _drop_nan_rows(X: pd.DataFrame, row_ids: pd.Index) -> tuple[pd.DataFrame, pd.Index]:
    """
    Drop rows with any NaN/inf values (Ridge can't handle NaNs).
    Returns filtered (X_clean, row_ids_clean) preserving alignment.
    """
    X2 = X.replace([float("inf"), float("-inf")], pd.NA)
    mask = X2.notna().all(axis=1)
    X_clean = X.loc[mask].copy()
    row_ids_clean = row_ids[mask.to_numpy()]
    return X_clean, row_ids_clean


def _get_active_fields(active_ref_dict: dict[str, Any] | None) -> tuple[str | None, str | None, str | None, str | None]:
    if active_ref_dict is None:
        return None, None, None, None
    active_model_type = (
        str(active_ref_dict.get("model_type")) if active_ref_dict.get("model_type") is not None else None
    )
    active_model_id = (
        str(active_ref_dict.get("model_id")) if active_ref_dict.get("model_id") is not None else None
    )
    active_model_version = (
        str(active_ref_dict.get("version")) if active_ref_dict.get("version") is not None else None
    )
    active_regime = (
        str(active_ref_dict.get("regime")) if active_ref_dict.get("regime") is not None else None
    )
    return active_model_type, active_model_id, active_model_version, active_regime


def run(config: BatchPredictConfig) -> Path:
    ts = int(config.inference_ts) if config.inference_ts is not None else int(time.time())

    if not config.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {config.features_path}")

    df = pd.read_parquet(config.features_path)

    # Keep full DF around for ARIMA (needs a target series, not feature matrix)
    df_full = df.copy()

    # Build X by dropping target if present
    X_raw = df.drop(columns=[config.target_col, "timestamp"], errors="ignore").copy()

    if not pd.api.types.is_integer_dtype(X_raw.index):
        X_raw = X_raw.reset_index(drop=True)

    row_ids = pd.Index(X_raw.index.astype(int))

    X, converted_cols, dropped_cols = _make_numeric_X(X_raw)
    nan_rows = int((~X.replace([float("inf"), float("-inf")], pd.NA).notna().all(axis=1)).sum())

    if X.shape[1] == 0:
        raise RuntimeError(
            "After preprocessing, X has 0 usable numeric feature columns. "
            "Your features parquet may be mostly timestamps/strings. "
            f"Dropped columns: {dropped_cols}"
        )

    models = discover_models(config.models_dir)

    # Load active model via registry (opt-in)
    active_load_error: str | None = None
    active_ref_dict: dict[str, Any] | None = None
    active_artifact_path: str | None = None
    active_model_obj: Any | None = None
    active_meta: dict[str, Any] | None = None

    if config.active_file is not None and config.active_file.exists():
        try:
            active_model_obj, active_meta, active_ref = load_active_model(active_file=config.active_file)
            active_ref_dict = {
                "model_type": active_ref.model_type,
                "regime": active_ref.regime,
                "model_id": active_ref.model_id,
                "version": active_ref.version,
                "artifact_path": active_ref.artifact_path.as_posix(),
                "metadata_path": active_ref.metadata_path.as_posix() if active_ref.metadata_path else None,
                "updated_at": active_ref.updated_at,
            }
            active_artifact_path = active_ref.artifact_path.as_posix()
        except RegistryError as e:
            active_load_error = repr(e)

    active_model_type, active_model_id, active_model_version, active_regime = _get_active_fields(active_ref_dict)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    # 1) Run active model first (if available) and label it explicitly
    if active_model_obj is not None and active_artifact_path is not None:
        try:
            # If active artifact is JSON, treat it as ARIMA meta and run series-based inference.
            if active_artifact_path.lower().endswith(".json"):
                meta = cast(dict[str, Any], active_model_obj)

                order = (
                    int(meta["order"]["p"]),
                    int(meta["order"]["d"]),
                    int(meta["order"]["q"]),
                )
                trend = str(meta.get("trend", "n"))
                refit_interval = int(meta.get("refit_interval", 50))
                train_window_raw = meta.get("train_window", None)
                train_window = int(train_window_raw) if train_window_raw not in (None, 0, "0") else None
                min_train_size = int(meta.get("min_train_size", 120))

                target_col = str(meta.get("target_col", "log_return_1_x"))

                if target_col not in df_full.columns:
                    raise RuntimeError(
                        f"Active ARIMA target_col '{target_col}' not in df columns: {sorted(df_full.columns)}"
                    )

                # IMPORTANT: do NOT shift here. This predicts next-step aligned to current row.
                y = df_full[target_col].astype(float)
                y = y.fillna(0.0)

                y_pred = _walk_forward_arima_predict(
                    y,
                    order=order,
                    trend=trend,
                    refit_interval=refit_interval,
                    train_window=train_window,
                    min_train_size=min_train_size,
                    fallback=0.0,
                )

                for rid, pred in zip(row_ids, y_pred.to_numpy(), strict=False):
                    rows.append(
                        {
                            "row_id": int(rid),
                            "model_name": "active",
                            "model_source": "registry",
                            "y_pred": _safe_pred_value(pred),
                            "inference_ts": ts,
                            "features_path": str(config.features_path),
                            "model_path": active_artifact_path,
                            "is_active": True,
                            "active_model_type": active_model_type,
                            "active_model_id": active_model_id,
                            "active_model_version": active_model_version,
                            "active_regime": active_regime,
                        }
                    )
            else:
                active_model = _unwrap_model(active_model_obj)

                X_aligned = _align_X_for_model(active_model, X)
                X_clean, row_ids_clean = _drop_nan_rows(X_aligned, row_ids)

                if len(X_clean) == 0:
                    raise RuntimeError("After dropping NaNs, no rows remain for active model inference.")

                preds = active_model.predict(X_clean)

                for rid, pred in zip(row_ids_clean, preds, strict=False):
                    rows.append(
                        {
                            "row_id": int(rid),
                            "model_name": "active",
                            "model_source": "registry",
                            "y_pred": _safe_pred_value(pred),
                            "inference_ts": ts,
                            "features_path": str(config.features_path),
                            "model_path": active_artifact_path,
                            "is_active": True,
                            "active_model_type": active_model_type,
                            "active_model_id": active_model_id,
                            "active_model_version": active_model_version,
                            "active_regime": active_regime,
                        }
                    )
        except Exception as e:
            failed.append(
                {
                    "model_name": "active",
                    "model_source": "registry",
                    "model_path": str(active_artifact_path),
                    "error": repr(e),
                }
            )

    # 2) Run shadow predictions for all discovered models
    for spec in models:
        model_path = Path(spec["model_path"])
        try:
            if spec.get("expert_kind") == "arima":
                meta = _load_json_dict(model_path)

                order = (
                    int(meta["order"]["p"]),
                    int(meta["order"]["d"]),
                    int(meta["order"]["q"]),
                )
                trend = str(meta.get("trend", "n"))
                refit_interval = int(meta.get("refit_interval", 50))
                train_window_raw = meta.get("train_window", None)
                train_window = int(train_window_raw) if train_window_raw not in (None, 0, "0") else None
                min_train_size = int(meta.get("min_train_size", 120))

                target_col = str(meta.get("target_col", "log_return_1_x"))
                if target_col not in df_full.columns:
                    raise RuntimeError(
                        f"ARIMA target_col '{target_col}' not in df columns: {sorted(df_full.columns)}"
                    )

                # IMPORTANT: do NOT shift here; series predictor already produces 1-step-ahead aligned to row i.
                y = df_full[target_col].astype(float).fillna(0.0)

                y_pred = _walk_forward_arima_predict(
                    y,
                    order=order,
                    trend=trend,
                    refit_interval=refit_interval,
                    train_window=train_window,
                    min_train_size=min_train_size,
                    fallback=0.0,
                )

                for rid, pred in zip(row_ids, y_pred.to_numpy(), strict=False):
                    rows.append(
                        {
                            "row_id": int(rid),
                            "model_name": spec["model_name"],
                            "model_source": spec["model_source"],
                            "y_pred": _safe_pred_value(pred),
                            "inference_ts": ts,
                            "features_path": str(config.features_path),
                            "model_path": str(model_path),
                            "is_active": False,
                            "active_model_type": active_model_type,
                            "active_model_id": active_model_id,
                            "active_model_version": active_model_version,
                            "active_regime": active_regime,
                        }
                    )
                continue

            loaded = joblib.load(model_path)
            model = _unwrap_model(loaded)

            X_aligned = _align_X_for_model(model, X)
            X_clean, row_ids_clean = _drop_nan_rows(X_aligned, row_ids)

            if len(X_clean) == 0:
                raise RuntimeError("After dropping NaNs, no rows remain for this model inference.")

            preds = model.predict(X_clean)

            for rid, pred in zip(row_ids_clean, preds, strict=False):
                rows.append(
                    {
                        "row_id": int(rid),
                        "model_name": spec["model_name"],
                        "model_source": spec["model_source"],
                        "y_pred": _safe_pred_value(pred),
                        "inference_ts": ts,
                        "features_path": str(config.features_path),
                        "model_path": str(model_path),
                        "is_active": False,
                        "active_model_type": active_model_type,
                        "active_model_id": active_model_id,
                        "active_model_version": active_model_version,
                        "active_regime": active_regime,
                    }
                )

        except Exception as e:
            failed.append(
                {
                    "model_name": spec.get("model_name", "unknown"),
                    "model_source": spec.get("model_source", "unknown"),
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
        .sort_values(["row_id", "model_source", "model_name"], kind="mergesort")
        .reset_index(drop=True)
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"predictions_{ts}.parquet"
    out_df.to_parquet(output_path, index=False)

    latest_path = config.output_dir / "latest.parquet"
    shutil.copyfile(output_path, latest_path)

    config.runs_dir.mkdir(parents=True, exist_ok=True)
    run_meta: dict[str, Any] = {
        "rows_with_nan_or_inf": nan_rows,
        "run_type": "batch_inference",
        "timestamp": ts,
        "features_path": str(config.features_path),
        "models_discovered": models,
        "output_path": str(output_path),
        "latest_path": str(latest_path),
        "num_prediction_rows": int(len(out_df)),
        "num_models_succeeded": int(out_df[["model_source", "model_name"]].drop_duplicates().shape[0]),
        "failed_models": failed,
        "feature_preprocessing": {
            "converted_datetime_cols": converted_cols,
            "dropped_non_numeric_cols": dropped_cols,
            "final_num_features": int(X.shape[1]),
        },
        "active_registry": {
            "active_file": str(config.active_file),
            "active_ref": active_ref_dict,
            "active_load_error": active_load_error,
        },
    }

    with open(config.runs_dir / f"run_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return output_path


def run_stage(
    *,
    features_path: Path,
    active_file: Path = ACTIVE_FILE,
    target_col: str = "target",
    models_dir: Path = Path("models"),
    output_dir: Path = Path("data/predictions"),
    runs_dir: Path = Path("data/runs"),
    inference_ts: int | None = None,
) -> Path:
    """
    Orchestration-friendly entrypoint.
    """
    cfg = BatchPredictConfig(
        features_path=features_path,
        models_dir=models_dir,
        output_dir=output_dir,
        runs_dir=runs_dir,
        target_col=target_col,
        active_file=active_file,
        inference_ts=inference_ts,
    )
    return run(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference across all models (active + shadow predictions)."
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
        help="Target column to drop if present. Default: target",
    )
    parser.add_argument(
        "--active-file",
        type=Path,
        default=ACTIVE_FILE,
        help="Path to registry active model yaml. Default: registry/active_model.yaml",
    )
    args = parser.parse_args()

    config = BatchPredictConfig(
        features_path=args.features_path,
        target_col=args.target_col,
        active_file=args.active_file,
    )
    out_path = run(config)
    print(f"Wrote predictions: {out_path}")
    print(f"Updated latest: {config.output_dir / 'latest.parquet'}")


if __name__ == "__main__":
    main()

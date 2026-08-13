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

from src.features.stationary import augment_pairwise_stationary_features
from src.registry.registry import ACTIVE_FILE, RegistryError, load_active_model


@dataclass
class BatchPredictConfig:
    features_path: Path = Path("data/regimes/latest.parquet")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/predictions")
    runs_dir: Path = Path("data/runs")
    target_col: str = "log_return_1_x"
    active_file: Path | None = None
    inference_ts: int | None = None
    output_name: str | None = None
    latest_name: str | None = "latest.parquet"
    run_meta_name: str | None = None
    record_features_path: str | None = None
    # The target is a decimal next-period log return.  A 20% single-period
    # move is deliberately generous, while still rejecting artifacts that
    # were trained against a different target or feature contract.
    max_abs_prediction: float = 0.20
    # Diversity is measured on the newest rows because a historical batch can
    # contain long out-of-distribution periods for a tree model.
    prediction_diversity_window: int = 252
    min_prediction_nunique: int = 3
    min_prediction_unique_fraction: float = 0.02
    # Production discovery accepts only explicit, validated publications.
    # Tests and one-off diagnostics can opt into legacy artifacts deliberately.
    require_published_model_contract: bool = True
    # Demo bootstrapping can intentionally run just its explicit active model
    # rather than every locally accumulated shadow artifact.
    include_discovered_models: bool = True


def _latest_timestamp_dir(parent: Path) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"No timestamped dirs found in {parent}")
    return max(candidates, key=lambda p: int(p.name))


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def _json_file_is_arima_meta(path: Path) -> bool:
    try:
        payload = _load_json_dict(path)
    except Exception:
        return False
    return str(payload.get("model_type", "")).lower() == "arima"


def _published_metadata(
    path: Path | None,
    *,
    expected_model_type: str | None = None,
) -> dict[str, Any] | None:
    """Return a v2, quality-gated production artifact metadata payload."""
    if path is None or not path.exists():
        return None
    try:
        payload = _load_json_dict(path)
    except Exception:
        return None
    if int(payload.get("artifact_contract_version", 0) or 0) < 2:
        return None
    if bool(payload.get("candidate_only", True)) or not bool(payload.get("promotion_eligible")):
        return None
    if expected_model_type is not None:
        actual_type = str(payload.get("model_type", "")).strip().lower()
        if actual_type != expected_model_type.lower():
            return None
    return payload


def _walk_forward_arima_predict(
    y: pd.Series,
    *,
    order: tuple[int, int, int],
    trend: str,
    refit_interval: int,
    train_window: int | None,
    min_train_size: int,
    history_regimes: pd.Series | None = None,
    training_regime: str | None = None,
    include_current_observation: bool = True,
    fallback: float = 0.0,
) -> pd.Series:
    """
    Leakage-safe forecasts aligned to the feature row.

    For a target shifted by -1, the return observed in row ``i`` is known at
    inference time and is the last observation available when forecasting the
    target for that row.  ``include_current_observation=True`` therefore
    avoids the prior one-bar lag.  When ``training_regime`` is supplied, the
    ARIMA history contains only observations from that training regime; the
    resulting expert can still be scored as a global shadow model without
    mixing regime histories.
    """
    y_arr = y.to_numpy(dtype=np.float64)
    regime_arr: np.ndarray[Any, Any] | None = None
    normalized_training_regime = (
        str(training_regime).strip().lower() if training_regime is not None else None
    )
    if history_regimes is not None:
        if len(history_regimes) != len(y_arr):
            raise ValueError("history_regimes must be aligned to the ARIMA target series")
        regime_arr = history_regimes.astype("string").str.strip().str.lower().to_numpy()
    if normalized_training_regime is not None and regime_arr is None:
        raise ValueError("training_regime requires an aligned regime column for ARIMA inference")

    # Fill with fallback first, then overwrite with forecasts when available.
    preds: NDArray[np.float64] = np.full(
        shape=(len(y_arr),),
        fill_value=np.float64(fallback),
        dtype=np.float64,
    )

    history: list[float] = []
    last_fit_history_size = -(10**9)
    last_result_history_size = 0
    last_result: Any | None = None

    for i in range(len(y_arr)):
        yi = float(y_arr[i])
        row_matches_training_regime = normalized_training_regime is None or (
            regime_arr is not None and str(regime_arr[i]) == normalized_training_regime
        )

        # A next-period target can use the current observed return.  Invalid
        # observations are never added to history.
        if include_current_observation and row_matches_training_regime and np.isfinite(yi):
            history.append(yi)

        hist_used = history[-train_window:] if (train_window and train_window > 0) else history
        hist_used = [x for x in hist_used if np.isfinite(x)]

        if len(hist_used) >= int(min_train_size):
            need_refit = (len(history) - last_fit_history_size) >= int(
                refit_interval
            ) or last_result is None
            if need_refit:
                try:
                    import warnings

                    from statsmodels.tools.sm_exceptions import ConvergenceWarning

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        last_result = ARIMA(
                            np.asarray(hist_used, dtype=np.float64), order=order, trend=trend
                        ).fit()
                        last_fit_history_size = len(history)
                        last_result_history_size = len(history)
                except Exception:
                    last_result = None

            # Forecast results are stateful.  Without appending intervening
            # realized observations, every row between refits receives the
            # same one-step forecast.  Keep the fitted state current while
            # preserving the bounded rolling-window reset at the next refit.
            if last_result is not None and len(history) > last_result_history_size:
                try:
                    additions = np.asarray(history[last_result_history_size:], dtype=np.float64)
                    last_result = last_result.append(additions, refit=False)
                    last_result_history_size = len(history)
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

        if not include_current_observation and row_matches_training_regime and np.isfinite(yi):
            history.append(yi)

    return pd.Series(preds, index=y.index, name="y_pred")


def discover_models(
    models_dir: Path,
    *,
    require_published_model_contract: bool = False,
) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []

    # baseline: the published pointer is canonical.  Falling back to the
    # newest versioned artifact preserves backwards compatibility for older
    # worktrees that predate latest.joblib.
    baseline_root = models_dir / "baseline"
    if baseline_root.exists():
        model_path = baseline_root / "latest.joblib"
        if not model_path.exists():
            latest_dir = _latest_timestamp_dir(baseline_root)
            model_path = latest_dir / "model.joblib"
        baseline_meta = baseline_root / "latest.json"
        if model_path.exists() and (
            not require_published_model_contract
            or _published_metadata(baseline_meta, expected_model_type="ridge") is not None
        ):
            models.append(
                {
                    "model_name": "baseline",
                    "model_source": "baseline",
                    "model_path": str(model_path),
                }
            )

    # Optional frozen global LightGBM control. It is intentionally separate
    # from regime experts so the daily experiment can compare routing against
    # a like-for-like non-regime model without reusing the mutable registry.
    global_root = models_dir / "global"
    global_model = global_root / "latest.joblib"
    global_meta = global_root / "latest.json"
    if global_model.exists() and (
        not require_published_model_contract
        or _published_metadata(global_meta, expected_model_type="lightgbm") is not None
    ):
        models.append(
            {
                "model_name": "global_lightgbm",
                "model_source": "global",
                "model_path": str(global_model),
                "expert_kind": "sklearn",
            }
        )

    # experts:
    #   - LightGBM artifact: models/experts/<regime>/latest.joblib
    #   - Canonical ARIMA metadata: models/experts/<regime>/arima/<model_id>.json
    #   - Legacy ARIMA pointer: models/experts/<regime>/latest.arima.json
    experts_root = models_dir / "experts"
    if experts_root.exists():
        for regime_dir in experts_root.iterdir():
            if not regime_dir.is_dir():
                continue

            # sklearn-style expert
            latest_model = regime_dir / "latest.joblib"
            lightgbm_meta = regime_dir / "latest.json"
            if latest_model.exists() and (
                not require_published_model_contract
                or _published_metadata(lightgbm_meta, expected_model_type="lightgbm") is not None
            ):
                models.append(
                    {
                        "model_name": f"expert_lightgbm_{regime_dir.name}",
                        "model_source": "expert",
                        "model_path": str(latest_model),
                        "expert_kind": "sklearn",
                        "regime": regime_dir.name,
                    }
                )

            # Multiple ARIMA experts may coexist per regime.  The directory
            # layout carries the published, immutable-by-name model metadata;
            # latest.arima.json remains a compatibility pointer only.
            seen_arima_model_ids: set[str] = set()
            canonical_arima_root = regime_dir / "arima"
            if canonical_arima_root.exists():
                for canonical_arima_meta_path in sorted(canonical_arima_root.glob("*.json")):
                    try:
                        arima_meta = _load_json_dict(canonical_arima_meta_path)
                    except Exception:
                        continue
                    if str(arima_meta.get("model_type", "")).lower() != "arima":
                        continue
                    if (
                        require_published_model_contract
                        and _published_metadata(
                            canonical_arima_meta_path, expected_model_type="arima"
                        )
                        is None
                    ):
                        continue
                    model_id_raw = arima_meta.get("model_id")
                    model_id = (
                        str(model_id_raw).strip()
                        if model_id_raw not in (None, "")
                        else f"expert_arima_{regime_dir.name}_{canonical_arima_meta_path.stem}"
                    )
                    seen_arima_model_ids.add(model_id)
                    models.append(
                        {
                            "model_name": model_id,
                            "model_source": "expert",
                            "model_path": str(canonical_arima_meta_path),
                            "expert_kind": "arima",
                            "regime": regime_dir.name,
                        }
                    )

            # Prefer the dedicated legacy filename, but retain even older
            # latest.json metadata when it is unambiguously ARIMA.
            latest_arima_json = regime_dir / "latest.arima.json"
            legacy_latest_json = regime_dir / "latest.json"
            arima_meta_path: Path | None = None
            if latest_arima_json.exists():
                arima_meta_path = latest_arima_json
            elif legacy_latest_json.exists() and _json_file_is_arima_meta(legacy_latest_json):
                arima_meta_path = legacy_latest_json

            if arima_meta_path is not None:
                arima_meta = _load_json_dict(arima_meta_path)
                if (
                    require_published_model_contract
                    and _published_metadata(arima_meta_path, expected_model_type="arima") is None
                ):
                    continue
                model_id_raw = arima_meta.get("model_id")
                model_id = (
                    str(model_id_raw).strip()
                    if model_id_raw not in (None, "")
                    else f"expert_arima_{regime_dir.name}"
                )
                if model_id in seen_arima_model_ids:
                    continue
                models.append(
                    {
                        "model_name": model_id,
                        "model_source": "expert",
                        "model_path": str(arima_meta_path),
                        "expert_kind": "arima",
                        "regime": regime_dir.name,
                    }
                )

    # pretrained: models/pretrained/*.joblib
    pretrained_root = models_dir / "pretrained"
    if pretrained_root.exists():
        for model_path in pretrained_root.glob("*.joblib"):
            metadata_path = model_path.with_suffix(".metadata.json")
            if (
                require_published_model_contract
                and _published_metadata(metadata_path, expected_model_type="ridge") is None
            ):
                continue
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


def _bundle_feature_columns(obj: Any) -> list[str] | None:
    """Return an explicit feature contract stored with a serialized bundle."""
    if not isinstance(obj, dict):
        return None

    raw = obj.get("feature_cols", obj.get("feature_columns"))
    if not isinstance(raw, list):
        return None

    cols = [str(column) for column in raw]
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("serialized model bundle has an empty or duplicate feature contract")
    return cols


def _align_X_for_model(
    model: Any,
    X: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Align X to what the model expects.

    Priority:
    1) Serialized bundle feature contract.
    2) ``feature_names_in_`` exposed by the estimator.
    3) Legacy positional alignment for old bare estimators only.
    """
    names = (
        feature_columns
        if feature_columns is not None
        else getattr(model, "feature_names_in_", None)
    )
    if names is not None:
        name_list = [str(name) for name in names]
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


def _safe_pred_value(v: Any, *, max_abs_prediction: float) -> float:
    # unwrap numpy scalars
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass

    if isinstance(v, pd.Timestamp | np.datetime64):
        raise ValueError(f"prediction is datetime-like, refusing: {v!r}")

    f = float(v)

    if not np.isfinite(f):
        raise ValueError(f"prediction is not finite: {f!r}")

    if abs(f) > float(max_abs_prediction):
        raise ValueError(
            "prediction exceeds the configured next-period log-return limit "
            f"({f!r}; limit={max_abs_prediction!r})"
        )

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


def _finite_nunique(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def _normalised_path(path: str | Path) -> Path:
    """Resolve a local artifact path without requiring it to exist."""
    return Path(path).resolve(strict=False)


def _same_artifact(left: str | Path, right: str | Path) -> bool:
    return _normalised_path(left) == _normalised_path(right)


def _validate_prediction_array(
    values: Any,
    *,
    config: BatchPredictConfig,
    model_name: str,
) -> np.ndarray:
    """Validate scale and recent diversity before an artifact is published."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"model returned no predictions: {model_name}")
    if not np.isfinite(arr).all():
        raise ValueError(f"model returned non-finite predictions: {model_name}")

    largest = float(np.max(np.abs(arr)))
    if largest > float(config.max_abs_prediction):
        raise ValueError(
            "prediction scale violates next-period log-return contract for "
            f"{model_name}: max_abs={largest:.6g}, limit={config.max_abs_prediction:.6g}"
        )

    window_size = min(max(int(config.prediction_diversity_window), 1), int(arr.size))
    recent = arr[-window_size:]
    unique = _finite_nunique(recent)
    required_unique = max(
        int(config.min_prediction_nunique),
        int(np.ceil(window_size * float(config.min_prediction_unique_fraction))),
    )
    if unique < required_unique:
        raise ValueError(
            "prediction diversity gate failed for "
            f"{model_name}: recent_nunique={unique}, required={required_unique}, "
            f"window={window_size}"
        )

    return cast(np.ndarray, arr)


def _arima_predictions_from_metadata(
    meta: dict[str, Any],
    *,
    df_full: pd.DataFrame,
) -> pd.Series:
    """Run a regime-specific ARIMA metadata artifact against an inference frame."""
    try:
        order = (
            int(meta["order"]["p"]),
            int(meta["order"]["d"]),
            int(meta["order"]["q"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ARIMA order metadata: {meta.get('order')!r}") from exc

    target_col = str(meta.get("target_col", "log_return_1_x"))
    if target_col not in df_full.columns:
        raise RuntimeError(
            f"ARIMA target_col '{target_col}' not in inference columns: {sorted(df_full.columns)}"
        )

    training_regime_raw = meta.get("training_regime", meta.get("regime"))
    training_regime = (
        str(training_regime_raw).strip().lower()
        if training_regime_raw not in (None, "", "none")
        else None
    )
    history_regimes: pd.Series | None = None
    if training_regime is not None:
        if "regime" not in df_full.columns:
            raise RuntimeError(
                "regime-specific ARIMA inference requires a 'regime' column; "
                "pass the regimes parquet rather than features-only input"
            )
        history_regimes = df_full["regime"]

    train_window_raw = meta.get("train_window", None)
    train_window = int(train_window_raw) if train_window_raw not in (None, 0, "0") else None
    target_shift = int(meta.get("target_shift", -1))

    return _walk_forward_arima_predict(
        pd.to_numeric(df_full[target_col], errors="coerce"),
        order=order,
        trend=str(meta.get("trend", "n")),
        refit_interval=int(meta.get("refit_interval", 50)),
        train_window=train_window,
        min_train_size=int(meta.get("min_train_size", 120)),
        history_regimes=history_regimes,
        training_regime=training_regime,
        include_current_observation=target_shift < 0,
        fallback=0.0,
    )


def _prediction_rows(
    *,
    row_ids: pd.Index,
    values: np.ndarray,
    spec: dict[str, str],
    is_active: bool,
    timestamp: int,
    features_path: str,
    active_model_type: str | None,
    active_model_id: str | None,
    active_model_version: str | None,
    active_regime: str | None,
    max_abs_prediction: float,
) -> list[dict[str, Any]]:
    if len(row_ids) != len(values):
        raise RuntimeError(
            f"prediction row alignment failed for {spec['model_name']}: "
            f"row_ids={len(row_ids)} predictions={len(values)}"
        )

    return [
        {
            "row_id": int(row_id),
            "model_name": spec["model_name"],
            "model_source": spec["model_source"],
            "y_pred": _safe_pred_value(value, max_abs_prediction=max_abs_prediction),
            "inference_ts": timestamp,
            "features_path": features_path,
            "model_path": spec["model_path"],
            "is_active": is_active,
            "active_model_type": active_model_type,
            "active_model_id": active_model_id,
            "active_model_version": active_model_version,
            "active_regime": active_regime,
        }
        for row_id, value in zip(row_ids, values, strict=True)
    ]


def _get_active_fields(
    active_ref_dict: dict[str, Any] | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if active_ref_dict is None:
        return None, None, None, None
    active_model_type = (
        str(active_ref_dict.get("model_type"))
        if active_ref_dict.get("model_type") is not None
        else None
    )
    active_model_id = (
        str(active_ref_dict.get("model_id"))
        if active_ref_dict.get("model_id") is not None
        else None
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

    recorded_features_path = config.record_features_path or str(config.features_path)
    # Prediction row ids are positional.  Reset once so the feature frame,
    # ARIMA target series, and every emitted model share the same contract.
    df = pd.read_parquet(config.features_path).reset_index(drop=True)

    # Keep full DF around for ARIMA (needs a target series, not feature matrix)
    df_full = df.copy()

    # Keep the observed return available while deriving stationary features.
    # It is then removed from the tabular model inputs below, so a next-period
    # return model cannot consume its own observed return as a direct feature.
    X_raw = df.drop(columns=["timestamp"], errors="ignore").copy()

    row_ids = pd.Index(X_raw.index.astype(int))

    X, converted_cols, dropped_cols = _make_numeric_X(X_raw)
    X, added_stationary_cols = augment_pairwise_stationary_features(X)
    X = X.drop(columns=[config.target_col], errors="ignore")
    nan_rows = int((~X.replace([float("inf"), float("-inf")], pd.NA).notna().all(axis=1)).sum())

    if X.shape[1] == 0:
        raise RuntimeError(
            "After preprocessing, X has 0 usable numeric feature columns. "
            "Your features parquet may be mostly timestamps/strings. "
            f"Dropped columns: {dropped_cols}"
        )

    eligible_mask = X.replace([float("inf"), float("-inf")], pd.NA).notna().all(axis=1)
    eligible_row_ids = row_ids[eligible_mask.to_numpy()]
    if len(eligible_row_ids) == 0:
        raise RuntimeError("After applying the shared finite-feature contract, no rows remain.")

    models = (
        discover_models(
            config.models_dir,
            require_published_model_contract=config.require_published_model_contract,
        )
        if config.include_discovered_models
        else []
    )

    # Load active model via registry (opt-in)
    active_load_error: str | None = None
    active_ref_dict: dict[str, Any] | None = None
    active_artifact_path: str | None = None
    active_registry_requested = config.active_file is not None and config.active_file.exists()

    if config.active_file is not None and config.active_file.exists():
        try:
            _active_model_obj, _active_meta, active_ref = load_active_model(
                active_file=config.active_file
            )
            if config.require_published_model_contract:
                expected_type = (
                    "arima"
                    if active_ref.artifact_path.suffix.lower() == ".json"
                    else "lightgbm"
                    if active_ref.model_type == "expert"
                    else "ridge"
                )
                if (
                    _published_metadata(active_ref.metadata_path, expected_model_type=expected_type)
                    is None
                ):
                    raise RegistryError(
                        "active registry artifact is not a published v2 model that passed its "
                        f"quality gate: {active_ref.model_id}"
                    )
            active_ref_dict = {
                "model_type": active_ref.model_type,
                "regime": active_ref.regime,
                "model_id": active_ref.model_id,
                "version": active_ref.version,
                "artifact_path": active_ref.artifact_path.as_posix(),
                "metadata_path": active_ref.metadata_path.as_posix()
                if active_ref.metadata_path
                else None,
                "updated_at": active_ref.updated_at,
            }
            active_artifact_path = active_ref.artifact_path.as_posix()
        except (RegistryError, OSError, ValueError, TypeError) as e:
            active_load_error = repr(e)

    active_model_type, active_model_id, active_model_version, active_regime = _get_active_fields(
        active_ref_dict
    )
    active_model_source = (
        active_model_type if active_model_type in {"baseline", "expert", "pretrained"} else "expert"
    )

    # A configured global active pointer is a production safety contract.  Do
    # not turn an active load failure into a successful shadows-only run.
    if active_registry_requested and active_load_error is not None:
        raise RuntimeError(f"Active registry model could not be loaded: {active_load_error}")

    # The active model is represented once using its canonical model id.  It
    # is not also emitted as an independent "active" model, which previously
    # made active/shadow agreement look like duplicate-model behavior.
    resolved_models: list[dict[str, str]] = []
    active_was_discovered = False
    for raw_spec in models:
        spec = dict(raw_spec)
        is_active = active_artifact_path is not None and _same_artifact(
            spec["model_path"], active_artifact_path
        )
        if is_active:
            active_was_discovered = True
            if active_model_id:
                spec["model_name"] = active_model_id
            spec["model_source"] = active_model_source
            spec["is_active"] = "true"
        else:
            spec["is_active"] = "false"
        resolved_models.append(spec)

    if active_artifact_path is not None and not active_was_discovered:
        active_path = Path(active_artifact_path)
        resolved_models.append(
            {
                "model_name": active_model_id or "active_model",
                "model_source": active_model_source,
                "model_path": active_artifact_path,
                "expert_kind": "arima" if active_path.suffix.lower() == ".json" else "sklearn",
                "regime": active_regime or "",
                "is_active": "true",
            }
        )

    resolved_models.sort(key=lambda m: (m["model_source"], m["model_name"], m["model_path"]))

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    executed: list[dict[str, Any]] = []

    # Run the global active model plus each distinct shadow artifact.  An
    # active artifact remains global by registry choice; no regime-based
    # auto-selection happens here.
    for spec in resolved_models:
        model_path = Path(spec["model_path"])
        is_active = spec.get("is_active") == "true"
        try:
            if spec.get("expert_kind") == "arima":
                meta = _load_json_dict(model_path)
                all_predictions = _arima_predictions_from_metadata(meta, df_full=df_full)
                values = all_predictions.iloc[eligible_row_ids.to_numpy()].to_numpy(dtype=float)
            else:
                loaded = joblib.load(model_path)
                model = _unwrap_model(loaded)
                X_aligned = _align_X_for_model(
                    model,
                    X,
                    feature_columns=_bundle_feature_columns(loaded),
                )
                values = np.asarray(model.predict(X_aligned.loc[eligible_mask]), dtype=float)

            values = _validate_prediction_array(
                values, config=config, model_name=spec["model_name"]
            )
            rows.extend(
                _prediction_rows(
                    row_ids=eligible_row_ids,
                    values=values,
                    spec=spec,
                    is_active=is_active,
                    timestamp=ts,
                    features_path=recorded_features_path,
                    active_model_type=active_model_type,
                    active_model_id=active_model_id,
                    active_model_version=active_model_version,
                    active_regime=active_regime,
                    max_abs_prediction=config.max_abs_prediction,
                )
            )
            executed.append(
                {
                    "model_name": spec["model_name"],
                    "model_source": spec["model_source"],
                    "model_path": str(model_path),
                    "is_active": is_active,
                    "prediction_rows": int(len(values)),
                    "recent_prediction_nunique": _finite_nunique(
                        values[-min(len(values), config.prediction_diversity_window) :]
                    ),
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

    if active_ref_dict is not None and not any(bool(record["is_active"]) for record in executed):
        active_failure = next(
            (failure for failure in failed if failure.get("model_name") == active_model_id),
            None,
        )
        detail = active_failure.get("error") if active_failure is not None else "not discovered"
        raise RuntimeError(
            "Active registry model failed inference safety checks; refusing to publish a "
            f"shadows-only prediction run. model_id={active_model_id!r} detail={detail}"
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
    output_name = config.output_name or f"predictions_{ts}.parquet"
    output_path = config.output_dir / output_name
    out_df.to_parquet(output_path, index=False)

    latest_path: Path | None = None
    if config.latest_name:
        latest_path = config.output_dir / str(config.latest_name)
        shutil.copyfile(output_path, latest_path)

    config.runs_dir.mkdir(parents=True, exist_ok=True)
    run_meta: dict[str, Any] = {
        "rows_with_nan_or_inf": nan_rows,
        "run_type": "batch_inference",
        "timestamp": ts,
        "features_path": recorded_features_path,
        "models_discovered": resolved_models,
        "models_executed": executed,
        "output_path": str(output_path),
        "latest_path": str(latest_path) if latest_path is not None else None,
        "num_prediction_rows": int(len(out_df)),
        "num_models_succeeded": int(len(executed)),
        "failed_models": failed,
        "feature_preprocessing": {
            "converted_datetime_cols": converted_cols,
            "dropped_non_numeric_cols": dropped_cols,
            "added_stationary_cols": added_stationary_cols,
            "final_num_features": int(X.shape[1]),
        },
        "active_registry": {
            "active_file": str(config.active_file),
            "active_ref": active_ref_dict,
            "active_load_error": active_load_error,
        },
    }

    run_meta_name = config.run_meta_name or f"run_{ts}.json"
    with open(config.runs_dir / run_meta_name, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return output_path


def run_stage(
    *,
    features_path: Path,
    active_file: Path = ACTIVE_FILE,
    target_col: str = "log_return_1_x",
    models_dir: Path = Path("models"),
    output_dir: Path = Path("data/predictions"),
    runs_dir: Path = Path("data/runs"),
    inference_ts: int | None = None,
    output_name: str | None = None,
    latest_name: str | None = "latest.parquet",
    run_meta_name: str | None = None,
    record_features_path: str | None = None,
    include_discovered_models: bool = True,
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
        output_name=output_name,
        latest_name=latest_name,
        run_meta_name=run_meta_name,
        record_features_path=record_features_path,
        include_discovered_models=include_discovered_models,
    )
    return run(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference across all models (active + shadow predictions)."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/regimes/latest.parquet"),
        help=(
            "Path to the regimes parquet used for inference. Regime-specific ARIMA experts "
            "require its regime column. Default: data/regimes/latest.parquet"
        ),
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="log_return_1_x",
        help="Observed return column to drop from tabular features. Default: log_return_1_x",
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

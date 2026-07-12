from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.features.stationary import augment_pairwise_stationary_features, summarize_feature_ranges
from src.registry.registry import ActiveModelRef


@dataclass(frozen=True)
class PromotionGuardConfig:
    """
    Conservative gate for promoting a candidate model into the active pointer.

    The intent is to reject obviously broken or stale models without hiding a healthy
    model behind policy thresholds.
    """

    # Applied to the newest part of a batch so historical out-of-distribution
    # rows do not conceal a collapsed current model.
    prediction_diversity_window: int = 252
    min_prediction_nunique: int = 3
    min_prediction_unique_fraction: float = 0.02
    min_prediction_std: float = 1e-7
    max_abs_prediction: float = 0.20
    feature_span_slack_ratio: float = 0.50
    feature_std_slack_multiplier: float = 5.0
    feature_slack_floor: float = 1e-6


@dataclass(frozen=True)
class PromotionGuardResult:
    allowed: bool
    reason: str
    candidate_model_name: str | None = None
    prediction_nunique: int | None = None
    prediction_std: float | None = None
    prediction_window_rows: int | None = None
    required_prediction_nunique: int | None = None
    current_features_path: str | None = None
    metadata_path: str | None = None
    feature_violations: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported feature file type: {path.suffix}")


def _candidate_name_variants(candidate_model_name: str) -> list[str]:
    """
    Try the exact candidate name first, then a couple of backwards-compatible aliases.
    """
    name = str(candidate_model_name).strip()
    variants = [name]

    regime: str | None = None
    if name.startswith("expert_lightgbm_"):
        regime = name.split("_")[-1]
        variants.append(f"expert_{regime}")
        variants.append(f"expert_arima_{regime}")
    elif name.startswith("expert_arima_"):
        regime = name.split("_")[-1]
        variants.append(f"expert_{regime}")
        variants.append(f"expert_lightgbm_{regime}")
    elif name.startswith("expert_"):
        parts = name.split("_")
        if len(parts) >= 2:
            regime = parts[-1]
            if regime in {"bullish", "bearish", "sideways"}:
                variants.append(f"expert_lightgbm_{regime}")
                variants.append(f"expert_arima_{regime}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _candidate_rows(preds: pd.DataFrame, candidate_model_name: str) -> tuple[pd.DataFrame, str | None]:
    if "model_name" not in preds.columns:
        raise ValueError("predictions parquet missing required column: model_name")
    if "y_pred" not in preds.columns:
        raise ValueError("predictions parquet missing required column: y_pred")

    for name in _candidate_name_variants(candidate_model_name):
        rows = preds[preds["model_name"].astype(str) == name].copy()
        if not rows.empty:
            return rows, name
    return preds.iloc[0:0].copy(), None


def _prediction_dispersion(
    rows: pd.DataFrame, cfg: PromotionGuardConfig
) -> tuple[int, float, int, int, float]:
    ordered = rows.sort_values("row_id", kind="mergesort") if "row_id" in rows.columns else rows
    values = pd.to_numeric(ordered["y_pred"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, float("nan"), 0, 0, float("nan")

    window_size = min(max(int(cfg.prediction_diversity_window), 1), int(finite.size))
    recent = finite[-window_size:]
    required_unique = max(
        int(cfg.min_prediction_nunique),
        int(np.ceil(window_size * float(cfg.min_prediction_unique_fraction))),
    )
    return (
        int(np.unique(recent).size),
        float(np.std(recent, ddof=0)),
        window_size,
        required_unique,
        float(np.max(np.abs(recent))),
    )


def _load_current_features(
    *,
    predictions_rows: pd.DataFrame,
    current_features_path: Path | None,
) -> tuple[pd.DataFrame, Path]:
    if current_features_path is None:
        if "features_path" not in predictions_rows.columns:
            raise ValueError("predictions parquet missing required column: features_path")
        raw_value = predictions_rows.iloc[0]["features_path"]
        if pd.isna(raw_value):
            raise ValueError("predictions parquet has blank features_path for candidate rows")
        raw_path = str(raw_value).strip()
        if not raw_path or raw_path.lower() == "nan":
            raise ValueError("predictions parquet has blank features_path for candidate rows")
        current_features_path = Path(raw_path)

    df = _read_frame(current_features_path)
    df, _ = augment_pairwise_stationary_features(df)
    return df, current_features_path


def _compare_feature_envelopes(
    *,
    runtime_features: pd.DataFrame,
    training_feature_stats: dict[str, Any],
    cfg: PromotionGuardConfig,
) -> list[str]:
    violations: list[str] = []
    if not training_feature_stats:
        violations.append("training feature stats missing")
        return violations

    runtime_stats = summarize_feature_ranges(runtime_features, list(training_feature_stats.keys()))

    for feature_name, train_stats in training_feature_stats.items():
        if feature_name not in runtime_stats:
            violations.append(f"{feature_name}: missing in runtime features")
            continue

        cur_stats = runtime_stats[feature_name]
        if pd.isna(cur_stats["min"]) or pd.isna(cur_stats["max"]):
            violations.append(f"{feature_name}: runtime values are all missing or non-finite")
            continue

        try:
            train_min = float(train_stats["min"])
            train_max = float(train_stats["max"])
            train_std = float(train_stats.get("std", 0.0) or 0.0)
        except Exception:
            violations.append(f"{feature_name}: invalid training range stats")
            continue

        if not np.isfinite(train_min) or not np.isfinite(train_max) or not np.isfinite(train_std):
            violations.append(f"{feature_name}: invalid training range stats")
            continue

        train_span = max(train_max - train_min, 0.0)
        slack = max(
            train_span * cfg.feature_span_slack_ratio,
            train_std * cfg.feature_std_slack_multiplier,
            cfg.feature_slack_floor,
        )

        cur_min = float(cur_stats["min"])
        cur_max = float(cur_stats["max"])
        if cur_min < train_min - slack:
            violations.append(
                f"{feature_name}: runtime min {cur_min:.6g} below training envelope "
                f"{train_min:.6g} - {slack:.6g}"
            )
        if cur_max > train_max + slack:
            violations.append(
                f"{feature_name}: runtime max {cur_max:.6g} above training envelope "
                f"{train_max:.6g} + {slack:.6g}"
            )

    return violations


def evaluate_candidate_promotion_guard(
    *,
    candidate_ref: ActiveModelRef,
    predictions_path: Path,
    current_features_path: Path | None = None,
    candidate_model_name: str | None = None,
    cfg: PromotionGuardConfig | None = None,
) -> PromotionGuardResult:
    """
    Inspect the current candidate predictions and, for expert models, the current
    feature envelope before allowing promotion.
    """
    cfg = cfg or PromotionGuardConfig()
    candidate_name = str(candidate_model_name or candidate_ref.model_id)

    if not predictions_path.exists():
        return PromotionGuardResult(
            allowed=False,
            reason=f"predictions not found: {predictions_path}",
            candidate_model_name=candidate_name,
        )

    try:
        preds = pd.read_parquet(predictions_path)
        rows, matched_name = _candidate_rows(preds, candidate_name)
        if rows.empty:
            return PromotionGuardResult(
                allowed=False,
                reason=f"candidate predictions not found for model_name={candidate_name}",
                candidate_model_name=candidate_name,
            )

        (
            prediction_nunique,
            prediction_std,
            prediction_window_rows,
            required_prediction_nunique,
            max_abs_prediction,
        ) = _prediction_dispersion(rows, cfg)
        if prediction_nunique < required_prediction_nunique or not np.isfinite(prediction_std):
            return PromotionGuardResult(
                allowed=False,
                reason=(
                    f"candidate predictions are degenerate for model_name={matched_name or candidate_name} "
                    f"(recent_nunique={prediction_nunique}, required={required_prediction_nunique}, "
                    f"std={prediction_std})"
                ),
                candidate_model_name=matched_name or candidate_name,
                prediction_nunique=prediction_nunique,
                prediction_std=prediction_std,
                prediction_window_rows=prediction_window_rows,
                required_prediction_nunique=required_prediction_nunique,
            )
        if prediction_std <= cfg.min_prediction_std:
            return PromotionGuardResult(
                allowed=False,
                reason=(
                    f"candidate predictions are effectively flat for model_name={matched_name or candidate_name} "
                    f"(nunique={prediction_nunique}, std={prediction_std:.6g})"
                ),
                candidate_model_name=matched_name or candidate_name,
                prediction_nunique=prediction_nunique,
                prediction_std=prediction_std,
                prediction_window_rows=prediction_window_rows,
                required_prediction_nunique=required_prediction_nunique,
            )
        if max_abs_prediction > cfg.max_abs_prediction:
            return PromotionGuardResult(
                allowed=False,
                reason=(
                    "candidate prediction scale exceeds next-period return contract "
                    f"(max_abs={max_abs_prediction:.6g}, limit={cfg.max_abs_prediction:.6g})"
                ),
                candidate_model_name=matched_name or candidate_name,
                prediction_nunique=prediction_nunique,
                prediction_std=prediction_std,
                prediction_window_rows=prediction_window_rows,
                required_prediction_nunique=required_prediction_nunique,
            )
    except Exception as exc:
        return PromotionGuardResult(
            allowed=False,
            reason=f"promotion guard failed while inspecting predictions: {exc}",
            candidate_model_name=candidate_name,
        )

    # rows is guaranteed non-empty here.
    rows = rows.copy()

    metadata_path = candidate_ref.metadata_path
    is_expert = str(candidate_ref.model_type).strip().lower() == "expert"
    if metadata_path is None and is_expert:
        return PromotionGuardResult(
            allowed=False,
            reason=f"expert candidate is missing metadata_path: {candidate_ref.model_id}",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
        )
    if metadata_path is not None and not metadata_path.exists():
        return PromotionGuardResult(
            allowed=False,
            reason=f"expert candidate metadata not found: {metadata_path}",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix(),
        )

    metadata = _load_json(metadata_path) if metadata_path is not None else {}
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict) and not bool(quality_gate.get("promotion_eligible")):
        return PromotionGuardResult(
            allowed=False,
            reason=(
                "candidate failed its saved validation quality gate: "
                f"{quality_gate.get('reason', quality_gate.get('reasons', 'unknown reason'))}"
            ),
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix() if metadata_path is not None else None,
        )

    # ARIMA has no tabular feature envelope.  Its saved validation gate plus
    # the scale and diversity checks above are its applicable safety contract.
    if str(metadata.get("model_type", "")).strip().lower() == "arima":
        return PromotionGuardResult(
            allowed=True,
            reason="ARIMA candidate passed saved quality, scale, and diversity gates",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix() if metadata_path is not None else None,
        )

    # Baseline and pretrained bundles are protected by their serialized
    # feature contracts rather than a LightGBM feature envelope.
    if not is_expert:
        return PromotionGuardResult(
            allowed=True,
            reason="candidate passed saved quality, scale, and diversity gates",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix() if metadata_path is not None else None,
        )

    training_feature_stats = metadata.get("feature_range_stats")
    if not isinstance(training_feature_stats, dict) or not training_feature_stats:
        return PromotionGuardResult(
            allowed=False,
            reason=f"expert candidate metadata missing feature_range_stats: {metadata_path}",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix(),
        )

    try:
        runtime_features, resolved_features_path = _load_current_features(
            predictions_rows=rows,
            current_features_path=current_features_path,
        )
    except Exception as exc:
        return PromotionGuardResult(
            allowed=False,
            reason=f"failed to load current features for guard: {exc}",
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            prediction_window_rows=prediction_window_rows,
            required_prediction_nunique=required_prediction_nunique,
            metadata_path=metadata_path.as_posix(),
        )

    feature_violations = _compare_feature_envelopes(
        runtime_features=runtime_features,
        training_feature_stats=training_feature_stats,
        cfg=cfg,
    )
    if feature_violations:
        return PromotionGuardResult(
            allowed=False,
            reason="; ".join(feature_violations[:4]),
            candidate_model_name=matched_name or candidate_name,
            prediction_nunique=prediction_nunique,
            prediction_std=prediction_std,
            current_features_path=resolved_features_path.as_posix(),
            metadata_path=metadata_path.as_posix(),
            feature_violations=feature_violations,
        )

    return PromotionGuardResult(
        allowed=True,
        reason="candidate predictions are non-degenerate and feature envelope is acceptable",
        candidate_model_name=matched_name or candidate_name,
        prediction_nunique=prediction_nunique,
        prediction_std=prediction_std,
        prediction_window_rows=prediction_window_rows,
        required_prediction_nunique=required_prediction_nunique,
        current_features_path=resolved_features_path.as_posix(),
        metadata_path=metadata_path.as_posix(),
    )

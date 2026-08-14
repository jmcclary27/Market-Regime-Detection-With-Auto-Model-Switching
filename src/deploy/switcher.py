# src/deploy/switcher.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.deploy.history import append_deployment_event
from src.models.promotion_guard import evaluate_candidate_promotion_guard
from src.registry.registry import ActiveModelRef, write_active

# ---------- Config ----------


@dataclass(frozen=True)
class SwitchConfig:
    """
    Canary switcher config (v0: count-based only).

    metric_name:
      - Your scorecards use: "rmse", "mae"
      - Lower is better for both.
    """

    window_type: str = "count"
    window_value: int = 100
    metric_name: str = "rmse"

    # Candidate must be better than active by at least promote_margin to promote
    promote_margin: float = 0.0

    # Candidate is considered clearly worse if it is worse than active by rollback_margin
    rollback_margin: float = 0.0

    # Whether to actually update registry/active_model.yaml when promoting
    update_registry_on_promote: bool = True


def append_event(events_path: Path, event: dict[str, Any]) -> None:
    append_deployment_event(events_path, event)


def write_active_model_yaml(
    registry_path: Path,
    model_id: str,
    *,
    event_context: dict[str, Any] | None = None,
) -> bool:
    def _infer_regime(candidate_model_id: str) -> str | None:
        for regime in ("bullish", "bearish", "sideways"):
            if candidate_model_id == f"expert_{regime}" or candidate_model_id.endswith(
                f"_{regime}"
            ):
                return regime
            if candidate_model_id.startswith(f"expert_arima_{regime}_"):
                return regime
            if candidate_model_id.startswith(f"expert_lightgbm_{regime}_"):
                return regime
        return None

    def _arima_metadata_path(candidate_model_id: str, regime: str) -> Path:
        canonical = Path(f"models/experts/{regime}/arima/{candidate_model_id}.json")
        legacy = Path(f"models/experts/{regime}/latest.arima.json")
        if canonical.exists() or candidate_model_id != f"expert_arima_{regime}":
            return canonical
        return legacy

    regime = _infer_regime(model_id)
    if model_id == "baseline":
        ref = ActiveModelRef(
            model_type="baseline",
            model_id=model_id,
            version="latest",
            artifact_path=Path("models/baseline/latest.joblib"),
            regime=None,
            metadata_path=Path("models/baseline/latest.json"),
        )
    elif model_id.startswith("expert_arima_") and regime is not None:
        arima_path = _arima_metadata_path(model_id, regime)
        ref = ActiveModelRef(
            model_type="expert",
            model_id=model_id,
            version="0",
            artifact_path=arima_path,
            regime=regime,
            metadata_path=arima_path,
        )
    elif (
        model_id.startswith("expert_lightgbm_")
        or model_id in {"expert_bullish", "expert_bearish", "expert_sideways"}
    ) and regime is not None:
        ref = ActiveModelRef(
            model_type="expert",
            model_id=model_id,
            version="0",
            artifact_path=Path(f"models/experts/{regime}/latest.joblib"),
            regime=regime,
            metadata_path=Path(f"models/experts/{regime}/latest.json"),
        )
    else:
        ref = ActiveModelRef(
            model_type="pretrained",
            model_id=model_id,
            version="0",
            artifact_path=Path(f"models/pretrained/{model_id}.joblib"),
            regime=None,
            metadata_path=None,
        )

    return bool(write_active(ref, active_file=registry_path, event_context=event_context))


def infer_model_id_col(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    candidates = ["model_name", "model_id", "model", "model_key", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_pretrained_candidate_for_regime(
    *,
    project_root: Path,
    data_dir: Path,
    regime: str,
    metric_name: str,
) -> str | None:
    """
    Choose best pretrained model for a given regime from models/pretrained.

    Convention:
      models/pretrained/{regime}__*.joblib

    Selection:
      1) If scorecard has metrics for multiple candidates, pick the lowest metric (lower is better).
      2) Otherwise, pick the newest artifact by mtime.

    Returns:
      model_id string (which must match scorecard model_id/model_name and registry model_id usage)
      This is the filename stem (without .joblib), same as batch_predict.discover_models() uses.
    """
    pretrained_dir = project_root / "models" / "pretrained"
    if not pretrained_dir.exists():
        return None

    candidates = sorted(pretrained_dir.glob(f"{regime}__*.joblib"))
    if not candidates:
        return None

    # Candidate ids are filename stems, because discover_models() uses model_path.stem
    candidate_ids = [p.stem for p in candidates]

    # Try scorecard-based selection (best metric)
    scorecards_dir = data_dir / "scorecards"
    scorecard = load_latest_scorecard(scorecards_dir)
    if scorecard is not None:
        best_id: str | None = None
        best_val: float | None = None

        for cid in candidate_ids:
            val = extract_metric(scorecard, model_id=cid, metric_name=metric_name)
            if val is None:
                continue
            if best_val is None or val < best_val:
                best_val = val
                best_id = cid

        if best_id is not None:
            return best_id

    # Fallback: newest file
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.stem


def load_latest_scorecard(scorecards_dir: Path) -> pd.DataFrame | None:
    path = scorecards_dir / "latest.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _prefer_overall_rows(df: pd.DataFrame) -> pd.DataFrame:
    sub = df
    if "scope" in sub.columns:
        sub = sub[sub["scope"] == "overall"]
    if "regime" in sub.columns:
        sub = sub[sub["regime"].isna()]
    return sub


def extract_metric(
    df: pd.DataFrame,
    *,
    model_id: str,
    metric_name: str,
) -> float | None:
    id_col = infer_model_id_col(df)
    if id_col is None:
        return None
    if metric_name not in df.columns:
        return None

    sub = _prefer_overall_rows(df)

    rows = sub[sub[id_col] == model_id]
    if rows.empty:
        return None

    val = rows.iloc[0][metric_name]
    if pd.isna(val):
        return None
    return float(val)


def extract_n(df: pd.DataFrame, *, model_id: str) -> int | None:
    id_col = infer_model_id_col(df)
    if id_col is None or "n" not in df.columns:
        return None

    sub = _prefer_overall_rows(df)

    rows = sub[sub[id_col] == model_id]
    if rows.empty:
        return None

    val = rows.iloc[0]["n"]
    if pd.isna(val):
        return None
    return int(val)


def decide(
    *,
    active_metric: float,
    candidate_metric: float,
    promote_margin: float,
    rollback_margin: float,
) -> tuple[str, str]:
    """
    Lower is better.
    Returns (decision, reason).
    decision ∈ {"promote", "rollback", "hold"}
    """
    # candidate better by margin => promote
    if candidate_metric <= active_metric - promote_margin:
        return "promote", "candidate_better_than_active"

    # candidate worse by rollback margin => rollback
    if candidate_metric >= active_metric + rollback_margin:
        return "rollback", "candidate_worse_than_active"

    return "hold", "within_margins"


def load_latest_regime(regimes_path: Path) -> str | None:
    """
    Load the most recent regime label from data/regimes/latest.parquet.

    Returns:
      - "bullish" | "bearish" | "sideways" | "unknown" | None
    """
    if not regimes_path.exists():
        return None

    df = pd.read_parquet(regimes_path, columns=["timestamp", "regime"])
    if df.empty or "regime" not in df.columns:
        return None

    last = df.iloc[-1]["regime"]
    if pd.isna(last):
        return None

    return str(last)


def regime_to_candidate_model_ids(regime: str) -> list[str]:
    """
    Return ALL candidate model_ids that are eligible
    for the given regime.

    This allows the switcher to evaluate multiple experts
    (LightGBM, ARIMA, Ridge, etc.) per regime.
    """

    base_candidates = [
        "baseline",
        "expert_bullish_ridge_v0",
    ]

    regime_candidates = {
        "bullish": [
            "expert_lightgbm_bullish",
            "expert_arima_bullish",
        ],
        "bearish": [
            "expert_lightgbm_bearish",
            "expert_arima_bearish",
        ],
        "sideways": [
            "expert_lightgbm_sideways",
            "expert_arima_sideways",
        ],
    }

    return base_candidates + regime_candidates.get(regime, [])


# ---------- Switcher ----------


def run_switcher(
    *,
    data_dir: Path,
    config: SwitchConfig,
    active_model_id: str = "baseline",
    candidate_id: str | None = None,
    # Backwards-compatible alias for tests / older call sites.
    candidate_model_id: str | None = None,
) -> None:
    """
    Step 3 behavior:
    - Load latest regime label from data/regimes/latest.parquet
    - Map regime -> candidate expert model id
    - Load latest scorecard
    - Extract active vs candidate metric (overall)
    - Decide: promote / rollback / hold
    - Log deployment event
    - If promote and update_registry_on_promote=True, update registry/active_model.yaml

    Notes:
    - `candidate_model_id` is an alias of `candidate_id` for backwards compatibility.
      If both are provided, `candidate_model_id` takes precedence.
    """
    events_path = data_dir / "deployments" / "events.parquet"
    scorecards_dir = data_dir / "scorecards"
    run_ts = os.getenv("RUN_TS") or None

    regimes_path = data_dir / "regimes" / "latest.parquet"
    latest_regime = load_latest_regime(regimes_path)

    # Prefer the explicit alias if provided, else fall back to candidate_id.
    resolved_candidate_id: str | None = (
        candidate_model_id if candidate_model_id is not None else candidate_id
    )

    if resolved_candidate_id is None and latest_regime is not None and latest_regime != "unknown":
        project_root = data_dir.parent
        resolved_candidate_id = choose_pretrained_candidate_for_regime(
            project_root=project_root,
            data_dir=data_dir,
            regime=latest_regime,
            metric_name=config.metric_name,
        )

    # If we cannot choose a candidate, we still ran, but cannot act
    if resolved_candidate_id is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
                "run_ts": run_ts,
                "source": "switcher",
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": None,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "n": None,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "pointer_written": False,
                "reason": f"no_candidate_for_regime={latest_regime}",
            },
        )
        return

    scorecard = load_latest_scorecard(scorecards_dir)

    if scorecard is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
                "run_ts": run_ts,
                "source": "switcher",
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": resolved_candidate_id,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "n": None,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "pointer_written": False,
                "reason": "scorecard_missing",
            },
        )
        return

    id_col = infer_model_id_col(scorecard)
    if id_col is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
                "run_ts": run_ts,
                "source": "switcher",
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": resolved_candidate_id,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "n": None,
                "metric_name": config.metric_name,
                "active_metric_value": None,
                "candidate_metric_value": None,
                "decision": "no_action",
                "pointer_written": False,
                "reason": "scorecard_missing_model_id_column",
            },
        )
        return

    active_metric = extract_metric(
        scorecard,
        model_id=active_model_id,
        metric_name=config.metric_name,
    )
    candidate_metric = extract_metric(
        scorecard,
        model_id=resolved_candidate_id,
        metric_name=config.metric_name,
    )
    n_val = extract_n(scorecard, model_id=active_model_id)

    if active_metric is None or candidate_metric is None:
        append_event(
            events_path,
            {
                "ts": datetime.now(UTC).isoformat(),
                "run_ts": run_ts,
                "source": "switcher",
                "event_type": "canary_evaluated",
                "active_model_id_before": active_model_id,
                "candidate_model_id": resolved_candidate_id,
                "active_model_id_after": active_model_id,
                "window_type": config.window_type,
                "window_value": config.window_value,
                "n": n_val,
                "metric_name": config.metric_name,
                "active_metric_value": active_metric,
                "candidate_metric_value": candidate_metric,
                "decision": "no_action",
                "pointer_written": False,
                "reason": f"metrics_missing_for_model_id_or_metric regime={latest_regime}",
            },
        )
        return

    decision, reason = decide(
        active_metric=active_metric,
        candidate_metric=candidate_metric,
        promote_margin=config.promote_margin,
        rollback_margin=config.rollback_margin,
    )

    active_after = active_model_id
    event_type = "canary_evaluated"
    decision_label = decision
    pointer_written = False

    if decision == "promote":
        project_root = data_dir.parent
        predictions_path = project_root / "data" / "predictions" / "latest.parquet"
        candidate_type = "baseline"
        if (
            resolved_candidate_id.startswith("expert_lightgbm_")
            or resolved_candidate_id.startswith("expert_arima_")
            or resolved_candidate_id in {"expert_bullish", "expert_bearish", "expert_sideways"}
        ):
            candidate_type = "expert"
        elif resolved_candidate_id != "baseline":
            candidate_type = "pretrained"

        regime = None
        for maybe_regime in ("bullish", "bearish", "sideways"):
            if (
                resolved_candidate_id == f"expert_{maybe_regime}"
                or resolved_candidate_id.endswith(f"_{maybe_regime}")
                or resolved_candidate_id.startswith(f"expert_arima_{maybe_regime}_")
            ):
                regime = maybe_regime
                break

        guard = evaluate_candidate_promotion_guard(
            candidate_ref=ActiveModelRef(
                model_type=candidate_type,
                model_id=resolved_candidate_id,
                version="0",
                artifact_path=(
                    (
                        Path(f"models/experts/{regime}/arima/{resolved_candidate_id}.json")
                        if resolved_candidate_id.startswith("expert_arima_") and regime is not None
                        else Path(f"models/experts/{regime}/latest.joblib")
                    )
                    if candidate_type == "expert" and regime is not None
                    else Path(f"models/pretrained/{resolved_candidate_id}.joblib")
                    if candidate_type == "pretrained"
                    else Path("models/baseline/latest.joblib")
                ),
                regime=regime,
                metadata_path=(
                    (
                        project_root
                        / "models"
                        / "experts"
                        / regime
                        / "arima"
                        / f"{resolved_candidate_id}.json"
                        if resolved_candidate_id.startswith("expert_arima_")
                        else project_root / "models" / "experts" / regime / "latest.json"
                    )
                    if regime is not None and candidate_type == "expert"
                    else None
                ),
            ),
            predictions_path=predictions_path,
            candidate_model_name=resolved_candidate_id,
        )
        if guard.allowed:
            active_after = resolved_candidate_id
            event_type = "promoted"
            if config.update_registry_on_promote:
                registry_path = project_root / "registry" / "active_model.yaml"
                pointer_written = write_active_model_yaml(
                    registry_path,
                    resolved_candidate_id,
                    event_context={
                        "ts": datetime.now(UTC).replace(microsecond=0).isoformat(),
                        "source": "switcher",
                        "run_ts": run_ts,
                        "reason": reason,
                    },
                )
        else:
            decision_label = "blocked"
            event_type = "blocked"
            reason = f"{reason}; promotion_guard={guard.reason}"

    elif decision == "rollback":
        event_type = "rollback"

    else:  # hold
        event_type = "hold"

    append_event(
        events_path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "run_ts": run_ts,
            "source": "switcher",
            "event_type": event_type,
            "active_model_id_before": active_model_id,
            "candidate_model_id": resolved_candidate_id,
            "active_model_id_after": active_after,
            "window_type": config.window_type,
            "window_value": config.window_value,
            "n": n_val,
            "metric_name": config.metric_name,
            "active_metric_value": active_metric,
            "candidate_metric_value": candidate_metric,
            "promotion_guard_allowed": guard.allowed if decision == "promote" else None,
            "pointer_written": pointer_written,
            "decision": decision_label,
            "reason": f"{reason} regime={latest_regime}",
        },
    )


def run() -> None:
    main()


# ---------- CLI ----------


def main() -> None:
    config = SwitchConfig()
    run_switcher(
        data_dir=Path("data"),
        config=config,
        active_model_id="baseline",
    )


if __name__ == "__main__":
    main()

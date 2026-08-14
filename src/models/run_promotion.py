# src/models/run_promotion.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.deploy.history import append_deployment_event
from src.models.promotion import (
    PromotionConfig,
    PromotionDecision,
    decide_promotion,
    summarize_walkforward,
)
from src.models.promotion_guard import (
    PromotionGuardResult,
    evaluate_candidate_promotion_guard,
)
from src.registry.registry import ActiveModelRef, write_active

LOG = logging.getLogger("promotion")

LINEAGE_LATEST = Path("artifacts/lineage/latest.json")

NON_PROMOTABLE_MODEL_RULES = ("active",)


def is_non_promotable_model(model_name: str) -> bool:
    name = str(model_name).strip()
    return name == "active"


def _infer_expert_regime(model_name: str) -> str | None:
    """Infer a regime from legacy and multi-expert model identifiers."""
    name = str(model_name).strip().lower()
    for regime in ("bullish", "bearish", "sideways"):
        if name == f"expert_{regime}" or name.endswith(f"_{regime}"):
            return regime
        if name.startswith(f"expert_arima_{regime}_"):
            return regime
        if name.startswith(f"expert_lightgbm_{regime}_"):
            return regime
    return None


def _raise_if_non_promotable(preferred: str | None, *, role: str) -> None:
    if preferred is not None and is_non_promotable_model(preferred):
        raise ValueError(f"Requested {role} model is non-promotable: {preferred}")


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _lineage_features_path(lineage: dict[str, Any]) -> Path | None:
    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    features = artifacts.get("features_parquet")
    if not isinstance(features, dict):
        return None
    path = features.get("path")
    if path in (None, ""):
        return None
    return Path(str(path))


def _portfolio_metrics_path_for_run(run_ts: str) -> Path:
    return Path("data/walkforward") / f"portfolio_metrics_{run_ts}.parquet"


def _promotion_out_path_for_run(run_ts: str) -> Path:
    return Path("data/walkforward") / f"promotion_{run_ts}.json"


def _choose_incumbent(wf: pd.DataFrame, preferred: str | None) -> str:
    if "model_name" not in wf.columns:
        raise ValueError("walk-forward table missing required column: model_name")

    _raise_if_non_promotable(preferred, role="incumbent")

    names = sorted(set(map(str, wf["model_name"].dropna().unique().tolist())))
    names = [n for n in names if not is_non_promotable_model(n)]

    if not names:
        raise ValueError("walk-forward table has no promotable model_name values")

    if preferred and preferred in names:
        return preferred
    if "baseline" in names:
        return "baseline"
    return names[0]


def _choose_challenger(wf: pd.DataFrame, incumbent: str, preferred: str | None) -> str:
    _raise_if_non_promotable(preferred, role="challenger")

    names = sorted(set(map(str, wf["model_name"].dropna().unique().tolist())))
    names = [n for n in names if not is_non_promotable_model(n)]

    others = [n for n in names if n != incumbent]
    if not others:
        raise ValueError(f"no challenger candidates, only incumbent present: {incumbent}")

    if preferred and preferred in others:
        return preferred

    tmp = wf.copy()
    tmp["model_name"] = tmp["model_name"].astype(str)
    tmp = tmp[tmp["model_name"].isin(others)]

    if "sharpe" not in tmp.columns:
        return others[0]

    means = tmp.groupby("model_name", sort=True)["sharpe"].mean()
    means = means.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if means.empty:
        return others[0]

    return str(means.sort_values(ascending=False).index[0])


def _resolve_ref_from_predictions(model_name: str, predictions_path: Path) -> ActiveModelRef:
    """
    Resolve an ActiveModelRef for `model_name` using the latest predictions parquet,
    which already contains: model_name, model_source, model_path.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions not found: {predictions_path}")

    df = pd.read_parquet(predictions_path)

    need = {"model_name", "model_source", "model_path"}
    missing = sorted([c for c in need if c not in df.columns])
    if missing:
        raise ValueError(
            f"predictions parquet missing required columns: {missing}. columns={list(df.columns)}"
        )

    # Find rows for the chosen model_name
    rows = df.loc[df["model_name"].astype(str) == model_name]
    if rows.empty:
        available = sorted(df["model_name"].astype(str).dropna().unique().tolist())
        raise ValueError(f"Model '{model_name}' not found in predictions. Available={available}")

    row0 = rows.iloc[0]
    model_source = str(row0["model_source"])
    model_path = Path(str(row0["model_path"]))

    # Normalize to registry model_type convention
    if model_source not in {"baseline", "expert", "pretrained"}:
        raise ValueError(f"Unknown model_source '{model_source}' for model '{model_name}'")

    # Infer metadata paths from the actual published artifact, rather than an
    # ambiguous single "latest" convention.
    regime: str | None = None
    metadata_path: Path | None = None
    if model_source == "expert":
        regime = _infer_expert_regime(model_name)
        metadata_path = model_path if model_path.suffix.lower() == ".json" else None
        if metadata_path is None and regime is not None:
            metadata_path = model_path.parent / "latest.json"
    elif model_source == "baseline":
        candidate_meta = model_path.with_suffix(".json")
        metadata_path = (
            candidate_meta if candidate_meta.exists() else model_path.parent / "latest.json"
        )
    elif model_source == "pretrained":
        candidate_meta = model_path.with_suffix(".json")
        metadata_path = candidate_meta if candidate_meta.exists() else None

    return ActiveModelRef(
        model_type=model_source,  # baseline | expert | pretrained
        model_id=model_name,  # IMPORTANT: keep aligned with wf/model_name everywhere
        version="0",
        artifact_path=model_path,
        regime=regime,
        metadata_path=metadata_path,
    )


def run_promotion(
    *,
    challenger_model_name: str | None,
    incumbent_model_name: str | None,
    challenger_ref: ActiveModelRef,  # kept for backwards compatibility, not used for pointer writing
    cfg: PromotionConfig | None = None,
    lineage_path: Path = LINEAGE_LATEST,
    write_pointer: bool = True,
) -> dict[str, Any]:
    """
    Read latest lineage -> find walkforward portfolio metrics parquet -> decide promotion.

    Fix in this version:
      - Resolve the *selected* challenger to an ActiveModelRef using predictions parquet,
        and write THAT pointer (instead of whatever was passed in via challenger_ref).
    """
    cfg = cfg or PromotionConfig()

    if not lineage_path.exists():
        raise FileNotFoundError(f"lineage not found: {lineage_path}")

    lineage = _read_json(lineage_path)
    run_ts = str(lineage.get("run_ts", "")).strip()
    if not run_ts:
        raise ValueError("lineage missing run_ts")

    wf_path = _portfolio_metrics_path_for_run(run_ts)
    if not wf_path.exists():
        raise FileNotFoundError(f"walk-forward portfolio metrics missing: {wf_path}")

    wf = pd.read_parquet(wf_path)

    if "model_name" not in wf.columns:
        raise ValueError(
            f"walk-forward portfolio metrics missing 'model_name'. columns={list(wf.columns)}"
        )

    available_models = sorted(set(map(str, wf["model_name"].dropna().unique().tolist())))
    promotable_models = [m for m in available_models if not is_non_promotable_model(m)]
    LOG.info(
        "Promotion start | run_ts=%s wf_path=%s n_rows=%d n_models=%d",
        run_ts,
        wf_path,
        int(len(wf)),
        int(len(available_models)),
    )
    LOG.info("Promotion available models | %s", available_models)

    # ---- robust model selection ----
    incumbent = _choose_incumbent(wf, preferred=incumbent_model_name)
    challenger = _choose_challenger(wf, incumbent=incumbent, preferred=challenger_model_name)

    LOG.info(
        "Promotion selected | incumbent=%s (preferred=%s) challenger=%s (preferred=%s)",
        incumbent,
        incumbent_model_name,
        challenger,
        challenger_model_name,
    )

    # Resolve the actually-selected challenger to an artifact path (source of truth: predictions)
    preds_path = Path("data/predictions/latest.parquet")
    resolved_ref = _resolve_ref_from_predictions(challenger, preds_path)
    lineage_features_path = _lineage_features_path(lineage)

    # Summaries + decision
    chal = summarize_walkforward(wf, model_name=challenger, cfg=cfg)
    inc = summarize_walkforward(wf, model_name=incumbent, cfg=cfg)

    raw_decision = decide_promotion(
        challenger_summary=chal,
        incumbent_summary=inc,
        cfg=cfg,
    )

    promotion_guard: PromotionGuardResult | None = None
    decision = raw_decision
    if raw_decision.promote:
        promotion_guard = evaluate_candidate_promotion_guard(
            candidate_ref=resolved_ref,
            predictions_path=preds_path,
            current_features_path=lineage_features_path,
            candidate_model_name=challenger,
        )
        if not promotion_guard.allowed:
            decision = PromotionDecision(
                promote=False,
                reason=promotion_guard.reason,
                challenger=raw_decision.challenger,
                incumbent=raw_decision.incumbent,
                deltas=raw_decision.deltas,
            )

    # A short, human-friendly reason string for logs + top-level output
    reason = getattr(decision, "reason", None)
    if reason is None:
        reason = getattr(decision, "message", None)
    if reason is None:
        reason = "see decision object"

    promoted = bool(getattr(decision, "promote", False))
    event_type = "promoted" if promoted else "hold"
    decision_label = "promote" if promoted else "hold"
    if raw_decision.promote and not promoted:
        event_type = "blocked"
        decision_label = "blocked"

    event_ts = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Build output artifact
    out: dict[str, Any] = {
        "run_ts": run_ts,
        "git_commit": lineage.get("git_commit"),
        "config_sha256": lineage.get("config_sha256"),
        "challenger_model_name": challenger,
        "incumbent_model_name": incumbent,
        # Top-level visibility fields
        "promoted": promoted,
        "reason": str(reason),
        "pointer_written": bool(promoted and write_pointer),
        "raw_decision": asdict(raw_decision),
        # What we actually resolved + would write
        "resolved_challenger_ref": {
            "model_type": resolved_ref.model_type,
            "model_id": resolved_ref.model_id,
            "version": resolved_ref.version,
            "artifact_path": resolved_ref.artifact_path.as_posix(),
            "regime": resolved_ref.regime,
            "metadata_path": resolved_ref.metadata_path.as_posix()
            if resolved_ref.metadata_path is not None
            else None,
        },
        "promotion_guard": asdict(promotion_guard) if promotion_guard is not None else None,
        # Keep the passed challenger_ref visible for debugging (but do NOT write it)
        "passed_challenger_ref": {
            "model_type": challenger_ref.model_type,
            "model_id": challenger_ref.model_id,
            "version": challenger_ref.version,
            "artifact_path": challenger_ref.artifact_path.as_posix(),
            "regime": challenger_ref.regime,
            "metadata_path": challenger_ref.metadata_path.as_posix()
            if challenger_ref.metadata_path is not None
            else None,
        },
        # Full structured details
        "decision": asdict(decision),
        "promotion_config": asdict(cfg),
        "inputs": {
            "walkforward_metrics": str(wf_path),
            "lineage": str(lineage_path),
            "predictions_latest": str(preds_path),
        },
        "available_models": available_models,
        "promotable_models": promotable_models,
        "non_promotable_models": list(NON_PROMOTABLE_MODEL_RULES),
    }

    out_path = _promotion_out_path_for_run(run_ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = Path("data/walkforward/latest_promotion.json")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    # Write pointer if promoted
    pointer_written = False
    if promoted and write_pointer:
        LOG.warning(
            "MODEL PROMOTED | run_ts=%s challenger=%s incumbent=%s -> writing active pointer: %s",
            run_ts,
            challenger,
            incumbent,
            resolved_ref.artifact_path,
        )
        pointer_written = write_active(
            resolved_ref,
            event_context={
                "source": "run_promotion",
                "run_ts": run_ts,
                "reason": str(reason),
                "ts": event_ts,
            },
        )
    else:
        LOG.info(
            "Model not promoted | run_ts=%s promoted=%s reason=%s",
            run_ts,
            promoted,
            str(reason),
        )

    out["pointer_written"] = bool(pointer_written and promoted and write_pointer)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    append_deployment_event(
        Path("data/deployments/events.parquet"),
        {
            "ts": event_ts,
            "run_ts": run_ts,
            "source": "run_promotion",
            "event_type": event_type,
            "decision": decision_label,
            "active_model_id_before": incumbent,
            "candidate_model_id": challenger,
            "active_model_id_after": challenger if promoted else incumbent,
            "metric_name": cfg.sharpe_col,
            "active_metric_value": inc.get("sharpe"),
            "candidate_metric_value": chal.get("sharpe"),
            "active_max_drawdown": inc.get("max_drawdown"),
            "candidate_max_drawdown": chal.get("max_drawdown"),
            "promotion_guard_allowed": promotion_guard.allowed
            if promotion_guard is not None
            else None,
            "pointer_written": bool(pointer_written and promoted and write_pointer),
            "reason": str(reason),
        },
    )

    LOG.info("Promotion wrote artifact | %s", out_path)
    return out

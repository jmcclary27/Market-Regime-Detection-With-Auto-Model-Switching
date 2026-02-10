# src/models/run_promotion.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.promotion import (
    PromotionConfig,
    decide_promotion,
    summarize_walkforward,
)
from src.registry.registry import ActiveModelRef, write_active


LINEAGE_LATEST = Path("artifacts/lineage/latest.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _portfolio_metrics_path_for_run(run_ts: str) -> Path:
    # You can pick whatever convention you want, but keep it deterministic
    return Path("data/walkforward") / f"portfolio_metrics_{run_ts}.parquet"


def _promotion_out_path_for_run(run_ts: str) -> Path:
    return Path("data/walkforward") / f"promotion_{run_ts}.json"


def run_promotion(
    *,
    challenger_model_name: str,
    incumbent_model_name: str,
    challenger_ref: ActiveModelRef,  # what we'd activate if promoted
    cfg: PromotionConfig | None = None,
    lineage_path: Path = LINEAGE_LATEST,
    write_pointer: bool = True,
) -> dict[str, Any]:
    cfg = cfg or PromotionConfig()

    if not lineage_path.exists():
        raise FileNotFoundError(f"lineage not found: {lineage_path}")

    lineage = _read_json(lineage_path)
    run_ts = str(lineage.get("run_ts", "")).strip()
    if not run_ts:
        raise ValueError("lineage missing run_ts")

    wf_path = _portfolio_metrics_path_for_run(run_ts)
    if not wf_path.exists():
        raise FileNotFoundError(
            f"walk-forward portfolio metrics missing: {wf_path}. "
            "PR14 should add this artifact during walk-forward evaluation."
        )

    wf = pd.read_parquet(wf_path)

    chal = summarize_walkforward(wf, model_name=challenger_model_name, cfg=cfg)
    inc = summarize_walkforward(wf, model_name=incumbent_model_name, cfg=cfg)

    decision = decide_promotion(
        challenger_summary=chal,
        incumbent_summary=inc,
        cfg=cfg,
    )

    out = {
        "run_ts": run_ts,
        "git_commit": lineage.get("git_commit"),
        "config_sha256": lineage.get("config_sha256"),
        "challenger_model_name": challenger_model_name,
        "incumbent_model_name": incumbent_model_name,
        "decision": asdict(decision),
        "promotion_config": asdict(cfg),
        "inputs": {
            "walkforward_metrics": str(wf_path),
            "lineage": str(lineage_path),
        },
    }

    out_path = _promotion_out_path_for_run(run_ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = Path("data/walkforward/latest_promotion.json")
    latest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    # Optional: update registry pointer
    if decision.promote and write_pointer:
        # preserve updated_at auto behavior by not setting it here
        write_active(challenger_ref)

    return out

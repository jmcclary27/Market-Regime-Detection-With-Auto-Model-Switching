# src/models/run_promotion.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.promotion import PromotionConfig, decide_promotion, summarize_walkforward
from src.registry.registry import ActiveModelRef, write_active

LINEAGE_LATEST = Path("artifacts/lineage/latest.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _portfolio_metrics_path_for_run(run_ts: str) -> Path:
    return Path("data/walkforward") / f"portfolio_metrics_{run_ts}.parquet"


def _promotion_out_path_for_run(run_ts: str) -> Path:
    return Path("data/walkforward") / f"promotion_{run_ts}.json"


def _choose_incumbent(wf: pd.DataFrame, preferred: str | None) -> str:
    names = sorted(set(map(str, wf["model_name"].dropna().unique().tolist())))
    if not names:
        raise ValueError("walk-forward table has no model_name values")

    if preferred and preferred in names:
        return preferred
    if "baseline" in names:
        return "baseline"
    return names[0]


def _choose_challenger(wf: pd.DataFrame, incumbent: str, preferred: str | None) -> str:
    names = sorted(set(map(str, wf["model_name"].dropna().unique().tolist())))
    others = [n for n in names if n != incumbent]
    if not others:
        raise ValueError(f"no challenger candidates, only incumbent present: {incumbent}")

    if preferred and preferred in others:
        return preferred

    # default: pick best by mean sharpe across splits
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


def run_promotion(
    *,
    challenger_model_name: str | None,
    incumbent_model_name: str | None,
    challenger_ref: ActiveModelRef,
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
        raise FileNotFoundError(f"walk-forward portfolio metrics missing: {wf_path}")

    wf = pd.read_parquet(wf_path)

    # ---- robust model selection ----
    incumbent = _choose_incumbent(wf, preferred=incumbent_model_name)
    challenger = _choose_challenger(wf, incumbent=incumbent, preferred=challenger_model_name)

    chal = summarize_walkforward(wf, model_name=challenger, cfg=cfg)
    inc = summarize_walkforward(wf, model_name=incumbent, cfg=cfg)

    decision = decide_promotion(
        challenger_summary=chal,
        incumbent_summary=inc,
        cfg=cfg,
    )

    out = {
        "run_ts": run_ts,
        "git_commit": lineage.get("git_commit"),
        "config_sha256": lineage.get("config_sha256"),
        "challenger_model_name": challenger,
        "incumbent_model_name": incumbent,
        "decision": asdict(decision),
        "promotion_config": asdict(cfg),
        "inputs": {
            "walkforward_metrics": str(wf_path),
            "lineage": str(lineage_path),
        },
        "available_models": sorted(set(map(str, wf["model_name"].dropna().unique().tolist()))),
    }

    out_path = _promotion_out_path_for_run(run_ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = Path("data/walkforward/latest_promotion.json")
    latest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    if decision.promote and write_pointer:
        write_active(challenger_ref)

    return out

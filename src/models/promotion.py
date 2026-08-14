# src/models/promotion.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class PromotionConfig:
    model_col: str = "model_name"
    fold_col: str = "fold_id"

    # These match src/backtest/metrics.py names
    sharpe_col: str = "sharpe"
    maxdd_col: str = "max_drawdown"

    # Rule
    max_drawdown_threshold: float = 0.35  # allowed magnitude, e.g. 0.35 => require max_dd >= -0.35
    require_sharpe_beats_incumbent: bool = True

    # Fold aggregation
    sharpe_agg: Literal["mean", "median"] = "mean"
    maxdd_agg: Literal["min", "mean"] = "min"  # "min" = worst (most negative) across folds


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    challenger: dict[str, float]
    incumbent: dict[str, float]
    deltas: dict[str, float]


def load_walkforward_metrics(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"walk-forward metrics not found: {p}")
    return pd.read_parquet(p)


def _agg(series: pd.Series, mode: str) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    if mode == "mean":
        return float(s.mean())
    if mode == "median":
        return float(s.median())
    if mode == "min":
        return float(s.min())
    raise ValueError(f"unknown agg mode: {mode}")


def summarize_walkforward(
    wf: pd.DataFrame,
    *,
    model_name: str,
    cfg: PromotionConfig,
) -> dict[str, float]:
    required = {cfg.model_col, cfg.sharpe_col, cfg.maxdd_col}
    missing = required - set(wf.columns)
    if missing:
        raise ValueError(f"walk-forward metrics missing columns: {sorted(missing)}")

    m = wf[wf[cfg.model_col] == model_name].copy()
    if m.empty:
        raise ValueError(f"no rows found for model_name={model_name}")

    sharpe = _agg(m[cfg.sharpe_col], cfg.sharpe_agg)
    maxdd = _agg(m[cfg.maxdd_col], cfg.maxdd_agg)  # negative number, more negative is worse

    n_folds = float(m[cfg.fold_col].nunique()) if cfg.fold_col in m.columns else float("nan")

    return {"sharpe": sharpe, "max_drawdown": maxdd, "n_folds": n_folds}


def decide_promotion(
    *,
    challenger_summary: dict[str, float],
    incumbent_summary: dict[str, float],
    cfg: PromotionConfig,
) -> PromotionDecision:
    c_sh = float(challenger_summary.get("sharpe", float("nan")))
    i_sh = float(incumbent_summary.get("sharpe", float("nan")))
    c_dd = float(challenger_summary.get("max_drawdown", float("nan")))
    i_dd = float(incumbent_summary.get("max_drawdown", float("nan")))

    deltas = {"sharpe": c_sh - i_sh, "max_drawdown": c_dd - i_dd}

    # Must have challenger metrics
    if pd.isna(c_sh) or pd.isna(c_dd):
        return PromotionDecision(
            promote=False,
            reason="challenger metrics missing (nan sharpe or max_drawdown)",
            challenger=challenger_summary,
            incumbent=incumbent_summary,
            deltas=deltas,
        )

    # Drawdown guardrail: max_drawdown is negative, e.g. -0.25.
    # Allowed threshold is magnitude (positive), so require c_dd >= -threshold.
    dd_floor = -float(cfg.max_drawdown_threshold)
    if c_dd < dd_floor:
        return PromotionDecision(
            promote=False,
            reason=f"challenger max_drawdown {c_dd:.3f} worse than allowed floor {dd_floor:.3f}",
            challenger=challenger_summary,
            incumbent=incumbent_summary,
            deltas=deltas,
        )

    if cfg.require_sharpe_beats_incumbent:
        if pd.isna(i_sh):
            return PromotionDecision(
                promote=False,
                reason="incumbent sharpe missing (nan), refusing to auto-promote",
                challenger=challenger_summary,
                incumbent=incumbent_summary,
                deltas=deltas,
            )
        if c_sh <= i_sh:
            return PromotionDecision(
                promote=False,
                reason=f"challenger sharpe {c_sh:.4f} does not beat incumbent {i_sh:.4f}",
                challenger=challenger_summary,
                incumbent=incumbent_summary,
                deltas=deltas,
            )

    return PromotionDecision(
        promote=True,
        reason="challenger beats incumbent on sharpe and satisfies max_drawdown guardrail",
        challenger=challenger_summary,
        incumbent=incumbent_summary,
        deltas=deltas,
    )

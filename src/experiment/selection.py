"""Pre-registered selection rules for frozen experiment model artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def select_static_model(scorecard: pd.DataFrame) -> dict[str, Any]:
    """Select a single global candidate by net Sharpe, drawdown, then return.

    The caller supplies only the predeclared global Ridge and global LightGBM
    candidates; specialist models are not eligible for this control arm.
    """
    required = {"model_id", "walk_forward_net_sharpe", "max_drawdown", "cumulative_return"}
    missing = required.difference(scorecard.columns)
    if missing:
        raise ValueError(f"Scorecard is missing required columns: {sorted(missing)}")
    eligible = scorecard.copy()
    if "is_global_candidate" in eligible.columns:
        eligible = eligible[eligible["is_global_candidate"].fillna(False)]
    if eligible.empty:
        raise ValueError("No eligible global static-model candidates")
    eligible = eligible.dropna(
        subset=["walk_forward_net_sharpe", "max_drawdown", "cumulative_return"]
    )
    if eligible.empty:
        raise ValueError("No global candidates have complete walk-forward metrics")
    ordered = eligible.sort_values(
        ["walk_forward_net_sharpe", "max_drawdown", "cumulative_return", "model_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return dict(ordered.iloc[0])

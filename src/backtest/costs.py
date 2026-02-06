from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0


def compute_costs_from_turnover(
    turnover: pd.Series,
    cfg: CostConfig,
) -> pd.Series:
    """
    Deterministic linear cost model:
      cost_t = turnover_t * (fee + spread + slippage) / 10000

    turnover is fraction of portfolio traded at t (>=0).
    """
    t = turnover.astype(float).fillna(0.0).clip(lower=0.0)
    rate = (float(cfg.fee_bps) + float(cfg.spread_bps) + float(cfg.slippage_bps)) / 10_000.0
    costs = t * rate
    # safety: no -0.0 or inf
    costs = costs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return costs

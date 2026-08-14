from __future__ import annotations

import pandas as pd

from src.backtest.costs import CostConfig, compute_costs_from_turnover


def test_costs_linear_in_turnover_and_bps() -> None:
    turnover = pd.Series([0.0, 0.5, 1.0, 2.0])  # 200% turnover allowed in test
    cfg = CostConfig(fee_bps=1.0, spread_bps=2.0, slippage_bps=3.0)  # total 6 bps
    costs = compute_costs_from_turnover(turnover, cfg)

    # 6 bps = 0.0006
    expected = turnover * 0.0006
    pd.testing.assert_series_equal(costs, expected)


def test_costs_clip_negative_turnover_to_zero() -> None:
    turnover = pd.Series([-1.0, 0.0, 1.0])
    cfg = CostConfig(fee_bps=10.0)
    costs = compute_costs_from_turnover(turnover, cfg)

    assert float(costs.iloc[0]) == 0.0
    assert float(costs.iloc[1]) == 0.0
    assert float(costs.iloc[2]) == 1.0 * (10.0 / 10_000.0)

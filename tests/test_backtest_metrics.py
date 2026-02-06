from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    compute_cagr,
    compute_max_drawdown,
    compute_portfolio_metrics,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
)


def test_max_drawdown_monotone_equity_is_zero() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    equity = pd.Series([100.0, 101.0, 105.0, 110.0, 120.0], index=idx)
    mdd = compute_max_drawdown(equity)
    assert np.isfinite(mdd)
    assert mdd == 0.0


def test_max_drawdown_simple_drop() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    # peak 120 -> trough 90 => 90/120 - 1 = -0.25
    mdd = compute_max_drawdown(equity)
    assert np.isfinite(mdd)
    assert abs(mdd - (-0.25)) < 1e-12


def test_sharpe_constant_returns_is_nan() -> None:
    r = pd.Series([0.001] * 10)
    s = compute_sharpe(r, periods_per_year=252)
    assert np.isnan(s)


def test_sortino_all_positive_returns_is_nan() -> None:
    r = pd.Series([0.001] * 10)
    s = compute_sortino(r, periods_per_year=252)
    assert np.isnan(s)


def test_profit_factor_basic() -> None:
    r = pd.Series([0.02, -0.01, 0.01, -0.02])
    # gains = 0.03, losses = 0.03 => PF = 1
    pf = compute_profit_factor(r)
    assert np.isfinite(pf)
    assert abs(pf - 1.0) < 1e-12


def test_cagr_known_constant_daily_return() -> None:
    # If returns are constant, CAGR should be close to (1+r)^ppy - 1
    r_daily = 0.001
    n = 252
    r = pd.Series([r_daily] * n)
    cagr = compute_cagr(r, periods_per_year=252)
    expected = (1.0 + r_daily) ** 252 - 1.0
    assert np.isfinite(cagr)
    assert abs(cagr - expected) < 1e-12


def test_compute_portfolio_metrics_smoke() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    results_df = pd.DataFrame(
        {
            "equity": [100.0, 101.0, 100.0, 102.0, 103.0],
            "returns_net": [0.0, 0.01, -0.0099009901, 0.02, 0.0098039216],
        },
        index=idx,
    )
    trades_df = pd.DataFrame({"delta_position": [0.0, 1.0, 0.5, 0.0, -0.25]})

    m = compute_portfolio_metrics(results_df=results_df, trades_df=trades_df, periods_per_year=252)

    # We don't assert exact sharpe/sortino here (small sample),
    # but we do enforce type/stability and that turnover matches.
    assert np.isfinite(m.max_drawdown)
    assert np.isfinite(m.cagr) or np.isnan(m.cagr)
    assert m.turnover == float(trades_df["delta_position"].abs().sum())
    assert np.isfinite(m.profit_factor) or np.isinf(m.profit_factor) or np.isnan(m.profit_factor)

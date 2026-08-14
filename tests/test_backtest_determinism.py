from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest


def test_backtest_is_deterministic_for_fixed_seed() -> None:
    n = 200
    rng = np.random.default_rng(0)

    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    rets = rng.normal(loc=0.0002, scale=0.01, size=n)
    price = 100.0 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame({"SPY": price}, index=idx)

    sig = (pd.Series(rets, index=idx) > 0.0).astype(float)
    signals = pd.DataFrame({"SPY": sig}, index=idx)

    cfg = BacktestConfig(
        initial_cash=100_000.0,
        fee_bps=1.0,
        spread_bps=2.0,
        slippage_bps=3.0,
        seed=123,
    )

    r1 = run_backtest(prices=prices, signals=signals, cfg=cfg)
    r2 = run_backtest(prices=prices, signals=signals, cfg=cfg)

    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
    pd.testing.assert_series_equal(r1.returns_net, r2.returns_net)
    pd.testing.assert_frame_equal(r1.trades, r2.trades)


def test_backtest_changes_when_costs_change() -> None:
    n = 200
    rng = np.random.default_rng(0)

    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    rets = rng.normal(loc=0.0002, scale=0.01, size=n)
    price = 100.0 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame({"SPY": price}, index=idx)

    sig = (pd.Series(rets, index=idx) > 0.0).astype(float)
    signals = pd.DataFrame({"SPY": sig}, index=idx)

    cfg0 = BacktestConfig(
        initial_cash=100_000.0,
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        seed=123,
    )
    cfg1 = BacktestConfig(
        initial_cash=100_000.0,
        fee_bps=5.0,
        spread_bps=5.0,
        slippage_bps=5.0,
        seed=123,
    )

    r0 = run_backtest(prices=prices, signals=signals, cfg=cfg0)
    r1 = run_backtest(prices=prices, signals=signals, cfg=cfg1)

    assert not r0.returns_net.equals(r1.returns_net)
    assert float(r1.equity_curve.iloc[-1]) <= float(r0.equity_curve.iloc[-1]) + 1e-9

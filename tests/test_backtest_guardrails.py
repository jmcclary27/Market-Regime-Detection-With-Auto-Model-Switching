# tests/test_backtest_guardrails.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest


def test_backtest_outputs_are_finite() -> None:
    """
    CI guardrail: prevent NaN/inf explosions in the core backtest outputs.
    """
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    # Simple, deterministic upward-ish price path (no randomness needed)
    price = 100.0 + pd.Series(range(n), dtype="float64") * 0.1
    prices = pd.DataFrame({"SPY": price.to_numpy()}, index=idx)

    # Simple deterministic signal that flips a few times
    sig = pd.Series(0.0, index=idx, dtype="float64")
    sig.iloc[10:50] = 1.0
    sig.iloc[50:100] = -1.0
    sig.iloc[100:200] = 0.5
    sig.iloc[200:] = 0.0
    signals = pd.DataFrame({"SPY": sig}, index=idx)

    cfg = BacktestConfig(
        initial_cash=100_000.0,
        fee_bps=1.0,
        spread_bps=1.0,
        slippage_bps=2.0,
        seed=0,
    )

    res = run_backtest(prices=prices, signals=signals, cfg=cfg)

    assert np.isfinite(float(res.equity_curve.iloc[-1]))
    assert np.isfinite(float(res.returns_net.sum()))
    assert np.isfinite(float(res.returns_gross.sum()))

    # No NaNs in key series
    assert not res.equity_curve.isna().any()
    assert not res.returns_net.isna().any()
    assert not res.returns_gross.isna().any()

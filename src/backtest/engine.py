from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    max_leverage: float = 1.0
    clip_signal: float = 1.0  # clip to [-clip_signal, +clip_signal]
    seed: int = 0

    # frictions (bps = 1/10,000)
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    returns_gross: pd.Series
    returns_net: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame


def run_backtest(
    *,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: BacktestConfig,
) -> BacktestResult:
    """
    Deterministic backtest v1.

    Contract:
      - prices index is timestamp, must be sorted ascending
      - signals index matches prices index
      - columns overlap (we use the intersection); for v1 assume single column SPY
      - signal is interpreted as target position in [-1, 1]
    """
    if prices.empty:
        raise ValueError("prices is empty")
    if signals.empty:
        raise ValueError("signals is empty")
    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals index must match exactly")

    # v1: use common columns
    cols = [c for c in signals.columns if c in prices.columns]
    if not cols:
        raise ValueError("signals columns must overlap prices columns")
    if len(cols) != 1:
        raise ValueError(f"v1 expects 1 traded column, got {cols}")

    col = cols[0]
    px = prices[col].astype(float)

    # close-to-close returns
    ret = px.pct_change().fillna(0.0)

    # target positions from signals
    sig = signals[col].astype(float).fillna(0.0)
    target = sig.clip(lower=-cfg.clip_signal, upper=cfg.clip_signal)

    # Apply leverage cap
    target = target.clip(lower=-cfg.max_leverage, upper=cfg.max_leverage)

    # positions held from t to t+1 => use lagged position
    pos = target.shift(1).fillna(0.0)

    # Gross strategy returns
    strat_gross = pos * ret

    # Trades happen when target changes (using target, not lagged pos)
    delta_pos = target.diff().fillna(target)
    turnover = delta_pos.abs()  # fraction of portfolio traded (since position is fraction)

    # Simple deterministic costs in return space (bps applied to turnover)
    from src.backtest.costs import CostConfig, compute_costs_from_turnover

    costs = compute_costs_from_turnover(
        turnover,
        CostConfig(fee_bps=cfg.fee_bps, spread_bps=cfg.spread_bps, slippage_bps=cfg.slippage_bps),
    )

    strat_net = strat_gross - costs

    equity = (1.0 + strat_net).cumprod() * float(cfg.initial_cash)

    positions = pd.DataFrame({col: pos}, index=prices.index)

    trades = pd.DataFrame(
        {
            "timestamp": prices.index,
            "asset": col,
            "target_position": target.values,
            "position": pos.values,
            "delta_position": delta_pos.values,
            "price": px.values,
            "ret": ret.values,
            "gross_return": strat_gross.values,
            "cost": costs.values,
            "net_return": strat_net.values,
        }
    )

    return BacktestResult(
        equity_curve=equity.rename("equity"),
        returns_gross=strat_gross.rename("returns_gross"),
        returns_net=strat_net.rename("returns_net"),
        positions=positions,
        trades=trades,
    )

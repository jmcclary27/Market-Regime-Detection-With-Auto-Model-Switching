from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioMetrics:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover: float
    profit_factor: float


def _safe_float(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def compute_max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return _safe_float(dd.min())


def compute_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.astype(float)
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    n_years = len(r) / float(periods_per_year)
    if n_years <= 0.0 or total <= 0.0:
        return float("nan")
    return _safe_float(total ** (1.0 / n_years) - 1.0)


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.astype(float)
    if r.empty:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if sd == 0.0:
        return float("nan")
    return _safe_float((mu / sd) * np.sqrt(periods_per_year))


def compute_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.astype(float)
    if r.empty:
        return float("nan")
    mu = float(r.mean())
    downside = r[r < 0.0]
    dd = float(downside.std(ddof=0)) if not downside.empty else 0.0
    if dd == 0.0:
        return float("nan")
    return _safe_float((mu / dd) * np.sqrt(periods_per_year))


def compute_profit_factor(returns: pd.Series) -> float:
    r = returns.astype(float)
    if r.empty:
        return float("nan")
    gains = float(r[r > 0.0].sum())
    losses = float((-r[r < 0.0]).sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return _safe_float(gains / losses)


def compute_turnover(trades: pd.DataFrame) -> float:
    # Our engine emits delta_position as "fraction of portfolio traded"
    if "delta_position" not in trades.columns:
        return float("nan")
    return _safe_float(float(trades["delta_position"].abs().sum()))


def compute_portfolio_metrics(
    *,
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    periods_per_year: int = 252,
) -> PortfolioMetrics:
    if "returns_net" not in results_df.columns:
        raise ValueError("results_df missing returns_net")
    if "equity" not in results_df.columns:
        raise ValueError("results_df missing equity")

    returns_net = results_df["returns_net"]
    equity = results_df["equity"]

    return PortfolioMetrics(
        cagr=compute_cagr(returns_net, periods_per_year=periods_per_year),
        sharpe=compute_sharpe(returns_net, periods_per_year=periods_per_year),
        sortino=compute_sortino(returns_net, periods_per_year=periods_per_year),
        max_drawdown=compute_max_drawdown(equity),
        turnover=compute_turnover(trades_df),
        profit_factor=compute_profit_factor(returns_net),
    )

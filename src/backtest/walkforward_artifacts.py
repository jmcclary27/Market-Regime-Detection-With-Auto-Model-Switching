# src/backtest/walkforward_artifacts.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.backtest.metrics import PortfolioMetrics


def write_walkforward_portfolio_metrics(
    rows: list[dict[str, object]],
    *,
    out_dir: str | Path = "data/walkforward",
    run_ts: str,
) -> Path:
    """
    rows: list of dicts with keys at least:
      - fold_id (int or str)
      - model_name (str)
      - plus PortfolioMetrics fields (cagr, sharpe, sortino, max_drawdown, turnover, profit_factor)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    required = {
        "fold_id",
        "model_name",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "turnover",
        "profit_factor",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"walk-forward portfolio metrics missing columns: {sorted(missing)}")

    p = out_dir / f"portfolio_metrics_{run_ts}.parquet"
    df.to_parquet(p, index=False)

    (out_dir / "latest.parquet").write_bytes(p.read_bytes())
    return p


def portfolio_metrics_row(
    *,
    run_ts: str,
    fold_id: int,
    model_name: str,
    metrics: PortfolioMetrics,
) -> dict[str, object]:
    r: dict[str, object] = {
        "run_ts": run_ts,
        "fold_id": int(fold_id),
        "model_name": str(model_name),
    }
    r.update(asdict(metrics))
    return r

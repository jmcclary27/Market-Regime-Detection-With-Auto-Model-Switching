"""Build sanitized static-dashboard payloads from immutable experiment events."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.metrics import compute_max_drawdown, compute_sharpe, compute_sortino
from src.experiment.engine import PORTFOLIOS
from src.experiment.manifest import FrozenExperimentManifest


def _metrics(values: pd.Series, *, initial_cash: float) -> dict[str, float | None]:
    returns = values.astype(float).pct_change().dropna()
    if values.empty:
        return {"cumulative_return": None, "sharpe": None, "max_drawdown": None, "sortino": None}
    return {
        "cumulative_return": float(values.iloc[-1] / initial_cash - 1),
        "sharpe": float(compute_sharpe(returns)) if len(returns) >= 20 else None,
        "max_drawdown": float(compute_max_drawdown(values)),
        "sortino": float(compute_sortino(returns)) if len(returns) >= 20 else None,
    }


def build_dashboard_payload(
    *, manifest: FrozenExperimentManifest, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return only public, non-secret fields used by the static dashboard."""
    latest = events[-1] if events else None
    histories: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, float | None]] = {}
    for name in PORTFOLIOS:
        series = pd.Series([event["portfolio_values"][name] for event in events], dtype=float)
        histories[name] = [
            {"date": event["bar_timestamp_utc"], "value": float(event["portfolio_values"][name])}
            for event in events
        ]
        metrics[name] = _metrics(series, initial_cash=manifest.starting_cash)
    return {
        "schema_version": 1,
        "disclaimer": "Paper trading only. Frozen-model out-of-sample experiment; not investment advice.",
        "experiment_id": manifest.experiment_id,
        "data_cutoff": manifest.data_cutoff,
        "study_sessions": len(events),
        "current_regime": None if latest is None else latest["regime"],
        "regime_confidence": None if latest is None else latest["regime_confidence"],
        "active_regime_model": None if latest is None else latest["regime_model_id"],
        "metrics": metrics,
        "equity_history": histories,
        "recent_decisions": [
            {
                "date": event["bar_timestamp_utc"],
                "regime": event["regime"],
                "static_target": event["targets"]["static_ml"],
                "regime_target": event["targets"]["regime_ml"],
            }
            for event in events[-10:][::-1]
        ],
        "regime_history": [
            {"date": event["bar_timestamp_utc"], "regime": event["regime"]} for event in events
        ],
    }

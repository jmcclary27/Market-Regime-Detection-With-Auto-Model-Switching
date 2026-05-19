from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from src.trading.execution import ExecutionConfig, execute_signal, log_trade
from src.trading.signals import SignalConfig, prediction_to_signal
from src.trading.state import (
    load_account_state,
    log_account_snapshot,
    save_account_state,
)


def run_trading_cycle(
    *,
    prediction: float,
    price: float,
    regime: str | None = None,
    active_model_id: str | None = None,
    state_path: Path = Path("data/live_sim/account_state.json"),
    trades_path: Path = Path("data/live_sim/trades.parquet"),
    equity_path: Path = Path("data/live_sim/equity_curve.parquet"),
    starting_cash: float = 100_000.0,
    signal_config: SignalConfig | None = None,
    execution_config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(UTC).isoformat()

    state = load_account_state(state_path, starting_cash=starting_cash)

    signal = prediction_to_signal(
        prediction,
        regime=regime,
        config=signal_config,
    )

    trade = execute_signal(
        state=state,
        signal=signal,
        price=price,
        timestamp=timestamp,
        config=execution_config,
    )

    trade.update(
        {
            "regime": regime,
            "active_model_id": active_model_id,
            "prediction": float(prediction),
        }
    )

    log_trade(trades_path, trade)

    log_account_snapshot(
        path=equity_path,
        state=state,
        timestamp=timestamp,
        price=price,
        regime=regime,
        active_model_id=active_model_id,
        prediction=float(prediction),
        signal=signal,
        action_taken=trade["action"],
        reason=trade["reason"],
    )

    save_account_state(state, state_path)

    return trade
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.trading.state import AccountState, append_event


@dataclass
class ExecutionConfig:
    max_position_pct: float = 0.95
    trade_fraction: float = 0.25
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_short: bool = False


def execute_signal(
    *,
    state: AccountState,
    signal: str,
    price: float,
    timestamp: str | None = None,
    config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    config = config or ExecutionConfig()
    timestamp = timestamp or datetime.now(UTC).isoformat()

    signal = signal.upper()
    price = float(price)

    state.mark_to_market(price)

    action = "NONE"
    shares_delta = 0.0
    reason = "hold signal"
    fill_price = price
    trade_value = 0.0
    fee = 0.0
    realized_pnl_delta = 0.0

    slippage = price * (config.slippage_bps / 10_000)
    cost_rate = config.transaction_cost_bps / 10_000

    if signal == "BUY":
        max_position_value = state.portfolio_value * config.max_position_pct
        current_position_value = state.position * price
        available_position_room = max_position_value - current_position_value

        budget = min(state.cash * config.trade_fraction, available_position_room)

        if budget <= 0:
            reason = "no cash or max position reached"
        else:
            fill_price = price + slippage
            shares_delta = budget / fill_price
            trade_value = shares_delta * fill_price
            fee = trade_value * cost_rate

            if trade_value + fee > state.cash:
                trade_value = state.cash / (1 + cost_rate)
                fee = trade_value * cost_rate
                shares_delta = trade_value / fill_price

            state.cash -= trade_value + fee
            state.position += shares_delta
            state.cost_basis += trade_value + fee
            action = "BUY"
            reason = "bought simulated shares"

    elif signal == "SELL":
        if state.position <= 0 and not config.allow_short:
            reason = "no long position to sell"
        else:
            shares_delta = -state.position * config.trade_fraction

            if abs(shares_delta) <= 0:
                reason = "position too small to sell"
            else:
                old_position = state.position
                old_cost_basis = state.cost_basis
                fill_price = price - slippage
                trade_value = abs(shares_delta) * fill_price
                fee = trade_value * cost_rate
                cost_basis_delta = old_cost_basis * (abs(shares_delta) / old_position)
                realized_pnl_delta = trade_value - cost_basis_delta - fee

                state.cash += trade_value - fee
                state.position += shares_delta
                state.cost_basis -= cost_basis_delta
                state.realized_pnl += realized_pnl_delta
                if state.position <= 1e-12:
                    state.position = 0.0
                    state.cost_basis = 0.0
                action = "SELL"
                reason = "sold simulated shares"

    elif signal == "HOLD":
        reason = "hold signal"

    else:
        reason = f"unknown signal: {signal}"

    state.mark_to_market(price)

    return {
        "timestamp": timestamp,
        "signal": signal,
        "action": action,
        "price": price,
        "fill_price": fill_price,
        "shares_delta": shares_delta,
        "trade_value": trade_value,
        "fee": fee,
        "realized_pnl_delta": realized_pnl_delta,
        "cash": state.cash,
        "position": state.position,
        "portfolio_value": state.portfolio_value,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": state.unrealized_pnl,
        "total_pnl": state.total_pnl,
        "cost_basis": state.cost_basis,
        "avg_entry_price": state.avg_entry_price,
        "reason": reason,
    }


def log_trade(path: Path, trade: dict[str, Any]) -> None:
    append_event(path, trade)

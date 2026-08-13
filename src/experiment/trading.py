"""Long-only target-exposure accounting for the frozen experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.trading.state import AccountState


@dataclass(frozen=True)
class TargetExposureConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 2.0


def prediction_to_target(prediction: float, *, lower: float, upper: float) -> float:
    if prediction <= lower:
        return 0.0
    if prediction >= upper:
        return 1.0
    return 0.5


def rebalance_to_target(
    *, state: AccountState, target_exposure: float, open_price: float, config: TargetExposureConfig
) -> dict[str, Any]:
    """Fill a previously queued target at the next bar's open, with common friction."""
    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must be between 0 and 1")
    if open_price <= 0:
        raise ValueError("open_price must be positive")
    state.mark_to_market(open_price)
    starting_value = state.portfolio_value
    current_value = state.position * open_price
    desired_value = starting_value * target_exposure
    delta_value = desired_value - current_value
    fee_rate = config.fee_bps / 10_000
    slip_rate = config.slippage_bps / 10_000
    action = "NONE"
    shares_delta = 0.0
    fee = 0.0
    fill_price = open_price

    if delta_value > 1e-9:
        fill_price = open_price * (1 + slip_rate)
        # A buy's target is expressed at the reference open. Scale it if the
        # fee and adverse fill would otherwise exceed cash.
        spend = min(delta_value, state.cash / (1 + fee_rate))
        shares_delta = spend / fill_price
        trade_value = shares_delta * fill_price
        fee = trade_value * fee_rate
        state.cash -= trade_value + fee
        if state.cash > -1e-9:
            state.cash = 0.0
        state.position += shares_delta
        state.cost_basis += trade_value + fee
        action = "BUY"
    elif delta_value < -1e-9:
        fill_price = open_price * (1 - slip_rate)
        shares_delta = max(delta_value / open_price, -state.position)
        quantity = -shares_delta
        trade_value = quantity * fill_price
        fee = trade_value * fee_rate
        old_position = state.position
        basis_released = state.cost_basis * quantity / old_position if old_position else 0.0
        state.position -= quantity
        state.cost_basis -= basis_released
        state.cash += trade_value - fee
        state.realized_pnl += trade_value - fee - basis_released
        if state.position <= 1e-12:
            state.position = 0.0
            state.cost_basis = 0.0
        action = "SELL"
    else:
        trade_value = 0.0

    state.mark_to_market(open_price)
    return {
        "action": action,
        "target_exposure": target_exposure,
        "current_exposure_before": current_value / starting_value if starting_value else 0.0,
        "shares_delta": shares_delta,
        "reference_open_price": open_price,
        "fill_price": fill_price,
        "trade_value": trade_value,
        "fee": fee,
        "cash": state.cash,
        "position": state.position,
        "portfolio_value": state.portfolio_value,
    }

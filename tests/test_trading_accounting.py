from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trading.execution import ExecutionConfig, execute_signal
from src.trading.state import AccountState, load_account_state


def _cfg() -> ExecutionConfig:
    return ExecutionConfig(
        max_position_pct=1.0,
        trade_fraction=0.5,
        transaction_cost_bps=100.0,
        slippage_bps=0.0,
    )


def test_first_buy_sets_cost_basis_and_unrealized_pnl() -> None:
    state = AccountState(cash=1_000.0, portfolio_value=1_000.0)

    trade = execute_signal(state=state, signal="BUY", price=10.0, config=_cfg())

    assert trade["action"] == "BUY"
    assert state.cash == pytest.approx(495.0)
    assert state.position == pytest.approx(50.0)
    assert state.cost_basis == pytest.approx(505.0)
    assert state.avg_entry_price == pytest.approx(10.1)
    assert state.unrealized_pnl == pytest.approx(-5.0)
    assert state.total_pnl == pytest.approx(-5.0)


def test_second_buy_updates_weighted_average_cost() -> None:
    state = AccountState(cash=1_000.0, portfolio_value=1_000.0)
    execute_signal(state=state, signal="BUY", price=10.0, config=_cfg())

    execute_signal(state=state, signal="BUY", price=20.0, config=_cfg())

    assert state.cash == pytest.approx(245.025)
    assert state.position == pytest.approx(62.375)
    assert state.cost_basis == pytest.approx(754.975)
    assert state.avg_entry_price == pytest.approx(754.975 / 62.375)
    assert state.unrealized_pnl == pytest.approx(492.525)
    assert state.total_pnl == pytest.approx(492.525)


def test_partial_sell_realizes_pnl_and_reduces_cost_basis() -> None:
    state = AccountState(cash=0.0, position=100.0, cost_basis=1_000.0)
    state.mark_to_market(10.0)
    cfg = ExecutionConfig(
        max_position_pct=1.0,
        trade_fraction=0.25,
        transaction_cost_bps=100.0,
        slippage_bps=0.0,
    )

    trade = execute_signal(state=state, signal="SELL", price=20.0, config=cfg)

    assert trade["action"] == "SELL"
    assert trade["shares_delta"] == pytest.approx(-25.0)
    assert trade["realized_pnl_delta"] == pytest.approx(245.0)
    assert state.cash == pytest.approx(495.0)
    assert state.position == pytest.approx(75.0)
    assert state.cost_basis == pytest.approx(750.0)
    assert state.realized_pnl == pytest.approx(245.0)
    assert state.unrealized_pnl == pytest.approx(750.0)
    assert state.total_pnl == pytest.approx(995.0)


def test_hold_only_marks_to_market() -> None:
    state = AccountState(cash=100.0, position=10.0, cost_basis=100.0)
    state.mark_to_market(10.0)

    trade = execute_signal(state=state, signal="HOLD", price=12.0, config=_cfg())

    assert trade["action"] == "NONE"
    assert state.cash == pytest.approx(100.0)
    assert state.position == pytest.approx(10.0)
    assert state.cost_basis == pytest.approx(100.0)
    assert state.unrealized_pnl == pytest.approx(20.0)
    assert state.total_pnl == pytest.approx(20.0)


def test_old_account_state_auto_migrates(tmp_path: Path) -> None:
    path = tmp_path / "account_state.json"
    path.write_text(
        json.dumps(
            {
                "cash": 500.0,
                "position": 50.0,
                "last_price": 12.0,
                "portfolio_value": 1_100.0,
                "realized_pnl": 0.0,
                "updated_at": "2026-05-28T20:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    state = load_account_state(path, starting_cash=1_000.0)

    assert state.cost_basis == pytest.approx(500.0)
    assert state.avg_entry_price == pytest.approx(10.0)
    assert state.portfolio_value == pytest.approx(1_100.0)
    assert state.unrealized_pnl == pytest.approx(100.0)
    assert state.total_pnl == pytest.approx(100.0)

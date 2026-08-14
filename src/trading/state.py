from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AccountState:
    cash: float = 100_000.0
    position: float = 0.0
    last_price: float = 0.0
    portfolio_value: float = 100_000.0
    realized_pnl: float = 0.0
    avg_entry_price: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    updated_at: str = ""

    def mark_to_market(self, price: float) -> None:
        self.last_price = float(price)
        if self.position <= 0:
            self.position = 0.0
            self.cost_basis = 0.0
            self.avg_entry_price = 0.0
        else:
            self.avg_entry_price = float(self.cost_basis / self.position)
        self.portfolio_value = float(self.cash + self.position * price)
        self.unrealized_pnl = float(self.position * price - self.cost_basis)
        self.total_pnl = float(self.realized_pnl + self.unrealized_pnl)
        self.updated_at = datetime.now(UTC).isoformat()


def _account_state_from_dict(data: dict[str, Any], *, starting_cash: float) -> AccountState:
    cash = float(data.get("cash", starting_cash))
    position = float(data.get("position", 0.0))
    last_price = float(data.get("last_price", 0.0))
    realized_pnl = float(data.get("realized_pnl", 0.0))

    cost_basis = data.get("cost_basis")
    if cost_basis is None and position > 0:
        inferred = starting_cash - cash
        cost_basis = inferred if inferred > 0 else 0.0

    state = AccountState(
        cash=cash,
        position=position,
        last_price=last_price,
        portfolio_value=float(data.get("portfolio_value", cash + position * last_price)),
        realized_pnl=realized_pnl,
        avg_entry_price=float(data.get("avg_entry_price", 0.0)),
        cost_basis=float(cost_basis or 0.0),
        unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
        total_pnl=float(data.get("total_pnl", realized_pnl)),
        updated_at=str(data.get("updated_at", "")),
    )
    if state.position > 0 and state.cost_basis > 0:
        state.avg_entry_price = float(state.cost_basis / state.position)
    if state.last_price > 0:
        state.mark_to_market(state.last_price)
    return state


def load_account_state(path: Path, starting_cash: float = 100_000.0) -> AccountState:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        state = AccountState(
            cash=starting_cash,
            portfolio_value=starting_cash,
            updated_at=datetime.now(UTC).isoformat(),
        )
        save_account_state(state, path)
        return state

    data = json.loads(path.read_text(encoding="utf-8"))
    return _account_state_from_dict(data, starting_cash=starting_cash)


def save_account_state(state: AccountState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([event])

    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row

    out.to_parquet(path, index=False)


def log_account_snapshot(
    *,
    path: Path,
    state: AccountState,
    timestamp: str,
    price: float,
    regime: str | None = None,
    active_model_id: str | None = None,
    prediction: float | None = None,
    signal: str | None = None,
    action_taken: str | None = None,
    reason: str | None = None,
) -> None:
    append_event(
        path,
        {
            "timestamp": timestamp,
            "cash": state.cash,
            "position": state.position,
            "last_price": price,
            "portfolio_value": state.portfolio_value,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": state.unrealized_pnl,
            "total_pnl": state.total_pnl,
            "cost_basis": state.cost_basis,
            "avg_entry_price": state.avg_entry_price,
            "regime": regime,
            "active_model_id": active_model_id,
            "prediction": prediction,
            "signal": signal,
            "action_taken": action_taken,
            "reason": reason,
        },
    )

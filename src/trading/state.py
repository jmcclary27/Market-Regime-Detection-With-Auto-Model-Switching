from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
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
    updated_at: str = ""

    def mark_to_market(self, price: float) -> None:
        self.last_price = float(price)
        self.portfolio_value = float(self.cash + self.position * price)
        self.updated_at = datetime.now(UTC).isoformat()


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
    return AccountState(**data)


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
            "regime": regime,
            "active_model_id": active_model_id,
            "prediction": prediction,
            "signal": signal,
            "action_taken": action_taken,
            "reason": reason,
        },
    )
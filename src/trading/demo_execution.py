from __future__ import annotations

from pathlib import Path

from src.trading.execution import execute_signal, log_trade
from src.trading.state import load_account_state, save_account_state


def main() -> None:
    state_path = Path("data/live_sim/account_state.json")
    trades_path = Path("data/live_sim/trades.parquet")

    state = load_account_state(state_path)

    trade = execute_signal(
        state=state,
        signal="BUY",
        price=500.0,
    )

    log_trade(trades_path, trade)
    save_account_state(state, state_path)

    print(trade)


if __name__ == "__main__":
    main()
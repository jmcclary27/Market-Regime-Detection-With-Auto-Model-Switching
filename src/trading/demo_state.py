from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.trading.state import (
    load_account_state,
    log_account_snapshot,
    save_account_state,
)


def main() -> None:
    state_path = Path("data/live_sim/account_state.json")
    equity_path = Path("data/live_sim/equity_curve.parquet")

    state = load_account_state(state_path, starting_cash=100_000.0)

    price = 500.0
    state.mark_to_market(price)

    log_account_snapshot(
        path=equity_path,
        state=state,
        timestamp=datetime.now(UTC).isoformat(),
        price=price,
        regime="sideways",
        active_model_id="expert_lightgbm_sideways",
        prediction=0.0012,
        signal="HOLD",
        action_taken="NONE",
        reason="demo snapshot only",
    )

    save_account_state(state, state_path)

    print(f"Saved account state to: {state_path}")
    print(f"Saved equity curve to: {equity_path}")
    print(state)


if __name__ == "__main__":
    main()

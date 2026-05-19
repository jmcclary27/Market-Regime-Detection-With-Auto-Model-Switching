from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from src.trading.trading_cycle import run_trading_cycle


def get_latest_prediction(predictions_path: Path) -> dict:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_parquet(predictions_path)

    if df.empty:
        raise ValueError(f"Predictions file is empty: {predictions_path}")

    row = df.sort_values("timestamp").iloc[-1]

    return {
        "timestamp": row.get("timestamp"),
        "prediction": float(row["prediction"]),
        "price": float(row.get("close", row.get("close_x"))),
        "regime": str(row.get("regime", "unknown")),
        "active_model_id": str(row.get("model_name", "unknown")),
    }


def main() -> None:
    predictions_path = Path(os.getenv("PREDICTIONS_PATH", "data/predictions/latest.parquet"))
    interval_seconds = int(os.getenv("LIVE_SIM_INTERVAL_SECONDS", "60"))

    print(f"Starting live simulation loop")
    print(f"Predictions path: {predictions_path}")
    print(f"Interval seconds: {interval_seconds}")

    while True:
        try:
            latest = get_latest_prediction(predictions_path)

            result = run_trading_cycle(
                prediction=latest["prediction"],
                price=latest["price"],
                regime=latest["regime"],
                active_model_id=latest["active_model_id"],
            )

            print(result)

        except Exception as exc:
            print(f"Live simulation cycle failed: {exc}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
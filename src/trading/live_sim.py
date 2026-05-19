from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.trading.trading_cycle import run_trading_cycle


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_latest_prediction(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    preds = pd.read_parquet(predictions_path)

    if preds.empty:
        raise ValueError(f"Predictions file is empty: {predictions_path}")

    # Prefer the active model prediction
    if "is_active" in preds.columns:
        active = preds[preds["is_active"] == True]
        if not active.empty:
            preds = active

    if "inference_ts" in preds.columns:
        preds = preds.sort_values("inference_ts")

    row = preds.iloc[-1]

    prediction = float(row["y_pred"])
    row_id = int(row["row_id"])

    features_path = Path(str(row["features_path"]))
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features = pd.read_parquet(features_path)

    if row_id >= len(features):
        raise ValueError(
            f"row_id={row_id} is out of bounds for features with {len(features)} rows"
        )

    feature_row = features.iloc[row_id]

    price_col = _first_existing_column(
        features,
        ["close_x", "close", "price", "last_price", "close_y"],
    )

    if price_col is None:
        raise ValueError(f"No price column found in features. columns={list(features.columns)}")

    regime_col = _first_existing_column(
        features,
        ["regime", "regime_label", "detected_regime", "active_regime"],
    )

    return {
        "timestamp": str(row.get("inference_ts", row_id)),
        "prediction": prediction,
        "price": float(feature_row[price_col]),
        "regime": str(row.get("active_regime", None)),
        "active_model_id": str(row.get("active_model_id", row.get("model_name", "unknown"))),
    }


def main() -> None:
    predictions_path = Path(os.getenv("PREDICTIONS_PATH", "data/predictions/latest.parquet"))
    interval_seconds = int(os.getenv("LIVE_SIM_INTERVAL_SECONDS", "60"))

    print("Starting live simulation loop")
    print(f"Predictions path: {predictions_path}")
    print(f"Interval seconds: {interval_seconds}")

    last_seen_timestamp: str | None = None

    while True:
        try:
            latest = get_latest_prediction(predictions_path)

            if latest["timestamp"] == last_seen_timestamp:
                print(f"No new prediction. Last timestamp={last_seen_timestamp}")
            else:
                result = run_trading_cycle(
                    prediction=latest["prediction"],
                    price=latest["price"],
                    regime=latest["regime"],
                    active_model_id=latest["active_model_id"],
                )

                last_seen_timestamp = latest["timestamp"]
                print(result)

        except Exception as exc:
            print(f"Live simulation cycle failed: {exc}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
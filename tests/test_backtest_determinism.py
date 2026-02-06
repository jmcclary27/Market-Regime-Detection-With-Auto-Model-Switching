from __future__ import annotations

import pandas as pd

from src.backtest.adapters import signals_from_predictions_long


def test_signals_adapter_uses_active_only_and_aligns_by_row_id() -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
        }
    )

    preds = pd.DataFrame(
        {
            "row_id": [0, 0, 1, 1, 2],
            "model_name": ["a", "b", "a", "b", "a"],
            "y_pred": [0.1, 0.9, 0.2, 0.8, 0.3],
            "is_active": [False, True, True, False, True],
        }
    )

    out = signals_from_predictions_long(preds, features=features, asset="SPY")
    sig = out.signals["SPY"]

    # row_id 0 uses active 0.9, row_id 1 uses active 0.2, row_id 2 uses active 0.3
    assert float(sig.iloc[0]) == 0.9
    assert float(sig.iloc[1]) == 0.2
    assert float(sig.iloc[2]) == 0.3

    # others default to 0.0
    assert float(sig.iloc[3]) == 0.0
    assert float(sig.iloc[9]) == 0.0

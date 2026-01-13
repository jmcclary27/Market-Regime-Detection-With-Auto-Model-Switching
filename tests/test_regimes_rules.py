import pandas as pd

from src.regimes.rules import label_regimes


def test_label_regimes_outputs_expected_columns():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:31:00"],
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 101.0],
            "log_return_1": [float("nan"), 0.01],
            "sma_10": [float("nan"), float("nan")],
        }
    )

    out = label_regimes(df)

    assert "regime" in out.columns
    assert "regime_explanation" in out.columns
    assert len(out) == len(df)

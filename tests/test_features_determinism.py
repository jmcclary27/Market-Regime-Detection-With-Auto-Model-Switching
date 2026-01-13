import pandas as pd

from src.features.builder import build_features


def test_build_features_is_deterministic_under_shuffle():
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 09:30:00",
                "2026-01-01 09:31:00",
                "2026-01-01 09:32:00",
                "2026-01-01 09:30:00",
                "2026-01-01 09:31:00",
                "2026-01-01 09:32:00",
            ],
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "close": [100, 101, 99, 200, 199, 201],
        }
    )

    f1 = build_features(bars)

    bars_shuffled = bars.sample(frac=1.0, random_state=123).reset_index(drop=True)
    f2 = build_features(bars_shuffled)

    # Same columns, same row order, same values
    assert list(f1.columns) == list(f2.columns)
    pd.testing.assert_frame_equal(f1, f2, check_dtype=True)

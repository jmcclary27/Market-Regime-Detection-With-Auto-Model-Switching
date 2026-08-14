import pandas as pd

from src.features.builder import build_features
from src.features.manifest import dataframe_sha256, schema_from_df


def test_schema_from_df_returns_name_and_dtype():
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:31:00"],
            "symbol": ["AAA", "AAA"],
            "close": [100, 101],
        }
    )
    feats = build_features(bars)
    schema = schema_from_df(feats)

    assert isinstance(schema, list)
    assert all("name" in c and "dtype" in c for c in schema)
    assert [c["name"] for c in schema] == list(feats.columns)


def test_dataframe_sha256_is_stable_for_same_features():
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
    h1 = dataframe_sha256(f1)

    bars_shuffled = bars.sample(frac=1.0, random_state=7).reset_index(drop=True)
    f2 = build_features(bars_shuffled)
    h2 = dataframe_sha256(f2)

    assert h1 == h2
    assert len(h1) == 64

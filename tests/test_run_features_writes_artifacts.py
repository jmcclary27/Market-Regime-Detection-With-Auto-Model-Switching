from pathlib import Path

import pandas as pd

from src.features.builder import build_features
from src.features.manifest import FeatureManifest, dataframe_sha256, schema_from_df, write_manifest


def test_writes_parquet_and_manifest(tmp_path: Path):
    # Arrange: make a tiny bars df
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

    feats = build_features(bars)

    timestamp = "2026-01-02T170000Z"
    parquet_path = tmp_path / f"{timestamp}.parquet"
    manifest_path = tmp_path / f"{timestamp}.manifest.json"

    # Act: write parquet + manifest using the same utilities as the runner
    feats.to_parquet(parquet_path, index=False)

    manifest = FeatureManifest(
        timestamp=timestamp,
        parquet_path=str(parquet_path.as_posix()),
        row_count=int(len(feats)),
        columns=schema_from_df(feats),
        content_sha256=dataframe_sha256(feats),
    )
    write_manifest(manifest, manifest_path)

    # Assert: files exist and basic contents make sense
    assert parquet_path.exists()
    assert manifest_path.exists()

    reloaded = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(reloaded, feats, check_dtype=True)

    text = manifest_path.read_text(encoding="utf-8")
    assert manifest.content_sha256 in text

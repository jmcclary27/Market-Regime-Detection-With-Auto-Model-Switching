from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ingestion.quality import audit_raw_bars


def test_audit_raw_bars_flags_duplicates_missing_and_lateness(tmp_path: Path) -> None:
    raw_path = tmp_path / "bars.parquet"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T14:00:00Z",
                "2026-01-01T14:05:00Z",
                "2026-01-01T14:05:00Z",
                "2026-01-01T14:15:00Z",
            ],
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "open": [1.0, 1.1, 1.1, 1.3],
            "close": [1.0, 1.1, 1.1, 1.3],
        }
    ).to_parquet(raw_path, index=False)

    out_path = tmp_path / "audit.json"
    audit = audit_raw_bars(
        raw_path=raw_path,
        run_ts="20260101_000000Z",
        finished_at_utc="2026-01-01T14:40:00Z",
        symbols=["SPY"],
        interval="5m",
        provider_failure_count=1,
        provider_attempt_count=2,
        output_path=out_path,
    )

    assert audit["duplicate_bar_count"] == 1
    assert audit["missing_bar_count"] == 1
    assert audit["late_data_count"] == 1
    assert audit["provider_failure_rate"] == 0.5
    assert audit["status"] == "warning"
    assert json.loads(out_path.read_text(encoding="utf-8"))["run_ts"] == "20260101_000000Z"

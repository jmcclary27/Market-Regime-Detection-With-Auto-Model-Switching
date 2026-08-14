from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.jobs.poll_market_data import main


def test_poll_job_writes_files(tmp_path: Path, monkeypatch) -> None:
    # Run inside temp dir so we don't touch the real repo
    monkeypatch.chdir(tmp_path)

    def fake_fetch(symbol: str, _start: str, _end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=3, freq="D"),
                "close": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "open": [100.0, 101.0, 102.0],
                "volume": [1_000, 1_100, 1_200],
                "symbol": symbol,
            }
        )

    monkeypatch.setattr("src.jobs.poll_market_data.fetch_market_data", fake_fetch)

    main([])

    # Assert raw output exists
    raw_dir = Path("data") / "raw"
    assert raw_dir.exists()
    assert any(raw_dir.glob("SPY_*.csv")) or any(raw_dir.glob("SPY-*_*.csv"))
    assert (raw_dir / "latest.csv").exists()

    # Assert run log exists
    runs_dir = Path("data") / "runs"
    assert runs_dir.exists()
    assert any(runs_dir.glob("poll_SPY_*.json")) or any(runs_dir.glob("poll_SPY-*_*.json"))

from __future__ import annotations

from pathlib import Path

from src.jobs.poll_market_data import main


def test_poll_job_writes_files(tmp_path: Path, monkeypatch) -> None:
    # Run inside temp dir so we don't touch the real repo
    monkeypatch.chdir(tmp_path)

    main()

    # Assert raw output exists
    raw_dir = Path("data") / "raw"
    assert raw_dir.exists()
    assert any(raw_dir.glob("SPY_*.csv"))
    assert (raw_dir / "SPY_latest.csv").exists()

    # Assert run log exists
    runs_dir = Path("data") / "runs"
    assert runs_dir.exists()
    assert any(runs_dir.glob("poll_SPY_*.json"))

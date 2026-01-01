from pathlib import Path

from src.jobs.poll_market_data import main


def test_poll_job_writes_files():
    main()

    raw_dir = Path("data/raw")
    runs_dir = Path("runs")

    assert raw_dir.exists()
    assert runs_dir.exists()

    # latest pointer should exist
    assert (raw_dir / "SPY_latest.csv").exists()

    # at least one run log should exist
    logs = list(runs_dir.glob("poll_SPY_*.json"))
    assert len(logs) >= 1
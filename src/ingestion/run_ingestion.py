# src/ingestion/run_ingestion.py
from __future__ import annotations

from pathlib import Path

from src.jobs.poll_market_data import main as poll_market_data_main


def main(*, config_path: Path | None = None) -> None:
    argv = [] if config_path is None else ["--config", str(config_path)]
    poll_market_data_main(argv)


def run(*, config_path: Path | None = None) -> None:
    main(config_path=config_path)


if __name__ == "__main__":
    main()

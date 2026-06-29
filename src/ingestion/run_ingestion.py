# src/ingestion/run_ingestion.py
from __future__ import annotations

from src.jobs.poll_market_data import main as poll_market_data_main


def main() -> None:
    poll_market_data_main([])


def run() -> None:
    main()


if __name__ == "__main__":
    main()

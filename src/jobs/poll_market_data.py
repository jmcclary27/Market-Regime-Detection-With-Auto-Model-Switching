# src/jobs/poll_market_data.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.ingestion.fetch_market_data import fetch_market_data


@dataclass
class PollRunRecord:
    started_at_utc: str
    finished_at_utc: str
    symbols: list[str]
    start_date: str
    end_date: str
    output_path: str
    latest_path: str
    rows: int


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    cfg = load_config()

    symbols = list(cfg["market"]["symbols"][:2])
    if len(symbols) < 2:
        raise ValueError(
            "Need at least 2 symbols in cfg['market']['symbols'] for *_x/*_y features."
        )

    start_date = "2020-01-01"
    end_date = "2020-03-01" 

    raw_dir = Path(cfg["data"]["raw_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = fetch_market_data(sym, start_date, end_date).copy()

        # Ensure symbol column exists, even if fetch_market_data returns only an index + OHLCV
        if "symbol" not in df.columns:
            df["symbol"] = sym

        frames.append(df)

    bars = pd.concat(frames, axis=0, ignore_index=False)

    ts = _utc_ts()
    symbol_tag = "-".join(symbols)

    output_file = raw_dir / f"{symbol_tag}_{start_date}_{end_date}_{ts}.csv"
    bars.to_csv(output_file, index=True)

    latest_file = raw_dir / "latest.csv"
    bars.to_csv(latest_file, index=True)

    finished = datetime.now(UTC)

    record = PollRunRecord(
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_path=str(output_file),
        latest_path=str(latest_file),
        rows=int(len(bars)),
    )

    run_log = runs_dir / f"poll_{symbol_tag}_{ts}.json"
    run_log.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    print(f"Wrote: {output_file}")
    print(f"Updated latest: {latest_file}")
    print(f"Run log: {run_log}")


if __name__ == "__main__":
    main()

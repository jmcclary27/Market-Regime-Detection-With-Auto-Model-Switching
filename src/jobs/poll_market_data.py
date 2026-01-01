from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.load_config import load_config
from src.ingestion.fetch_market_data import fetch_market_data


@dataclass
class PollRunRecord:
    started_at_utc: str
    finished_at_utc: str
    symbol: str
    start_date: str
    end_date: str
    output_path: str
    latest_path: str
    rows: int


def _utc_ts() -> str:
    # File-system friendly timestamp
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    cfg = load_config()

    symbol = cfg["market"]["symbols"][0]
    # keep PR1 tiny and deterministic, can be config-driven later
    start_date = "2020-01-01"
    end_date = "2020-01-10"

    raw_dir = Path(cfg["data"]["raw_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)

    df = fetch_market_data(symbol, start_date, end_date)

    ts = _utc_ts()
    output_file = raw_dir / f"{symbol}_{start_date}_{end_date}_{ts}.csv"
    df.to_csv(output_file, index=True)

    latest_file = raw_dir / f"{symbol}_latest.csv"
    df.to_csv(latest_file, index=True)

    finished = datetime.now(timezone.utc)

    record = PollRunRecord(
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_path=str(output_file),
        latest_path=str(latest_file),
        rows=int(len(df)),
    )

    run_log = runs_dir / f"poll_{symbol}_{ts}.json"
    run_log.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    print(f"Wrote: {output_file}")
    print(f"Updated latest: {latest_file}")
    print(f"Run log: {run_log}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.ingestion.fetch_market_data import fetch_market_data
from src.ingestion.quality import audit_raw_bars, default_audit_output


@dataclass
class PollRunRecord:
    run_ts: str
    started_at_utc: str
    finished_at_utc: str
    symbols: list[str]
    start_date: str
    end_date: str
    interval: str
    output_path: str
    latest_path: str
    rows: int
    provider_failure_count: int
    provider_attempt_count: int
    data_quality_path: str | None = None


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Poll market data to data/raw")
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default=datetime.now(UTC).date().isoformat())
    p.add_argument(
        "--symbols", default=None, help="Comma-separated, default uses cfg market.symbols[:2]"
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = load_config()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(cfg["market"]["symbols"][:2])

    if len(symbols) < 2:
        raise ValueError(
            "Need at least 2 symbols in cfg['market']['symbols'] for *_x/*_y features."
        )

    start_date = args.start_date
    end_date = args.end_date

    raw_dir = Path(cfg["data"]["raw_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for sym in symbols:
        try:
            df = fetch_market_data(sym, start_date, end_date).copy()
        except Exception as exc:
            failures.append(f"{sym}: {exc}")
            continue
        if "symbol" not in df.columns:
            df["symbol"] = sym
        frames.append(df)

    if failures:
        raise RuntimeError(f"poll failed for {len(failures)} symbol(s): {failures}")

    bars = pd.concat(frames, axis=0, ignore_index=False)

    ts = _utc_ts()
    symbol_tag = "-".join(symbols)

    output_file = raw_dir / f"{symbol_tag}_{start_date}_{end_date}_{ts}.csv"
    bars.to_csv(output_file, index=True)

    latest_file = raw_dir / "latest.csv"
    bars.to_csv(latest_file, index=True)

    finished = datetime.now(UTC)

    record = PollRunRecord(
        run_ts=ts,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        output_path=str(output_file),
        latest_path=str(latest_file),
        rows=int(len(bars)),
        provider_failure_count=0,
        provider_attempt_count=len(symbols),
    )

    run_log = runs_dir / f"poll_{symbol_tag}_{ts}.json"
    run_payload = asdict(record)
    audit_path = default_audit_output(Path.cwd(), ts)
    audit_raw_bars(
        raw_path=output_file,
        run_ts=ts,
        finished_at_utc=record.finished_at_utc,
        symbols=symbols,
        interval=record.interval,
        provider_failure_count=record.provider_failure_count,
        provider_attempt_count=record.provider_attempt_count,
        run_type="poll_market_data",
        output_path=audit_path,
    )
    run_payload["data_quality_path"] = str(audit_path)
    run_log.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {output_file}")
    print(f"Updated latest: {latest_file}")
    print(f"Run log: {run_log}")


if __name__ == "__main__":
    main()

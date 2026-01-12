# src/regimes/run_regime_detection.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.regimes.rules import label_regimes


def utc_timestamp_for_filename() -> str:
    # e.g. 20260103T192530Z
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    input_path = Path("data/features/test-run.parquet")
    output_dir = Path("data/regimes")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    regimes = label_regimes(df)

    # Keep key identifiers for joining/debugging
    out = pd.concat(
        [
            df[["timestamp", "symbol"]].reset_index(drop=True),
            regimes.reset_index(drop=True),
        ],
        axis=1,
    )

    out_path = output_dir / f"regimes_{utc_timestamp_for_filename()}.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Wrote regimes to: {out_path}")
    print(out.head(10))

def run() -> None:
    main()

if __name__ == "__main__":
    main()
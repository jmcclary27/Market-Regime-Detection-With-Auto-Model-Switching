from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.regimes.rules import label_regimes


def run(
    *,
    input_path: Path,
    timestamp: str,
    config_path: Path = Path("src/config/settings.yaml"),
) -> Path:
    """
    Programmatic entrypoint for orchestration (PR 9).

    Reads:
      - features parquet at input_path

    Writes:
      - <regimes_dir>/<timestamp>.parquet

    Output dataframe includes original features plus:
      - regime
      - regime_explanation

    Returns:
      - regimes parquet path
    """
    cfg = load_config(str(config_path))

    data_cfg = cfg.get("data", {})
    regimes_dir = Path(data_cfg.get("regimes_path", "data/regimes"))
    regimes_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    labels = label_regimes(df)
    out = df.join(labels)

    regimes_path = regimes_dir / f"{timestamp}.parquet"
    out.to_parquet(regimes_path, index=False)
    
    latest_path = regimes_dir / "latest.parquet"
    out.to_parquet(latest_path, index=False)

    print(f"Wrote: {regimes_path}")
    return regimes_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run rule-based regime detection and write parquet.")
    p.add_argument("--input", required=True, help="Path to features parquet (.parquet)")
    p.add_argument("--timestamp", required=True, help="Timestamp slug, e.g. 20260112_150402Z")
    p.add_argument("--config", default="src/config/settings.yaml", help="Path to settings yaml")
    args = p.parse_args(argv)

    run(
        input_path=Path(args.input),
        timestamp=args.timestamp,
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()

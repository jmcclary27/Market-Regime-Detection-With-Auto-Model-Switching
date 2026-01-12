from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config.load_config import load_config
from src.features.builder import build_features
from src.features.manifest import (
    FeatureManifest,
    dataframe_sha256,
    schema_from_df,
    write_manifest,
)


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix} (use .csv or .parquet)")


def _normalize_bars_columns(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """
    Normalize raw bars into the canonical schema expected by the feature builder:
      - timestamp
      - symbol
    """
    out = df.copy()

    # Handle common timestamp column patterns
    if "timestamp" not in out.columns:
        if "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "timestamp"})
        elif "Date" in out.columns:
            out = out.rename(columns={"Date": "timestamp"})
        elif out.index.name in ("Date", "Datetime"):
            out = out.reset_index().rename(columns={out.index.name: "timestamp"})

    # Ensure timestamp is datetime
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    # Ensure symbol column exists
    if "symbol" not in out.columns:
        out["symbol"] = symbol

    return out


def run(
    *,
    input_path: Path,
    timestamp: str,
    config_path: Path = Path("src/config/settings.yaml"),
) -> Tuple[Path, Path]:
    """
    Programmatic entrypoint for orchestration (PR 9).

    Writes:
      - <features_dir>/<timestamp>.parquet
      - <features_dir>/<timestamp>.manifest.json

    Returns:
      (parquet_path, manifest_path)
    """
    cfg = load_config(str(config_path))
    features_dir = Path(cfg["data"]["features_path"])
    features_dir.mkdir(parents=True, exist_ok=True)

    bars = _read_input(input_path)

    # Infer symbol from filename, e.g. SPY_latest.csv or SPY_2020-01-01_...
    symbol = input_path.name.split("_", 1)[0]

    bars = _normalize_bars_columns(bars, symbol=symbol)

    feats = build_features(bars)

    parquet_path = features_dir / f"{timestamp}.parquet"
    feats.to_parquet(parquet_path, index=False)

    manifest = FeatureManifest(
        timestamp=timestamp,
        parquet_path=str(parquet_path.as_posix()),
        row_count=int(len(feats)),
        columns=schema_from_df(feats),
        content_sha256=dataframe_sha256(feats),
    )

    manifest_path = features_dir / f"{timestamp}.manifest.json"
    write_manifest(manifest, manifest_path)

    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {manifest_path}")
    print(f"SHA256: {manifest.content_sha256}")

    return parquet_path, manifest_path


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Build deterministic features and write parquet + manifest."
    )
    p.add_argument("--input", required=True, help="Path to bars input (.csv or .parquet)")
    p.add_argument("--timestamp", required=True, help="Timestamp slug, e.g. 2026-01-02T170000Z")
    p.add_argument("--config", default="src/config/settings.yaml", help="Path to settings yaml")
    args = p.parse_args(argv)

    run(
        input_path=Path(args.input),
        timestamp=args.timestamp,
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
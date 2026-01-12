from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.features.builder import build_features
from src.features.manifest import FeatureManifest, dataframe_sha256, schema_from_df, write_manifest


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix} (use .csv or .parquet)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build deterministic features and write parquet + manifest.")
    p.add_argument("--input", required=True, help="Path to bars input (.csv or .parquet)")
    p.add_argument("--timestamp", required=True, help="Timestamp slug, e.g. 2026-01-02T170000Z")
    p.add_argument("--config", default="src/config/settings.yaml", help="Path to settings yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    features_dir = Path(cfg["data"]["features_path"])
    features_dir.mkdir(parents=True, exist_ok=True)

    bars_path = Path(args.input)
    bars = _read_input(bars_path)

    feats = build_features(bars)

    parquet_path = features_dir / f"{args.timestamp}.parquet"
    feats.to_parquet(parquet_path, index=False)

    manifest = FeatureManifest(
        timestamp=args.timestamp,
        parquet_path=str(parquet_path.as_posix()),
        row_count=int(len(feats)),
        columns=schema_from_df(feats),
        content_sha256=dataframe_sha256(feats),
    )

    manifest_path = features_dir / f"{args.timestamp}.manifest.json"
    write_manifest(manifest, manifest_path)

    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {manifest_path}")
    print(f"SHA256: {manifest.content_sha256}")

def run() -> None:
    main()

if __name__ == "__main__":
    main()
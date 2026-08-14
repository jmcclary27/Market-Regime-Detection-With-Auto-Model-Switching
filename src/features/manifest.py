# src/features/manifest.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def schema_from_df(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return a simple schema description for the manifest."""
    return [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]


def dataframe_sha256(df: pd.DataFrame) -> str:
    """
    Deterministic content hash for a dataframe.

    Assumes df is already deterministic in row order and column order.
    Uses pandas' stable hashing for row content, then sha256 over the bytes.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h = hashlib.sha256()
    h.update(row_hashes.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FeatureManifest:
    timestamp: str
    parquet_path: str
    row_count: int
    columns: list[dict[str, str]]
    content_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def write_manifest(manifest: FeatureManifest, path: str | Path) -> None:
    path = Path(path)
    path.write_text(manifest.to_json(), encoding="utf-8")

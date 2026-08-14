# src/data/lineage.py
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def try_git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


@dataclass(frozen=True)
class LineageArtifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class RunLineage:
    run_ts: str
    git_commit: str | None
    config_sha256: str
    artifacts: dict[str, LineageArtifact]
    params: dict[str, Any]


def write_run_lineage(
    *,
    project_root: Path,
    run_ts: str,
    config_text: str,
    artifacts: dict[str, Path],
    params: dict[str, Any] | None = None,
) -> Path:
    out_dir = project_root / "artifacts" / "lineage"
    out_dir.mkdir(parents=True, exist_ok=True)

    art: dict[str, LineageArtifact] = {}
    for name, p in artifacts.items():
        if not p.exists():
            raise FileNotFoundError(f"Lineage artifact missing: {name} -> {p}")
        art[name] = LineageArtifact(path=str(p), sha256=sha256_file(p))

    lineage = RunLineage(
        run_ts=run_ts,
        git_commit=try_git_commit(project_root),
        config_sha256=sha256_text(config_text),
        artifacts=art,
        params=params or {},
    )

    out_path = out_dir / f"lineage_{run_ts}.json"
    out_path.write_text(json.dumps(asdict(lineage), indent=2, sort_keys=True), encoding="utf-8")

    # optional “latest”
    (out_dir / "latest.json").write_text(
        json.dumps(asdict(lineage), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return out_path

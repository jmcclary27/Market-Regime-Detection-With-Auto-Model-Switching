"""Create the immutable active-plus-shadow model bundle consumed by Lambda.

This tool performs no AWS calls.  Upload the resulting tarball separately,
record its S3 VersionId and SHA-256, then reference both in request.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

# Support both ``python -m tools.package_lambda_model_bundle`` and the
# documented direct-script invocation from the repository root.
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.aws_lambda.model_bundle import ModelBundleError, validate_model_bundle_layout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to package symlinked model artifact: {path}")
        if path.is_file():
            yield path


def _copy_model_sources(models_dir: Path, destination: Path) -> None:
    for name in ("baseline", "experts", "pretrained"):
        source = models_dir / name
        if source.exists():
            for candidate in source.rglob("*"):
                if candidate.is_symlink():
                    raise ValueError(f"Refusing to package symlinked model artifact: {candidate}")
            shutil.copytree(source, destination / name)


def build_bundle(*, models_dir: Path, active_file: Path, output_path: Path) -> dict[str, object]:
    """Package all discoverable shadow models and the global active pointer."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"models directory not found: {models_dir}")
    if not active_file.is_file():
        raise FileNotFoundError(f"active registry file not found: {active_file}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lambda-model-bundle-") as temporary_dir:
        stage_root = Path(temporary_dir) / "bundle"
        stage_models = stage_root / "models"
        stage_models.mkdir(parents=True)
        _copy_model_sources(models_dir, stage_models)

        stage_registry = stage_root / "registry"
        stage_registry.mkdir(parents=True)
        shutil.copy2(active_file, stage_registry / "active_model.yaml")
        try:
            validate_model_bundle_layout(stage_root)
        except ModelBundleError as exc:
            raise ValueError(f"Invalid Lambda model bundle layout: {exc}") from exc

        files = list(_iter_files(stage_root))
        manifest = {
            "schema_version": 1,
            "active_registry_path": "registry/active_model.yaml",
            "files": [
                {
                    "path": path.relative_to(stage_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
        manifest_path = stage_root / "bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        with tarfile.open(output_path, mode="w:gz", compresslevel=9) as archive:
            for path in _iter_files(stage_root):
                arcname = path.relative_to(stage_root).as_posix()
                info = archive.gettarinfo(str(path), arcname=arcname)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)

    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package active + shadow model artifacts for the inference Lambda."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--active-file", type=Path, default=Path("registry/active_model.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(
        models_dir=args.models_dir,
        active_file=args.active_file,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

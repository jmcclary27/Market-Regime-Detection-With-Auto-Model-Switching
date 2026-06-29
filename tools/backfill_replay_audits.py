# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run import PipelineConfig, run_pipeline, setup_logging
from src.reporting.project_metrics import discover_lineage_runs

REQUIRED_ARTIFACTS = ("raw_csv", "features_parquet", "regimes_parquet", "predictions_parquet")


def _resolve_artifact_path(project_root: Path, raw_path: str | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    return candidate if candidate.is_absolute() else project_root / candidate


def _replay_inputs_exist(project_root: Path, lineage: dict[str, object]) -> tuple[bool, list[str]]:
    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, list(REQUIRED_ARTIFACTS)

    missing: list[str] = []
    for label in REQUIRED_ARTIFACTS:
        payload = artifacts.get(label)
        raw_path = payload.get("path") if isinstance(payload, dict) else None
        path = _resolve_artifact_path(project_root, str(raw_path) if raw_path is not None else None)
        if path is None or not path.exists():
            missing.append(label)
    return not missing, missing


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill replay audit summaries from lineage runs.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--run-ts", action="append", default=None, help="Replay one or more specific run_ts values.")
    parser.add_argument("--rewrite-existing", action="store_true", help="Regenerate replay audits even when they already exist.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort on the first replay execution error.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(str(args.project_root)).resolve()
    requested = {str(value) for value in (args.run_ts or [])}
    setup_logging(0)

    runs = discover_lineage_runs(project_root)
    if requested:
        runs = [run for run in runs if run.run_ts in requested]

    if not runs:
        print("No lineage runs matched the replay backfill request.")
        return 0

    succeeded = 0
    failed = 0
    skipped = 0

    for run in runs:
        audit_path = project_root / "artifacts" / "replay" / f"replay_{run.run_ts}.json"
        if audit_path.exists() and not bool(args.rewrite_existing):
            skipped += 1
            print(f"skip existing: {run.run_ts}")
            continue

        replayable, missing = _replay_inputs_exist(project_root, run.lineage)
        if not replayable:
            skipped += 1
            print(f"skip missing inputs: {run.run_ts} missing={missing}")
            continue

        cfg = PipelineConfig(
            project_root=project_root,
            data_dir=project_root / "data",
            run_ts=run.run_ts,
            mode="pipeline",
        )
        try:
            run_pipeline(cfg, replay=True, replay_ts=run.run_ts)
            succeeded += 1
            print(f"replay audit ok: {run.run_ts}")
        except Exception as exc:
            failed += 1
            status = "wrote failed audit" if audit_path.exists() else "replay failed"
            print(f"{status}: {run.run_ts} error={exc!r}")
            if bool(args.stop_on_error):
                break

    print(f"summary: succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

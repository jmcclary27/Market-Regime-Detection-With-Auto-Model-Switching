"""Train and immutably freeze the daily paper-experiment artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.experiment.freeze import FreezeConfig, FreezeError, freeze_experiment


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 date (YYYY-MM-DD)") from exc


def parse_args(argv: list[str] | None = None) -> FreezeConfig:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic daily experiment candidates."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--official-start-date", required=True, type=_date)
    parser.add_argument("--data-cutoff", required=True, type=_date)
    parser.add_argument("--features-path", required=True, type=Path)
    parser.add_argument("--regimes-path", required=True, type=Path)
    parser.add_argument("--feature-manifest-path", required=True, type=Path)
    parser.add_argument("--hmm-artifacts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--publish-s3", action="store_true")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-bundle-key")
    parser.add_argument("--s3-manifest-key")
    args = parser.parse_args(argv)
    return FreezeConfig(
        experiment_id=args.experiment_id,
        official_start_date=args.official_start_date,
        data_cutoff=args.data_cutoff,
        features_path=args.features_path,
        regimes_path=args.regimes_path,
        feature_manifest_path=args.feature_manifest_path,
        hmm_artifacts_dir=args.hmm_artifacts_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        publish_s3=args.publish_s3,
        s3_bucket=args.s3_bucket,
        s3_bundle_key=args.s3_bundle_key,
        s3_manifest_key=args.s3_manifest_key,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = freeze_experiment(parse_args(argv))
    except FreezeError as exc:
        raise SystemExit(f"freeze failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

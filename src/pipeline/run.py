from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


LOG = logging.getLogger("pipeline")


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    run_ts: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def setup_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s, %(message)s",
    )


def step(name: str, fn: Callable[[], None]) -> None:
    LOG.info("Starting step, %s", name)
    try:
        fn()
    except SystemExit as e:
        LOG.exception(
            "SystemExit raised in step, %s (code=%s)",
            name,
            getattr(e, "code", None),
        )
        raise
    except Exception:
        LOG.exception("Step failed, %s", name)
        raise
    LOG.info("Finished step, %s", name)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    default_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("PROJECT_ROOT", str(default_root))).resolve()

    data_dir = Path(os.environ.get("DATA_DIR", str(project_root / "data"))).resolve()

    run_ts = args.run_ts or os.environ.get("RUN_TS") or utc_timestamp()

    return PipelineConfig(
        project_root=project_root,
        data_dir=data_dir,
        run_ts=run_ts,
    )


def latest_raw_file(raw_dir: Path) -> Path:
    candidates = sorted(
        raw_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    return candidates[0]


def run_pipeline(cfg: PipelineConfig) -> None:
    LOG.info("Pipeline run started, run_ts=%s", cfg.run_ts)

    # ---- imports (cheap + explicit) ----
    from src.ingestion.run_ingestion import run as ingest_run
    from src.features.run_features import run as features_run
    from src.regimes.run_regime_detection import run as regimes_run

    # IMPORTANT: use orchestration-friendly wrapper we added
    from src.inference.batch_predict import run_stage as predict_run

    from src.eval.run_evaluator import run as eval_run
    from src.deploy.switcher import run as switch_run

    LOG.info("All entrypoints imported, starting steps...")

    # ---- poll ----
    step("poll", ingest_run)

    # ---- features ----
    features_parquet: Path | None = None

    def _features() -> None:
        nonlocal features_parquet
        raw_latest = latest_raw_file(cfg.data_dir / "raw")
        LOG.info("Using raw input: %s", raw_latest)
        features_parquet, _ = features_run(input_path=raw_latest, timestamp=cfg.run_ts)

    step("features", _features)

    # ---- regimes ----
    regimes_parquet: Path | None = None

    def _regimes() -> None:
        nonlocal regimes_parquet
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")
        regimes_parquet = regimes_run(input_path=features_parquet, timestamp=cfg.run_ts)

    step("regimes", _regimes)

    # ---- predict ----
    predictions_parquet: Path | None = None

    def _predict() -> None:
        nonlocal predictions_parquet
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")
        # batch_predict currently scores from features parquet
        predictions_parquet = predict_run(features_path=features_parquet)

    step("predict", _predict)

    # ---- eval ----
    # (likely next to refactor to accept predictions_parquet + timestamp)
    step("eval", eval_run)

    # ---- switch ----
    step("switch", switch_run)

    LOG.info(
        "Pipeline run completed, run_ts=%s (features=%s, regimes=%s, predictions=%s)",
        cfg.run_ts,
        str(features_parquet) if features_parquet else None,
        str(regimes_parquet) if regimes_parquet else None,
        str(predictions_parquet) if predictions_parquet else None,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full local pipeline")
    p.add_argument(
        "--run-ts",
        default=None,
        help="Optional shared timestamp, e.g. 20260112_141530Z",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    cfg = build_config(args)

    try:
        run_pipeline(cfg)
    except BaseException:
        LOG.exception("Pipeline crashed")
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
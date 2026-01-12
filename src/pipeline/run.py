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
        # A lot of "python -m ..." style scripts call sys.exit()
        LOG.exception("SystemExit raised in step, %s (code=%s)", name, getattr(e, "code", None))
        raise
    except Exception:
        LOG.exception("Step failed, %s", name)
        raise
    LOG.info("Finished step, %s", name)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    default_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("PROJECT_ROOT", str(default_root))).resolve()

    data_dir = Path(
        os.environ.get("DATA_DIR", str(project_root / "data"))
    ).resolve()

    run_ts = args.run_ts or os.environ.get("RUN_TS") or utc_timestamp()

    return PipelineConfig(
        project_root=project_root,
        data_dir=data_dir,
        run_ts=run_ts,
    )


def run_pipeline(cfg: PipelineConfig) -> None:
    LOG.info("Pipeline run started, run_ts=%s", cfg.run_ts)
    LOG.info("All entrypoints imported, starting steps...")

    # ---- PR 1: ingestion ----
    from src.ingestion.run_ingestion import run as ingest_run

    # ---- PR 2: features ----
    from src.features.run_features import run as features_run

    # ---- PR 3: regimes ----
    from src.regimes.run_regime_detection import run as regimes_run

    # ---- PR 5: inference ----
    from src.inference.batch_predict import run as predict_run

    # ---- PR 6: evaluation ----
    from src.eval.run_evaluator import run as eval_run

    # ---- PR 8: switching ----
    from src.deploy.switcher import run as switch_run
    
    LOG.info("All entrypoints imported, starting steps...")

    LOG.info("DEBUG: about to call step('poll', ingest_run)")

    step("poll", ingest_run)
    step("features", features_run)
    step("regimes", regimes_run)
    step("predict", predict_run)
    step("eval", eval_run)
    step("switch", switch_run)

    LOG.info("Pipeline run completed, run_ts=%s", cfg.run_ts)


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


if __name__ == "__main__":
    raise SystemExit(main())
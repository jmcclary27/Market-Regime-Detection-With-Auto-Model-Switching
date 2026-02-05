# src/pipeline/run.py
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from src.monitoring import metrics as m
from src.regimes.hmm import compute_hmm_diagnostics

LOG = logging.getLogger("pipeline")


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    run_ts: str


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")


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
    from src.deploy.switcher import run as switch_run
    from src.eval.run_evaluator import run as eval_run
    from src.features.run_features import run as features_run

    # IMPORTANT: use orchestration-friendly wrapper we added
    from src.inference.batch_predict import run_stage as predict_run
    from src.ingestion.run_ingestion import run as ingest_run
    from src.regimes.run_regime_detection import run as regimes_run

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
    
        # ---- regime diagnostics (PR11) ----
    def _regime_diagnostics() -> None:
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")

        # Read features to compute diagnostics on the same inputs regimes used
        df_features = pd.read_parquet(features_parquet)

        diag = compute_hmm_diagnostics(df_features, cfg={}, run_ts=cfg.run_ts)  # replace cfg={} with your loaded settings dict if available

        # Save JSON artifact
        out_dir = cfg.project_root / "artifacts" / "regimes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"diagnostics_{cfg.run_ts}.json"
        out_path.write_text(json.dumps(diag.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

        # Save quick-look CSV artifacts
        tc = pd.DataFrame(diag.transition_counts)
        tp = pd.DataFrame(diag.transition_probs)
        tc.to_csv(out_dir / f"transition_counts_{cfg.run_ts}.csv", index=False)
        tp.to_csv(out_dir / f"transition_probs_{cfg.run_ts}.csv", index=False)

        # Log to MLflow if active
        try:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.log_metric(m.REGIME_ENTROPY, diag.regime_entropy)
                mlflow.log_metric(m.AVG_REGIME_DURATION, diag.avg_regime_duration)
                mlflow.log_metric(m.SWITCHES_PER_1000_STEPS, diag.switches_per_1000_steps)
                for k, v in enumerate(diag.pct_time_regime):
                    mlflow.log_metric(f"{m.PCT_TIME_REGIME_PREFIX}{k}", float(v))

                # log artifacts if you want them attached to the run
                mlflow.log_artifact(str(out_path))
                mlflow.log_artifact(str(out_dir / f"transition_counts_{cfg.run_ts}.csv"))
                mlflow.log_artifact(str(out_dir / f"transition_probs_{cfg.run_ts}.csv"))
        except Exception:
            # Telemetry should never crash the pipeline
            LOG.exception("Regime diagnostics MLflow logging failed")

    step("regime_diagnostics", _regime_diagnostics)

    # ---- predict ----
    predictions_parquet: Path | None = None

    def _predict() -> None:
        nonlocal predictions_parquet
        if regimes_parquet is None:
            raise RuntimeError("regimes_parquet not set")
        predictions_parquet = predict_run(features_path=regimes_parquet)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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


def main(argv: list[str] | None = None) -> int:
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

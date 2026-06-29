# src/pipeline/run.py
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from src.config.load_config import load_config
from src.ingestion.quality import audit_raw_bars, default_audit_output
from src.monitoring import metrics as m
from src.monitoring.replay_audit import (
    build_replay_audit,
    default_replay_output,
    write_replay_audit,
)
from src.pipeline.telemetry import (
    PipelineRunRecorder,
    pipeline_run_summary_path,
    write_pipeline_run_summary,
)
from src.regimes.hmm import compute_hmm_diagnostics

LOG = logging.getLogger("pipeline")

PipelineMode = Literal["pipeline", "backtest"]


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    run_ts: str
    mode: PipelineMode


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")


def setup_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s, %(message)s",
    )


def step(name: str, fn: Callable[[], None], *, recorder: PipelineRunRecorder | None = None) -> None:
    LOG.info("Starting step, %s", name)
    if recorder is not None:
        recorder.start_step(name)
    try:
        fn()
    except SystemExit as exc:
        LOG.exception("SystemExit raised in step, %s (code=%s)", name, getattr(exc, "code", None))
        if recorder is not None:
            recorder.finish_step(name, status="failed", error=repr(exc))
        raise
    except Exception as exc:
        LOG.exception("Step failed, %s", name)
        if recorder is not None:
            recorder.finish_step(name, status="failed", error=repr(exc))
        raise
    if recorder is not None:
        recorder.finish_step(name, status="completed")
    LOG.info("Finished step, %s", name)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    default_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("PROJECT_ROOT", str(default_root))).resolve()
    data_dir = Path(os.environ.get("DATA_DIR", str(project_root / "data"))).resolve()
    run_ts = args.run_ts or os.environ.get("RUN_TS") or utc_timestamp()
    mode: PipelineMode = args.mode
    return PipelineConfig(project_root=project_root, data_dir=data_dir, run_ts=run_ts, mode=mode)


def latest_raw_file(raw_dir: Path) -> Path:
    candidates = sorted(
        (path for path in raw_dir.glob("*.csv") if path.name != "latest.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    return candidates[0]


# -------------------------
# Replay helpers
# -------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_lineage(project_root: Path, run_ts: str) -> dict[str, Any]:
    path = project_root / "artifacts" / "lineage" / f"lineage_{run_ts}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing lineage file: {path}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def run_ts_to_int(run_ts: str) -> int:
    digits = "".join(ch for ch in run_ts if ch.isdigit())
    return int(digits[:14]) if len(digits) >= 14 else int(digits)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _run_record_for_output(project_root: Path, raw_path: Path) -> dict[str, Any] | None:
    runs_dir = project_root / "data" / "runs"
    if not runs_dir.exists():
        return None

    candidates = {str(raw_path)}
    try:
        candidates.add(raw_path.relative_to(project_root).as_posix())
    except ValueError:
        pass
    candidates.add(raw_path.as_posix())

    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        output_path = str(payload.get("output_path", ""))
        latest_path = str(payload.get("latest_path", ""))
        if output_path in candidates or latest_path in candidates:
            return payload
    return None


# -------------------------
# Pipeline
# -------------------------
def run_pipeline(
    cfg: PipelineConfig, *, replay: bool = False, replay_ts: str | None = None
) -> None:
    LOG.info("Pipeline run started, run_ts=%s mode=%s replay=%s", cfg.run_ts, cfg.mode, replay)

    replay_subject_run_ts = replay_ts or cfg.run_ts
    replay_stem = f"replay_{replay_subject_run_ts}"
    lineage: dict[str, Any] | None = None
    if replay:
        lineage = _load_lineage(cfg.project_root, replay_subject_run_ts)
        LOG.info("Loaded lineage for replay: %s", replay_subject_run_ts)

    mlflow_obj: Any | None = None
    created_run = False
    recorder = PipelineRunRecorder(
        run_ts=cfg.run_ts,
        mode=cfg.mode,
        replay=replay,
        replay_ts=replay_subject_run_ts if replay else None,
    )
    try:
        import mlflow as _mlflow

        mlflow_obj = _mlflow
        if mlflow_obj.active_run() is None:
            exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pipeline")
            mlflow_obj.set_experiment(exp_name)
            mlflow_obj.start_run(run_name=f"pipeline_{cfg.run_ts}")
            created_run = True

        mlflow_obj.set_tag("run_ts", cfg.run_ts)
        mlflow_obj.set_tag("component", "pipeline")
        mlflow_obj.set_tag("mode", cfg.mode)
        mlflow_obj.set_tag("replay", str(bool(replay)))
        if replay:
            mlflow_obj.set_tag("replay_ts", replay_subject_run_ts)
    except Exception:
        LOG.exception("MLflow setup failed, continuing without MLflow")

    from src.features.run_features import run as features_run
    from src.inference.batch_predict import run_stage as predict_run
    from src.ingestion.run_ingestion import run as ingest_run
    from src.regimes.run_regime_detection import run as regimes_run

    LOG.info("All entrypoints imported, starting steps...")

    raw_latest: Path | None = None
    features_parquet: Path | None = None
    features_manifest: Path | None = None
    regimes_parquet: Path | None = None
    predictions_parquet: Path | None = None
    data_quality_audit_json: Path | None = None
    pipeline_run_json: Path | None = None
    wf_portfolio_metrics_parquet: Path | None = None
    promotion_decision_json: Path | None = None
    replay_artifacts: dict[str, Path] = {}

    def _write_lineage_snapshot() -> Path:
        if raw_latest is None:
            raise RuntimeError("raw_latest not set")
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")
        if features_manifest is None:
            raise RuntimeError("features_manifest not set")
        if regimes_parquet is None:
            raise RuntimeError("regimes_parquet not set")
        if predictions_parquet is None:
            raise RuntimeError("predictions_parquet not set")

        config_path = cfg.project_root / "src" / "config" / "settings.yaml"
        config_text = config_path.read_text(encoding="utf-8")

        from src.data.lineage import write_run_lineage

        artifacts: dict[str, Path] = {
            "raw_csv": raw_latest,
            "features_parquet": features_parquet,
            "features_manifest": features_manifest,
            "regimes_parquet": regimes_parquet,
            "predictions_parquet": predictions_parquet,
        }
        if data_quality_audit_json is not None and data_quality_audit_json.exists():
            artifacts["data_quality_audit_json"] = data_quality_audit_json
        if wf_portfolio_metrics_parquet is not None and wf_portfolio_metrics_parquet.exists():
            artifacts["walkforward_portfolio_metrics_parquet"] = wf_portfolio_metrics_parquet
        if promotion_decision_json is not None and promotion_decision_json.exists():
            artifacts["promotion_decision_json"] = promotion_decision_json
        if pipeline_run_json is not None and pipeline_run_json.exists():
            artifacts["pipeline_run_json"] = pipeline_run_json

        params = {
            "mode": cfg.mode,
            "market_symbols": load_config().get("market", {}).get("symbols", []),
        }

        out = write_run_lineage(
            project_root=cfg.project_root,
            run_ts=cfg.run_ts,
            config_text=config_text,
            artifacts=artifacts,
            params=params,
        )

        try:
            if mlflow_obj is not None and mlflow_obj.active_run() is not None:
                mlflow_obj.log_artifact(str(out))
                mlflow_obj.set_tag("lineage_path", str(out))
        except Exception:
            LOG.exception("Lineage MLflow logging failed")

        return out

    run_status = "completed"
    run_error: str | None = None
    try:
        if not replay:
            step("poll", ingest_run, recorder=recorder)
        else:
            LOG.info("Replay enabled, skipping poll step")

        def _features() -> None:
            nonlocal raw_latest, features_parquet, features_manifest

            if replay:
                assert lineage is not None
                raw_latest = _resolve_path(cfg.project_root, lineage["artifacts"]["raw_csv"]["path"])
                LOG.info("Replay raw input: %s", raw_latest)
            else:
                raw_latest = latest_raw_file(cfg.data_dir / "raw")
                LOG.info("Using raw input: %s", raw_latest)

            features_parquet, features_manifest = features_run(
                input_path=raw_latest,
                timestamp=cfg.run_ts,
                output_stem=replay_stem if replay else None,
                write_latest=not replay,
            )
            if replay:
                replay_artifacts["features_parquet"] = features_parquet
                replay_artifacts["features_manifest"] = features_manifest

        step("features", _features, recorder=recorder)

        if not replay:

            def _data_quality() -> None:
                nonlocal data_quality_audit_json
                if raw_latest is None:
                    raise RuntimeError("raw_latest not set")

                record = _run_record_for_output(cfg.project_root, raw_latest)
                settings = load_config()
                symbols = cast(list[str], settings.get("market", {}).get("symbols", []))
                interval = cast(str | None, settings.get("market", {}).get("frequency"))
                if record is not None and record.get("interval") not in (None, ""):
                    interval = str(record.get("interval"))

                output_path = default_audit_output(cfg.project_root, cfg.run_ts)
                provider_attempt_count = None
                if record is not None and record.get("provider_attempt_count") not in (None, ""):
                    provider_attempt_count = int(record["provider_attempt_count"])

                audit_raw_bars(
                    raw_path=raw_latest,
                    run_ts=cfg.run_ts,
                    finished_at_utc=cast(str | None, record.get("finished_at_utc")) if record else None,
                    symbols=cast(list[str] | None, record.get("symbols")) if record else symbols,
                    interval=interval,
                    provider_failure_count=int(record.get("provider_failure_count", 0)) if record else 0,
                    provider_attempt_count=provider_attempt_count,
                    run_type=cast(str | None, record.get("run_type")) if record else "pipeline_poll",
                    output_path=output_path,
                )
                data_quality_audit_json = output_path

            step("data_quality", _data_quality, recorder=recorder)

        def _regimes() -> None:
            nonlocal regimes_parquet
            if features_parquet is None:
                raise RuntimeError("features_parquet not set")
            regimes_parquet = regimes_run(
                input_path=features_parquet,
                timestamp=cfg.run_ts,
                output_stem=replay_stem if replay else None,
                write_latest=not replay,
            )
            if replay:
                replay_artifacts["regimes_parquet"] = regimes_parquet

        step("regimes", _regimes, recorder=recorder)

        if not replay:

            def _regime_diagnostics() -> None:
                if features_parquet is None:
                    raise RuntimeError("features_parquet not set")

                settings = load_config()
                df_features = pd.read_parquet(features_parquet)
                diag = compute_hmm_diagnostics(df_features, cfg=settings, run_ts=cfg.run_ts)

                out_dir = cfg.project_root / "artifacts" / "regimes"
                out_dir.mkdir(parents=True, exist_ok=True)

                diag_path = out_dir / f"diagnostics_{cfg.run_ts}.json"
                diag_path.write_text(json.dumps(diag.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

                pd.DataFrame(diag.transition_counts).to_csv(
                    out_dir / f"transition_counts_{cfg.run_ts}.csv", index=False
                )
                pd.DataFrame(diag.transition_probs).to_csv(
                    out_dir / f"transition_probs_{cfg.run_ts}.csv", index=False
                )

                try:
                    if mlflow_obj is not None and mlflow_obj.active_run() is not None:
                        mlflow_obj.log_metric(m.REGIME_ENTROPY, diag.regime_entropy)
                        mlflow_obj.log_metric(m.AVG_REGIME_DURATION, diag.avg_regime_duration)
                        mlflow_obj.log_metric(m.SWITCHES_PER_1000_STEPS, diag.switches_per_1000_steps)
                        for idx, value in enumerate(diag.pct_time_regime):
                            mlflow_obj.log_metric(f"{m.PCT_TIME_REGIME_PREFIX}{idx}", float(value))
                        mlflow_obj.log_artifact(str(diag_path))
                except Exception:
                    LOG.exception("Regime diagnostics MLflow logging failed")

            step("regime_diagnostics", _regime_diagnostics, recorder=recorder)

        def _predict() -> None:
            nonlocal predictions_parquet
            if regimes_parquet is None:
                raise RuntimeError("regimes_parquet not set")

            recorded_features_path = None
            if replay and lineage is not None:
                recorded_features_path = str(lineage["artifacts"]["regimes_parquet"]["path"])

            predictions_parquet = predict_run(
                features_path=regimes_parquet,
                inference_ts=run_ts_to_int(replay_subject_run_ts if replay else cfg.run_ts),
                output_name=(
                    f"predictions_replay_{replay_subject_run_ts}.parquet"
                    if replay
                    else f"predictions_{cfg.run_ts}.parquet"
                ),
                latest_name=None if replay else "latest.parquet",
                run_meta_name=(
                    f"run_replay_{replay_subject_run_ts}.json"
                    if replay
                    else f"run_{cfg.run_ts}.json"
                ),
                record_features_path=recorded_features_path,
            )
            if replay:
                replay_artifacts["predictions_parquet"] = predictions_parquet

        step("predict", _predict, recorder=recorder)

        if replay:
            LOG.info("Replay enabled, skipping lineage step")
        else:

            def _lineage() -> None:
                out = _write_lineage_snapshot()
                LOG.info("Wrote lineage: %s", out)

            step("lineage", _lineage, recorder=recorder)

        if replay:

            def _replay_verify() -> None:
                assert lineage is not None
                summary = build_replay_audit(
                    project_root=cfg.project_root,
                    run_ts=replay_subject_run_ts,
                    lineage=lineage,
                    replay_artifacts=replay_artifacts,
                )
                replay_path = write_replay_audit(
                    default_replay_output(cfg.project_root, replay_subject_run_ts),
                    summary,
                )
                LOG.info("Replay audit written: %s", replay_path)
                if summary["status"] != "passed":
                    raise AssertionError(
                        f"Replay audit failed for {replay_subject_run_ts}: {summary['failure_breakdown']}"
                    )

            step("replay_verify", _replay_verify, recorder=recorder)

        if cfg.mode == "backtest":

            def _backtest() -> None:
                if features_parquet is None:
                    raise RuntimeError("features_parquet not set")
                if predictions_parquet is None:
                    raise RuntimeError("predictions_parquet not set")

                from src.backtest.adapters import signals_spy_from_predictions
                from src.backtest.engine import BacktestConfig, run_backtest
                from src.backtest.metrics import compute_portfolio_metrics

                df_features = pd.read_parquet(features_parquet)
                prices = (
                    df_features[["timestamp", "close_x"]]
                    .rename(columns={"close_x": "SPY"})
                    .set_index("timestamp")
                    .sort_index()
                )

                df_preds = pd.read_parquet(predictions_parquet)
                fallback = os.getenv("BACKTEST_MODEL_NAME", "baseline")
                signals = signals_spy_from_predictions(
                    df_preds, features=df_features, fallback_model_name=fallback
                )
                signals = signals.reindex(prices.index).fillna(0.0)

                bt_cfg = BacktestConfig(
                    initial_cash=float(os.getenv("BACKTEST_INITIAL_CASH", "100000")),
                    fee_bps=float(os.getenv("BACKTEST_FEE_BPS", "0")),
                    spread_bps=float(os.getenv("BACKTEST_SPREAD_BPS", "0")),
                    slippage_bps=float(os.getenv("BACKTEST_SLIPPAGE_BPS", "0")),
                    seed=int(os.getenv("BACKTEST_SEED", "0")),
                )

                res = run_backtest(prices=prices, signals=signals, cfg=bt_cfg)

                out_dir = cfg.project_root / "artifacts" / "backtest"
                out_dir.mkdir(parents=True, exist_ok=True)
                results_path = out_dir / f"results_{cfg.run_ts}.parquet"
                trades_path = out_dir / f"trades_{cfg.run_ts}.parquet"

                results_df = pd.DataFrame(
                    {
                        "equity": res.equity_curve,
                        "returns_gross": res.returns_gross,
                        "returns_net": res.returns_net,
                    }
                )
                results_df.to_parquet(results_path, index=True)
                res.trades.to_parquet(trades_path, index=False)

                m_port = compute_portfolio_metrics(
                    results_df=results_df,
                    trades_df=res.trades,
                    periods_per_year=252,
                )

                try:
                    if mlflow_obj is not None and mlflow_obj.active_run() is not None:
                        mlflow_obj.log_artifact(str(results_path))
                        mlflow_obj.log_artifact(str(trades_path))
                        mlflow_obj.set_tag("backtest_asset", "SPY")
                        mlflow_obj.set_tag("backtest_model_fallback", fallback)
                        mlflow_obj.log_metric("bt_cagr", m_port.cagr)
                        mlflow_obj.log_metric("bt_sharpe", m_port.sharpe)
                        mlflow_obj.log_metric("bt_sortino", m_port.sortino)
                        mlflow_obj.log_metric("bt_max_drawdown", m_port.max_drawdown)
                        mlflow_obj.log_metric("bt_turnover", m_port.turnover)
                        mlflow_obj.log_metric("bt_profit_factor", m_port.profit_factor)
                except Exception:
                    LOG.exception("Backtest MLflow logging failed")

            step("backtest", _backtest, recorder=recorder)

        if cfg.mode == "pipeline" and not replay:

            def _eval() -> None:
                os.environ["RUN_TS"] = cfg.run_ts
                os.environ["EVAL_RUN_TS"] = cfg.run_ts

                from src.eval import run_evaluator as re

                argv = [
                    "--features",
                    "data/features/latest.parquet",
                    "--regimes",
                    "data/regimes/latest.parquet",
                    "--predictions",
                    "data/predictions/latest.parquet",
                    "--run-ts",
                    cfg.run_ts,
                    "--walk-forward",
                    "--wf-train",
                    os.getenv("EVAL_WF_TRAIN", str(252 * 2)),
                    "--wf-val",
                    os.getenv("EVAL_WF_VAL", str(252 // 2)),
                    "--wf-test",
                    os.getenv("EVAL_WF_TEST", str(252 // 2)),
                    "--wf-step",
                    os.getenv("EVAL_WF_STEP", str(252 // 2)),
                ]
                if os.getenv("EVAL_WF_ANCHORED", "1") == "1":
                    argv.append("--wf-anchored")

                re.main(argv)

            step("eval", _eval, recorder=recorder)

            def _promotion() -> None:
                nonlocal wf_portfolio_metrics_parquet, promotion_decision_json

                wf_portfolio_metrics_parquet = (
                    cfg.project_root / "data" / "walkforward" / f"portfolio_metrics_{cfg.run_ts}.parquet"
                )
                if not wf_portfolio_metrics_parquet.exists():
                    raise FileNotFoundError(
                        "Expected walk-forward portfolio metrics not found: "
                        f"{wf_portfolio_metrics_parquet}. "
                        "Implement PR14 writer in run_evaluator --walk-forward path."
                    )

                from src.models.run_promotion import run_promotion
                from src.registry.registry import ActiveModelRef

                challenger_model_name = os.getenv("PROMOTION_CHALLENGER_MODEL", "elasticnet_v0")
                incumbent_model_name = os.getenv("PROMOTION_INCUMBENT_MODEL", "baseline")

                challenger_ref = ActiveModelRef(
                    model_type=str(os.getenv("PROMOTION_CHALLENGER_TYPE", "pretrained")),
                    model_id=str(os.getenv("PROMOTION_CHALLENGER_ID", challenger_model_name)),
                    version=str(os.getenv("PROMOTION_CHALLENGER_VERSION", "0")),
                    artifact_path=Path(
                        str(
                            os.getenv(
                                "PROMOTION_CHALLENGER_ARTIFACT",
                                "models/pretrained/elasticnet/latest.joblib",
                            )
                        )
                    ),
                    regime=None,
                    metadata_path=None,
                )

                out = run_promotion(
                    challenger_model_name=challenger_model_name,
                    incumbent_model_name=incumbent_model_name,
                    challenger_ref=challenger_ref,
                )

                promotion_decision_json = (
                    cfg.project_root / "data" / "walkforward" / f"promotion_{cfg.run_ts}.json"
                )
                if not promotion_decision_json.exists():
                    raise RuntimeError("promotion decision json was not written as expected")

                try:
                    if mlflow_obj is not None and mlflow_obj.active_run() is not None:
                        mlflow_obj.log_artifact(str(promotion_decision_json))
                        mlflow_obj.log_artifact(str(wf_portfolio_metrics_parquet))
                    _ = out
                except Exception:
                    LOG.exception("Promotion MLflow logging failed")

            step("promotion", _promotion, recorder=recorder)

        LOG.info(
            "Pipeline run completed, run_ts=%s mode=%s (features=%s, regimes=%s, predictions=%s)",
            cfg.run_ts,
            cfg.mode,
            str(features_parquet) if features_parquet else None,
            str(regimes_parquet) if regimes_parquet else None,
            str(predictions_parquet) if predictions_parquet else None,
        )
    except BaseException as exc:
        run_status = "failed"
        run_error = repr(exc)
        raise
    finally:
        pipeline_run_json = pipeline_run_summary_path(cfg.project_root, cfg.run_ts, replay=replay)
        summary = recorder.build_summary(
            status=run_status,
            artifacts={
                "raw_csv": str(raw_latest) if raw_latest is not None else None,
                "features_parquet": str(features_parquet) if features_parquet is not None else None,
                "features_manifest": str(features_manifest) if features_manifest is not None else None,
                "regimes_parquet": str(regimes_parquet) if regimes_parquet is not None else None,
                "predictions_parquet": str(predictions_parquet) if predictions_parquet is not None else None,
                "data_quality_audit_json": str(data_quality_audit_json)
                if data_quality_audit_json is not None
                else None,
                "walkforward_portfolio_metrics_parquet": str(wf_portfolio_metrics_parquet)
                if wf_portfolio_metrics_parquet is not None
                else None,
                "promotion_decision_json": str(promotion_decision_json)
                if promotion_decision_json is not None
                else None,
                "pipeline_run_json": str(pipeline_run_json),
            },
            error=run_error,
        )
        write_pipeline_run_summary(pipeline_run_json, summary)

        if not replay and all(
            path is not None
            for path in (
                raw_latest,
                features_parquet,
                features_manifest,
                regimes_parquet,
                predictions_parquet,
            )
        ):
            try:
                out = _write_lineage_snapshot()
                LOG.info("Updated lineage snapshot: %s", out)
            except Exception:
                LOG.exception("Final lineage update failed")

        try:
            if mlflow_obj is not None and created_run:
                mlflow_obj.end_run()
        except Exception:
            LOG.exception("Failed to end MLflow run cleanly")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full local pipeline")
    parser.add_argument(
        "--run-ts", default=None, help="Optional shared timestamp, e.g. 20260112_141530Z"
    )
    parser.add_argument(
        "--mode",
        default="pipeline",
        choices=("pipeline", "backtest"),
        help="pipeline runs eval+switch, backtest runs backtest step and skips eval+switch.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)"
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Replay a prior run_ts using artifacts/lineage/lineage_<run_ts>.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    cfg = build_config(args)

    if args.replay:
        cfg = PipelineConfig(
            project_root=cfg.project_root,
            data_dir=cfg.data_dir,
            run_ts=str(args.replay),
            mode="pipeline",
        )

    run_pipeline(cfg, replay=bool(args.replay), replay_ts=args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

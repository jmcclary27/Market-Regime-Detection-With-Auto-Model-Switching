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
from src.monitoring import metrics as m
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


def step(name: str, fn: Callable[[], None]) -> None:
    LOG.info("Starting step, %s", name)
    try:
        fn()
    except SystemExit as e:
        LOG.exception("SystemExit raised in step, %s (code=%s)", name, getattr(e, "code", None))
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
    mode: PipelineMode = args.mode
    return PipelineConfig(project_root=project_root, data_dir=data_dir, run_ts=run_ts, mode=mode)


def latest_raw_file(raw_dir: Path) -> Path:
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
    p = project_root / "artifacts" / "lineage" / f"lineage_{run_ts}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing lineage file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def _resolve_path(project_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp)


def _assert_sha256(path: Path, expected: str, *, label: str) -> None:
    got = _sha256_file(path)
    if got != expected:
        raise AssertionError(
            f"Replay mismatch, {label} sha256 differs. expected={expected} got={got} path={path}"
        )


def run_ts_to_int(run_ts: str) -> int:
    # "20260207_123456Z" -> 20260207123456
    digits = "".join(ch for ch in run_ts if ch.isdigit())
    # keep it bounded but stable
    return int(digits[:14]) if len(digits) >= 14 else int(digits)


def _semantic_pred_compare(old_p: Path, new_p: Path) -> None:
    a = pd.read_parquet(old_p)
    b = pd.read_parquet(new_p)

    key = ["row_id", "model_name", "model_source", "is_active"]
    need = key + ["y_pred"]
    for c in need:
        if c not in a.columns or c not in b.columns:
            raise AssertionError(f"Replay compare missing column {c}")

    a2 = a[need].sort_values(key, kind="mergesort").reset_index(drop=True)
    b2 = b[need].sort_values(key, kind="mergesort").reset_index(drop=True)

    if not a2[key].equals(b2[key]):
        raise AssertionError(
            "Replay compare failed, prediction keys differ (row_id/model_name/...)"
        )

    diff = (a2["y_pred"] - b2["y_pred"]).abs().max()
    if pd.isna(diff):
        raise AssertionError("Replay compare failed, diff is NaN")
    if float(diff) > 1e-12:
        raise AssertionError(f"Replay compare failed, max |y_pred diff| = {diff}")


# -------------------------
# Pipeline
# -------------------------
def run_pipeline(
    cfg: PipelineConfig, *, replay: bool = False, replay_ts: str | None = None
) -> None:
    LOG.info("Pipeline run started, run_ts=%s mode=%s replay=%s", cfg.run_ts, cfg.mode, replay)

    lineage: dict[str, Any] | None = None
    if replay:
        if replay_ts is None:
            replay_ts = cfg.run_ts
        lineage = _load_lineage(cfg.project_root, replay_ts)
        LOG.info("Loaded lineage for replay: %s", replay_ts)

    mlflow_obj: Any | None = None
    created_run = False
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
        if replay and replay_ts is not None:
            mlflow_obj.set_tag("replay_ts", str(replay_ts))
    except Exception:
        LOG.exception("MLflow setup failed, continuing without MLflow")

    # ---- imports ----
    from src.deploy.switcher import run as switch_run
    from src.features.run_features import run as features_run
    from src.inference.batch_predict import run_stage as predict_run
    from src.ingestion.run_ingestion import run as ingest_run
    from src.regimes.run_regime_detection import run as regimes_run

    LOG.info("All entrypoints imported, starting steps...")

    # Track inputs/outputs for lineage
    raw_latest: Path | None = None
    features_parquet: Path | None = None
    features_manifest: Path | None = None
    regimes_parquet: Path | None = None
    predictions_parquet: Path | None = None
    wf_portfolio_metrics_parquet: Path | None = None
    promotion_decision_json: Path | None = None

    # ---- poll ----
    if not replay:
        step("poll", ingest_run)
    else:
        LOG.info("Replay enabled, skipping poll step")

    # ---- features ----
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
            input_path=raw_latest, timestamp=cfg.run_ts
        )

    step("features", _features)

    # ---- regimes ----
    def _regimes() -> None:
        nonlocal regimes_parquet
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")
        regimes_parquet = regimes_run(input_path=features_parquet, timestamp=cfg.run_ts)

    step("regimes", _regimes)

    # ---- regime diagnostics ----
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

                for k, v in enumerate(diag.pct_time_regime):
                    mlflow_obj.log_metric(f"{m.PCT_TIME_REGIME_PREFIX}{k}", float(v))

                mlflow_obj.log_artifact(str(diag_path))
        except Exception:
            LOG.exception("Regime diagnostics MLflow logging failed")

    step("regime_diagnostics", _regime_diagnostics)

    # ---- predict ----
    def _predict() -> None:
        nonlocal predictions_parquet
        if regimes_parquet is None:
            raise RuntimeError("regimes_parquet not set")

        predictions_parquet = predict_run(
            features_path=regimes_parquet,
            inference_ts=run_ts_to_int(cfg.run_ts),
        )

        # Ensure deterministic naming in data/predictions for this pipeline run_ts
        out_dir = cfg.project_root / "data" / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Always write a fresh copy (even if predict_run already wrote something)
        dfp = pd.read_parquet(predictions_parquet)

        if replay:
            # IMPORTANT: do NOT overwrite the recorded predictions_<run_ts>.parquet during replay
            dst = out_dir / f"predictions_replay_{cfg.run_ts}.parquet"
            dfp.to_parquet(dst, index=False)
            # also do not touch latest.parquet during replay
        else:
            dst = out_dir / f"predictions_{cfg.run_ts}.parquet"
            dfp.to_parquet(dst, index=False)
            dfp.to_parquet(out_dir / "latest.parquet", index=False)

        predictions_parquet = dst

    step("predict", _predict)

    # ---- lineage ----
    def _lineage() -> None:
        nonlocal features_manifest

        if raw_latest is None:
            raise RuntimeError("raw_latest not set")
        if features_parquet is None:
            raise RuntimeError("features_parquet not set")
        if regimes_parquet is None:
            raise RuntimeError("regimes_parquet not set")
        if predictions_parquet is None:
            raise RuntimeError("predictions_parquet not set")

        config_path = cfg.project_root / "src" / "config" / "settings.yaml"
        config_text = config_path.read_text(encoding="utf-8")

        if features_manifest is None:
            features_manifest = features_parquet.with_suffix(".manifest.json")

        from src.data.lineage import write_run_lineage

        artifacts = {
            "raw_csv": raw_latest,
            "features_parquet": features_parquet,
            "features_manifest": features_manifest,
            "regimes_parquet": regimes_parquet,
            # in non-replay mode this is the canonical predictions_<run_ts>.parquet
            "predictions_parquet": predictions_parquet,
        }
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

        LOG.info("Wrote lineage: %s", out)

    if replay:
        LOG.info("Replay enabled, skipping lineage step")
    else:
        step("lineage", _lineage)

    # ---- replay verification ----
    if replay:

        def _replay_verify() -> None:
            assert lineage is not None

            # verify recorded artifacts are still the same bytes
            # (these should NOT be rewritten during replay)
            for label in (
                "raw_csv",
                "features_parquet",
                "features_manifest",
                "regimes_parquet",
                "predictions_parquet",
            ):
                rec = lineage["artifacts"][label]
                p = _resolve_path(cfg.project_root, rec["path"])
                _assert_sha256(p, rec["sha256"], label=label)

            # semantic compare predictions: recorded predictions vs newly produced replay predictions
            old_preds = _resolve_path(
                cfg.project_root, lineage["artifacts"]["predictions_parquet"]["path"]
            )
            new_preds = predictions_parquet
            if new_preds is None:
                raise RuntimeError("predictions_parquet not set")

            # replay output must match recorded bytes too
            _assert_sha256(
                new_preds,
                lineage["artifacts"]["predictions_parquet"]["sha256"],
                label="predictions_parquet(replay_output)",
            )

            _semantic_pred_compare(old_preds, new_preds)

            LOG.info("Replay verification passed")

        step("replay_verify", _replay_verify)

    # ---- backtest ----
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

        step("backtest", _backtest)

    # ---- eval ----
    if cfg.mode == "pipeline":

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

        step("eval", _eval)
        
    if cfg.mode == "pipeline" and not replay:

        def _promotion() -> None:
            nonlocal wf_portfolio_metrics_parquet, promotion_decision_json

            # Convention: evaluator writes this parquet
            wf_portfolio_metrics_parquet = (
                cfg.project_root / "data" / "walkforward" / f"portfolio_metrics_{cfg.run_ts}.parquet"
            )
            if not wf_portfolio_metrics_parquet.exists():
                raise FileNotFoundError(
                    "Expected walk-forward portfolio metrics not found: "
                    f"{wf_portfolio_metrics_parquet}. "
                    "Implement PR14 writer in run_evaluator --walk-forward path."
                )

            # Run promotion decision + write artifact (and update active pointer if promoted)
            from src.models.run_promotion import run_promotion
            from src.registry.registry import read_active, ActiveModelRef

            incumbent = read_active()  # current active pointer

            challenger_model_name = os.getenv("PROMOTION_CHALLENGER_MODEL", "elasticnet_v0")
            incumbent_model_name = os.getenv("PROMOTION_INCUMBENT_MODEL", "baseline")

            # challenger_ref is what you'd point to if promoted.
            # In practice you’ll build this from your model zoo registry,
            # but for now wire it from env until PR14 registry lands.
            challenger_ref = ActiveModelRef(
                model_type=str(os.getenv("PROMOTION_CHALLENGER_TYPE", "pretrained")),
                model_id=str(os.getenv("PROMOTION_CHALLENGER_ID", "elasticnet_v0")),
                version=str(os.getenv("PROMOTION_CHALLENGER_VERSION", "0")),
                artifact_path=Path(str(os.getenv("PROMOTION_CHALLENGER_ARTIFACT", "models/pretrained/elasticnet/latest.joblib"))),
                regime=None,
                metadata_path=None,
            )

            out = run_promotion(
                challenger_model_name=challenger_model_name,
                incumbent_model_name=incumbent_model_name,
                challenger_ref=challenger_ref,
            )

            promotion_decision_json = cfg.project_root / "data" / "walkforward" / f"promotion_{cfg.run_ts}.json"
            if not promotion_decision_json.exists():
                # run_promotion writes it, so this is a sanity check
                raise RuntimeError("promotion decision json was not written as expected")

            # Optional MLflow artifact logging
            try:
                if mlflow_obj is not None and mlflow_obj.active_run() is not None:
                    mlflow_obj.log_artifact(str(promotion_decision_json))
                    mlflow_obj.log_artifact(str(wf_portfolio_metrics_parquet))
            except Exception:
                LOG.exception("Promotion MLflow logging failed")

        step("promotion", _promotion)

    # ---- switch ----
    if cfg.mode == "pipeline":
        if replay:
            LOG.info("Replay enabled, skipping switch step")
        else:
            step("switch", switch_run)

    if (not replay) and cfg.mode == "pipeline":

        def _lineage_update() -> None:
            if raw_latest is None or features_parquet is None or regimes_parquet is None or predictions_parquet is None:
                raise RuntimeError("base artifacts not set for lineage update")

            config_path = cfg.project_root / "src" / "config" / "settings.yaml"
            config_text = config_path.read_text(encoding="utf-8")

            if features_manifest is None:
                raise RuntimeError("features_manifest not set")

            from src.data.lineage import write_run_lineage

            artifacts = {
                "raw_csv": raw_latest,
                "features_parquet": features_parquet,
                "features_manifest": features_manifest,
                "regimes_parquet": regimes_parquet,
                "predictions_parquet": predictions_parquet,
            }

            if wf_portfolio_metrics_parquet is not None and wf_portfolio_metrics_parquet.exists():
                artifacts["walkforward_portfolio_metrics_parquet"] = wf_portfolio_metrics_parquet
            if promotion_decision_json is not None and promotion_decision_json.exists():
                artifacts["promotion_decision_json"] = promotion_decision_json

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
                LOG.exception("Lineage update MLflow logging failed")

            LOG.info("Updated lineage with eval/promotion artifacts: %s", out)

        step("lineage_update", _lineage_update)
    
    LOG.info(
        "Pipeline run completed, run_ts=%s mode=%s (features=%s, regimes=%s, predictions=%s)",
        cfg.run_ts,
        cfg.mode,
        str(features_parquet) if features_parquet else None,
        str(regimes_parquet) if regimes_parquet else None,
        str(predictions_parquet) if predictions_parquet else None,
    )

    try:
        if mlflow_obj is not None and created_run:
            mlflow_obj.end_run()
    except Exception:
        LOG.exception("Failed to end MLflow run cleanly")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full local pipeline")
    p.add_argument(
        "--run-ts", default=None, help="Optional shared timestamp, e.g. 20260112_141530Z"
    )
    p.add_argument(
        "--mode",
        default="pipeline",
        choices=("pipeline", "backtest"),
        help="pipeline runs eval+switch, backtest runs backtest step and skips eval+switch.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)"
    )
    p.add_argument(
        "--replay",
        default=None,
        help="Replay a prior run_ts using artifacts/lineage/lineage_<run_ts>.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    cfg = build_config(args)

    if args.replay:
        # run_ts becomes the *replayed* run id, so all produced artifacts are tagged with it
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

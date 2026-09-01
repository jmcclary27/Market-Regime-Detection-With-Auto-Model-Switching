from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.reporting.project_metrics import (
    discover_lineage_runs,
    generate_project_metrics_report,
    resolve_subject_run_ts,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_scorecard_artifacts(
    root: Path, *, run_ts: str, active_rmse: float, baseline_rmse: float
) -> None:
    scorecards_dir = root / "data" / "scorecards"
    scorecards_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "scope": [
                "overall",
                "overall",
                "overall",
                "regime",
                "regime",
                "regime",
                "regime",
                "regime",
                "regime",
            ],
            "regime": [
                None,
                None,
                None,
                "bullish",
                "bullish",
                "bullish",
                "bearish",
                "bearish",
                "bearish",
            ],
            "model_name": [
                "active",
                "baseline",
                "expert_lightgbm_bullish",
                "active",
                "baseline",
                "expert_lightgbm_bullish",
                "active",
                "baseline",
                "expert_lightgbm_bullish",
            ],
            "n": [100, 100, 100, 40, 40, 40, 60, 60, 60],
            "mae": [0.20, 0.40, 0.22, 0.18, 0.45, 0.20, 0.24, 0.36, 0.25],
            "rmse": [
                active_rmse,
                baseline_rmse,
                active_rmse + 0.01,
                active_rmse - 0.01,
                baseline_rmse + 0.02,
                active_rmse,
                active_rmse + 0.01,
                baseline_rmse - 0.03,
                active_rmse + 0.02,
            ],
        }
    )
    df.to_parquet(scorecards_dir / f"scorecard_{run_ts}.parquet", index=False)

    scorecard_json = {
        "timestamp": run_ts,
        "run_ts": run_ts,
        "metrics": ["mae", "rmse"],
        "target": {
            "requested_target_col": "log_return_1",
            "y_true_col": "log_return_1_x",
        },
        "overall": {
            "n": 100,
            "by_model": {
                "active": {"mae": 0.20, "rmse": active_rmse},
                "baseline": {"mae": 0.40, "rmse": baseline_rmse},
                "expert_lightgbm_bullish": {"mae": 0.22, "rmse": active_rmse + 0.01},
            },
            "rank": ["active", "expert_lightgbm_bullish", "baseline"],
        },
        "by_regime": {
            "bearish": {
                "n": 60,
                "by_model": {
                    "active": {"mae": 0.24, "rmse": active_rmse + 0.01},
                    "baseline": {"mae": 0.36, "rmse": baseline_rmse - 0.03},
                    "expert_lightgbm_bullish": {"mae": 0.25, "rmse": active_rmse + 0.02},
                },
                "rank": ["expert_lightgbm_bullish", "active", "baseline"],
            },
            "bullish": {
                "n": 40,
                "by_model": {
                    "active": {"mae": 0.18, "rmse": active_rmse - 0.01},
                    "baseline": {"mae": 0.45, "rmse": baseline_rmse + 0.02},
                    "expert_lightgbm_bullish": {"mae": 0.20, "rmse": active_rmse},
                },
                "rank": ["active", "expert_lightgbm_bullish", "baseline"],
            },
        },
        "notes": {
            "scoring_rule": "lower_is_better",
            "row_id_rule": "features: sort ['timestamp'] then row_id=index; regimes: sort ['timestamp'] then row_id=index",
        },
    }
    _write_json(scorecards_dir / f"scorecard_{run_ts}.json", scorecard_json)


def _write_walkforward_artifacts(
    root: Path,
    *,
    run_ts: str,
    active_sharpe: float,
    baseline_sharpe: float,
) -> None:
    scorecards_dir = root / "data" / "scorecards"
    walkforward_dir = root / "data" / "walkforward"
    scorecards_dir.mkdir(parents=True, exist_ok=True)
    walkforward_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.DataFrame(
        {
            "split_id": ["wf_0001", "wf_0001", "wf_0002", "wf_0002"],
            "scope": ["overall", "overall", "overall", "overall"],
            "regime": [None, None, None, None],
            "model_name": ["active", "baseline", "active", "baseline"],
            "n": [20, 20, 20, 20],
            "mae": [0.20, 0.35, 0.22, 0.37],
            "rmse": [0.30, 0.50, 0.31, 0.52],
        }
    )
    eval_df.to_parquet(scorecards_dir / f"walk_forward_metrics_{run_ts}.parquet", index=False)

    portfolio_df = pd.DataFrame(
        {
            "run_ts": [run_ts] * 6,
            "split_id": ["wf_0001", "wf_0001", "wf_0001", "wf_0002", "wf_0002", "wf_0002"],
            "model_name": [
                "active",
                "baseline",
                "expert_lightgbm_bullish",
                "active",
                "baseline",
                "expert_lightgbm_bullish",
            ],
            "cagr": [0.12, 0.05, 0.10, 0.15, 0.03, 0.11],
            "sharpe": [
                active_sharpe,
                baseline_sharpe,
                active_sharpe - 0.05,
                active_sharpe + 0.20,
                baseline_sharpe - 0.10,
                active_sharpe - 0.01,
            ],
            "sortino": [1.8, 1.0, 1.6, 2.1, 0.9, 1.7],
            "max_drawdown": [-0.05, -0.20, -0.08, -0.04, -0.22, -0.09],
            "turnover": [0.20, 0.60, 0.30, 0.25, 0.62, 0.35],
            "profit_factor": [1.40, 1.10, 1.30, 1.50, 1.00, 1.35],
        }
    )
    portfolio_df.to_parquet(walkforward_dir / f"portfolio_metrics_{run_ts}.parquet", index=False)


def _write_promotion(root: Path, *, run_ts: str, promoted: bool) -> None:
    payload = {
        "run_ts": run_ts,
        "promoted": promoted,
        "pointer_written": promoted,
        "challenger_model_name": "expert_lightgbm_bullish",
        "incumbent_model_name": "baseline",
        "reason": "challenger beats incumbent on sharpe and satisfies max_drawdown guardrail",
        "decision": {
            "promote": promoted,
            "reason": "promotion decision",
            "deltas": {
                "sharpe": 0.42,
                "max_drawdown": 0.12,
            },
        },
        "promotion_guard": {
            "allowed": True,
            "reason": "guard passed",
        },
        "non_promotable_models": ["active", "expert_arima_*"],
    }
    _write_json(root / "data" / "walkforward" / f"promotion_{run_ts}.json", payload)


def _write_regime_diagnostics(root: Path, *, run_ts: str) -> None:
    payload = {
        "run_ts": run_ts,
        "n_regimes": 3,
        "n_steps": 100,
        "n_switches": 12,
        "switches_per_1000_steps": 120.0,
        "avg_regime_duration": 8.3,
        "regime_entropy": 0.73,
        "pct_time_regime": [0.2, 0.3, 0.5],
        "confidence": {
            "mean": 0.91,
            "p10": 0.80,
            "p50": 0.94,
            "p90": 0.99,
            "min": 0.50,
            "max": 1.0,
        },
    }
    _write_json(root / "artifacts" / "regimes" / f"diagnostics_{run_ts}.json", payload)


def _write_data_quality(root: Path, *, run_ts: str, status: str = "ok") -> None:
    payload = {
        "run_ts": run_ts,
        "status": status,
        "interval": "1d",
        "row_count": 100,
        "duplicate_bar_count": 0,
        "duplicate_bar_rate": 0.0,
        "missing_bar_count": 0,
        "missing_bar_rate": 0.0,
        "late_data_count": 0,
        "late_data_rate": 0.0,
        "provider_failure_count": 0,
        "provider_failure_rate": 0.0,
        "stale_bar_p95_seconds": 30.0,
    }
    _write_json(root / "artifacts" / "data_quality" / f"data_quality_{run_ts}.json", payload)


def _write_pipeline_run(root: Path, *, run_ts: str, status: str = "completed") -> None:
    payload = {
        "run_ts": run_ts,
        "mode": "pipeline",
        "replay": False,
        "status": status,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:05:00+00:00",
        "duration_seconds": 300.0,
        "steps": [
            {
                "name": "poll",
                "status": "completed",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "finished_at_utc": "2026-01-01T00:01:00+00:00",
                "duration_seconds": 60.0,
                "error": None,
            },
            {
                "name": "predict",
                "status": "completed" if status == "completed" else "failed",
                "started_at_utc": "2026-01-01T00:03:00+00:00",
                "finished_at_utc": "2026-01-01T00:04:00+00:00",
                "duration_seconds": 60.0,
                "error": None if status == "completed" else "boom",
            },
        ],
        "artifacts": {},
        "error": None if status == "completed" else "RuntimeError('boom')",
    }
    _write_json(root / "artifacts" / "pipeline_runs" / f"pipeline_run_{run_ts}.json", payload)


def _write_replay_audit(root: Path, *, run_ts: str, status: str = "passed") -> None:
    payload = {
        "run_ts": run_ts,
        "status": status,
        "exact_pass": status == "passed",
        "semantic_pass": status == "passed",
        "max_prediction_drift": 0.0 if status == "passed" else 0.02,
        "checked_artifacts": [],
        "failure_breakdown": [] if status == "passed" else [{"kind": "prediction_drift"}],
    }
    _write_json(root / "artifacts" / "replay" / f"replay_{run_ts}.json", payload)


def _write_deployment_history(root: Path) -> None:
    df = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:10:00Z", "2026-01-02T00:10:00Z"],
            "run_ts": ["20260101_000000Z", "20260102_000000Z"],
            "source": ["run_promotion", "switcher"],
            "event_type": ["promoted", "hold"],
            "decision": ["promote", "hold"],
            "active_model_id_before": ["baseline", "expert_lightgbm_bullish"],
            "candidate_model_id": ["expert_lightgbm_bullish", "expert_lightgbm_bullish"],
            "active_model_id_after": ["expert_lightgbm_bullish", "expert_lightgbm_bullish"],
            "window_type": [None, "count"],
            "window_value": [None, 50],
            "n": [None, 100],
            "metric_name": ["sharpe", "rmse"],
            "active_metric_value": [0.9, 0.4],
            "candidate_metric_value": [1.2, 0.3],
            "metric_delta": [0.3, -0.1],
            "active_max_drawdown": [-0.2, None],
            "candidate_max_drawdown": [-0.1, None],
            "promotion_guard_allowed": [True, None],
            "pointer_written": [True, False],
            "reason": ["promoted", "held"],
        }
    )
    deployments_dir = root / "data" / "deployments"
    deployments_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(deployments_dir / "events.parquet", index=False)


def _write_registry_history(root: Path) -> None:
    df = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:10:00Z", "2026-01-03T00:10:00Z"],
            "event_type": ["pointer_update", "pointer_update"],
            "source": ["run_promotion", "manual"],
            "run_ts": ["20260101_000000Z", None],
            "reason": ["promoted", "manual refresh"],
            "previous_model_type": [None, "expert"],
            "previous_model_id": [None, "expert_lightgbm_bullish"],
            "previous_version": [None, "0"],
            "previous_artifact_path": [None, "models/experts/bullish/latest.joblib"],
            "previous_regime": [None, "bullish"],
            "new_model_type": ["expert", "expert"],
            "new_model_id": ["expert_lightgbm_bullish", "expert_lightgbm_sideways"],
            "new_version": ["0", "0"],
            "new_artifact_path": [
                "models/experts/bullish/latest.joblib",
                "models/experts/sideways/latest.joblib",
            ],
            "new_regime": ["bullish", "sideways"],
        }
    )
    registry_dir = root / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(registry_dir / "history.parquet", index=False)


def _write_lineage(
    root: Path,
    *,
    run_ts: str,
    include_promotion_refs: bool = True,
    include_data_quality_ref: bool = False,
    include_pipeline_run_ref: bool = False,
) -> None:
    lineage_dir = root / "artifacts" / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {
        "features_parquet": {"path": f"data/features/{run_ts}.parquet", "sha256": "features"},
        "features_manifest": {
            "path": f"data/features/{run_ts}.manifest.json",
            "sha256": "manifest",
        },
        "regimes_parquet": {"path": f"data/regimes/{run_ts}.parquet", "sha256": "regimes"},
        "predictions_parquet": {
            "path": f"data/predictions/predictions_{run_ts}.parquet",
            "sha256": "preds",
        },
        "raw_csv": {"path": f"data/raw/{run_ts}.csv", "sha256": "raw"},
        "walkforward_portfolio_metrics_parquet": {
            "path": f"data/walkforward/portfolio_metrics_{run_ts}.parquet",
            "sha256": "wf",
        },
    }
    if include_promotion_refs:
        artifacts["promotion_decision_json"] = {
            "path": f"data/walkforward/promotion_{run_ts}.json",
            "sha256": "promotion",
        }
    if include_data_quality_ref:
        artifacts["data_quality_audit_json"] = {
            "path": f"artifacts/data_quality/data_quality_{run_ts}.json",
            "sha256": "dq",
        }
    if include_pipeline_run_ref:
        artifacts["pipeline_run_json"] = {
            "path": f"artifacts/pipeline_runs/pipeline_run_{run_ts}.json",
            "sha256": "pipeline",
        }

    payload = {
        "run_ts": run_ts,
        "git_commit": f"commit-{run_ts}",
        "config_sha256": f"config-{run_ts}",
        "artifacts": artifacts,
        "params": {"mode": "pipeline"},
    }
    _write_json(lineage_dir / f"lineage_{run_ts}.json", payload)


def _write_latest_lineage(root: Path, *, run_ts: str) -> None:
    _write_json(root / "artifacts" / "lineage" / "latest.json", {"run_ts": run_ts})


def _write_live_sim(root: Path) -> None:
    live_dir = root / "data" / "live_sim"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_dir / "account_state.json",
        {
            "cash": 60000.0,
            "position": 50.0,
            "portfolio_value": 120000.0,
            "unrealized_pnl": 20000.0,
            "total_pnl": 20000.0,
            "last_price": 400.0,
            "updated_at": "2026-06-22T19:57:54+00:00",
        },
    )

    equity_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-19T13:53:29+00:00",
                "2026-05-20T13:53:29+00:00",
                "2026-05-21T13:53:29+00:00",
            ],
            "portfolio_value": [100000.0, 110000.0, 120000.0],
            "regime": ["bullish", "sideways", "bearish"],
            "active_model_id": [
                "expert_lightgbm_bullish",
                "expert_lightgbm_sideways",
                "expert_lightgbm_bearish",
            ],
        }
    )
    equity_df.to_parquet(live_dir / "equity_curve.parquet", index=False)

    trades_df = pd.DataFrame(
        {
            "timestamp": ["2026-05-19T13:56:19+00:00", "2026-05-20T13:56:19+00:00"],
            "action": ["BUY", "NONE"],
        }
    )
    trades_df.to_parquet(live_dir / "trades.parquet", index=False)


def _write_test_inventory_files(root: Path) -> None:
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_alpha.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n",
        encoding="utf-8",
    )
    (tests_dir / "test_beta.py").write_text(
        "def test_three():\n    assert True\n",
        encoding="utf-8",
    )


def _write_docs(root: Path) -> None:
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "future_metrics_roadmap.md").write_text(
        "# Future Metrics Roadmap\n\n"
        "| Capability | Status | Unlocked Stats Now | Remaining Work |\n"
        "| --- | --- | --- | --- |\n"
        "| Replay audits | Implemented | Exact replay pass rate, semantic replay pass rate, max prediction drift | Backfill more historical runs as artifacts grow |\n"
        "| Deployment event history | Implemented | Promotion precision, rollback rate, canary completion rate | Add richer rollback survival attribution |\n"
        "| Data quality audit | Implemented | Missing-bar rate, duplicate-bar rate, stale-bar p95, late-data rate | Add provider-specific rule thresholds |\n"
        "| Infrastructure telemetry | Implemented | Pipeline success rate, p50/p95 runtime, per-step failure rate, mean recovery time | Add step-level alert thresholds |\n"
        "| Registry change log | Implemented | Active model tenure and churn | Add richer manual override annotations |\n"
        "| Richer live/paper execution | Partial | Trade counts, portfolio growth and drawdown, decision coverage, and available execution costs | Add external fills, benchmark-relative return, and multi-asset normalized exposure |\n"
        "| Drift monitoring | Planned | None yet | Persist feature, prediction, and regime drift snapshots |\n"
        "| Resource usage collection | Partial | Local managed storage and artifact-size snapshots | Capture CPU, memory, remote storage, and time-series resource snapshots |\n"
        "| Explicit cost accounting | Planned | None yet | Join resource usage with cost models and storage pricing |\n"
        "| Alerting history | Planned | None yet | Persist alert event logs and acknowledgement timings |\n"
        "| Calibration or classification outputs | Planned | None yet | Extend predictions artifacts with class labels and calibrated probabilities |\n"
        "| Manual review workflow | Planned | None yet | Add review audit logs and human override reason codes |\n",
        encoding="utf-8",
    )


def _build_history_fixture(root: Path) -> None:
    _write_docs(root)
    _write_test_inventory_files(root)
    _write_live_sim(root)
    _write_deployment_history(root)
    _write_registry_history(root)

    first_run = "20260101_000000Z"
    second_run = "20260102_000000Z"

    _write_lineage(root, run_ts=first_run, include_promotion_refs=False)
    _write_scorecard_artifacts(root, run_ts=first_run, active_rmse=0.30, baseline_rmse=0.55)
    _write_walkforward_artifacts(root, run_ts=first_run, active_sharpe=1.10, baseline_sharpe=0.80)
    _write_pipeline_run(root, run_ts=first_run, status="failed")

    _write_lineage(
        root,
        run_ts=second_run,
        include_promotion_refs=True,
        include_data_quality_ref=True,
        include_pipeline_run_ref=True,
    )
    _write_scorecard_artifacts(root, run_ts=second_run, active_rmse=0.22, baseline_rmse=0.50)
    _write_walkforward_artifacts(root, run_ts=second_run, active_sharpe=1.60, baseline_sharpe=0.95)
    _write_promotion(root, run_ts=second_run, promoted=True)
    _write_regime_diagnostics(root, run_ts=second_run)
    _write_data_quality(root, run_ts=second_run, status="ok")
    _write_pipeline_run(root, run_ts=second_run, status="completed")
    _write_replay_audit(root, run_ts=second_run, status="passed")

    _write_latest_lineage(root, run_ts=second_run)


def test_discover_lineage_runs_and_latest_resolution(tmp_path: Path) -> None:
    _write_lineage(tmp_path, run_ts="20260101_000000Z")
    _write_lineage(tmp_path, run_ts="20260103_000000Z")

    runs = discover_lineage_runs(tmp_path)
    assert [run.run_ts for run in runs] == ["20260101_000000Z", "20260103_000000Z"]
    assert (
        resolve_subject_run_ts(runs, project_root=tmp_path, subject_run_ts="latest")
        == "20260103_000000Z"
    )

    _write_latest_lineage(tmp_path, run_ts="20260101_000000Z")
    assert (
        resolve_subject_run_ts(runs, project_root=tmp_path, subject_run_ts="latest")
        == "20260101_000000Z"
    )


def test_generate_project_metrics_best_effort_history(tmp_path: Path) -> None:
    _build_history_fixture(tmp_path)

    outputs = generate_project_metrics_report(project_root=tmp_path)
    for path in outputs.values():
        assert path.exists()

    runs_df = pd.read_parquet(outputs["runs_history_parquet"])
    models_df = pd.read_parquet(outputs["models_history_parquet"])
    latest_report = json.loads(outputs["latest_report_json"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["report_manifest_json"].read_text(encoding="utf-8"))
    latest_md = outputs["latest_report_md"].read_text(encoding="utf-8")

    assert list(runs_df["run_ts"]) == ["20260101_000000Z", "20260102_000000Z"]
    assert len(models_df) == 6

    first_row = runs_df[runs_df["run_ts"] == "20260101_000000Z"].iloc[0]
    second_row = runs_df[runs_df["run_ts"] == "20260102_000000Z"].iloc[0]
    assert bool(first_row["has_promotion_json"]) is False
    assert bool(first_row["has_regime_diagnostics_json"]) is False
    assert bool(second_row["has_promotion_json"]) is True
    assert bool(second_row["has_regime_diagnostics_json"]) is True
    assert bool(second_row["has_data_quality_audit_json"]) is True
    assert bool(second_row["has_pipeline_run_json"]) is True
    assert bool(second_row["has_replay_audit_json"]) is True
    assert second_row["walkforward_active_vs_baseline_sharpe_delta"] > 0
    assert second_row["pipeline_status"] == "completed"
    assert second_row["data_quality_status"] == "ok"
    assert bool(second_row["replay_exact_pass"]) is True

    active_second = models_df[
        (models_df["run_ts"] == "20260102_000000Z") & (models_df["model_name"] == "active")
    ].iloc[0]
    assert active_second["eval_rank_overall_rmse"] == 1
    assert active_second["wf_rank_mean_sharpe"] == 1
    assert active_second["wf_sharpe_mean"] > 1.0

    assert latest_report["subject_run_ts"] == "20260102_000000Z"
    assert latest_report["history_summary"]["run_count"] == 2
    assert latest_report["history_summary"]["has_promotion_json_rate"] == 0.5
    assert latest_report["history_summary"]["has_data_quality_audit_json_rate"] == 0.5
    assert latest_report["inventory"]["test_case_count"] == 3
    assert latest_report["inventory"]["saved_replay_audit_count"] == 1
    assert latest_report["live_sim"]["trade_count"] == 2
    assert latest_report["live_sim"]["equity_prediction_coverage_rate"] is None
    assert latest_report["live_sim"]["trade_fee_coverage_rate"] is None
    assert latest_report["live_sim"].get("execution_fee_total") is None
    assert latest_report["replay"]["exact_pass_rate"] == 1.0
    assert latest_report["data_quality"]["ok_rate"] == 1.0
    assert latest_report["pipeline"]["success_rate"] == 0.5
    assert latest_report["deployments"]["event_count"] == 2
    assert latest_report["roadmap"]["implemented_count"] == 5
    assert latest_report["roadmap"]["partial_count"] == 2
    assert latest_report["roadmap"]["planned_count"] == 5
    assert latest_report["artifact_storage"]["lineage_referenced_file_count"] == 15
    assert latest_report["artifact_storage"]["lineage_referenced_existing_file_count"] == 5
    assert latest_report["artifact_storage"]["lineage_referenced_total_bytes"] > 0
    assert "20260102_000000Z" in latest_md
    assert "docs/future_metrics_roadmap.md" in latest_md
    assert "Roadmap Coverage" in latest_md
    assert "Replay" in latest_md
    assert "Artifact Storage" in latest_md

    assert manifest["subject_run_ts"] == "20260102_000000Z"
    assert manifest["history_run_ts"] == ["20260101_000000Z", "20260102_000000Z"]
    assert "latest_report.json" in manifest["output_hashes"]
    assert "docs/future_metrics_roadmap.md" in manifest["input_hashes"]
    assert "data/deployments/events.parquet" in manifest["input_hashes"]


def test_generate_project_metrics_reports_available_execution_and_storage_metrics(
    tmp_path: Path,
) -> None:
    _build_history_fixture(tmp_path)
    live_dir = tmp_path / "data" / "live_sim"

    equity_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
            "cash": [1040.0, 1000.0, 940.0],
            "position": [0.5, 0.0, 1.0],
            "last_price": [120.0, 100.0, 110.0],
            "portfolio_value": [1100.0, 1000.0, 1050.0],
            "realized_pnl": [10.0, 0.0, 0.0],
            "unrealized_pnl": [50.0, 0.0, 50.0],
            "total_pnl": [60.0, 0.0, 50.0],
            "cost_basis": [50.0, 0.0, 100.0],
            "avg_entry_price": [100.0, 0.0, 100.0],
            "regime": ["bearish", "sideways", "bullish"],
            "active_model_id": ["model_bear", "model_sideways", "model_bull"],
            "prediction": [-0.01, 0.0, 0.01],
            "signal": ["SELL", "HOLD", "BUY"],
            "action_taken": ["SELL", "NONE", "BUY"],
            "reason": ["filled", "hold", "filled"],
        }
    )
    equity_df.to_parquet(live_dir / "equity_curve.parquet", index=False)

    trades_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "signal_bar_timestamp": [
                "2026-01-02T23:55:00Z",
                "2026-01-01T23:55:00Z",
                None,
            ],
            "fill_bar_timestamp": [
                "2026-01-03T00:00:00Z",
                "2026-01-02T00:00:00Z",
                None,
            ],
            "fill_policy": ["next_open", "next_open", None],
            "signal": ["SELL", "BUY", "HOLD"],
            "action": ["SELL", "BUY", "NONE"],
            "price": [120.0, 100.0, 100.0],
            "fill_price": [119.88, 100.10, None],
            "shares_delta": [-5.0, 10.0, 0.0],
            "trade_value": [599.4, 1001.0, 0.0],
            "fee": [0.05994, 0.1001, 0.0],
            "realized_pnl_delta": [10.0, 0.0, 0.0],
            "regime": ["bearish", "bullish", "sideways"],
            "active_model_id": ["model_bear", "model_bull", "model_sideways"],
            "prediction": [-0.01, 0.01, 0.0],
            "reason": ["filled", "filled", "hold"],
        }
    )
    trades_df.to_parquet(live_dir / "trades.parquet", index=False)

    model_path = tmp_path / "models" / "experts" / "candidate.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"candidate-model")

    outputs = generate_project_metrics_report(project_root=tmp_path)
    report = json.loads(outputs["latest_report_json"].read_text(encoding="utf-8"))
    live_sim = report["live_sim"]
    storage = report["artifact_storage"]

    assert live_sim["start_portfolio_value"] == 1000.0
    assert live_sim["end_portfolio_value"] == 1100.0
    assert live_sim["total_return_pct"] == pytest.approx(10.0)
    assert live_sim["equity_prediction_coverage_rate"] == 1.0
    assert live_sim["equity_active_model_id_coverage_rate"] == 1.0
    assert live_sim["equity_gross_exposure_coverage_rate"] == 1.0
    assert live_sim["equity_gross_exposure_pct_max"] == pytest.approx(110.0 / 1050.0 * 100.0)
    assert live_sim["equity_latest_total_pnl"] == 60.0
    assert live_sim["decision_signal_buy_count"] == 1
    assert live_sim["decision_action_sell_count"] == 1

    assert live_sim["filled_trade_count"] == 2
    assert live_sim["execution_signal_intent_count"] == 2
    assert live_sim["execution_signal_outcome_coverage_rate"] == 1.0
    assert live_sim["execution_signal_intent_fill_rate"] == 1.0
    assert live_sim["trade_fill_price_coverage_rate"] == pytest.approx(2.0 / 3.0)
    assert live_sim["signal_to_fill_delay_coverage_rate"] == 1.0
    assert live_sim["signal_to_fill_delay_seconds_mean"] == pytest.approx(300.0)
    assert live_sim["signal_to_fill_delay_seconds_p95"] == pytest.approx(300.0)
    assert live_sim["executed_notional_total"] == pytest.approx(1600.4)
    assert live_sim["execution_fee_total"] == pytest.approx(0.16004)
    assert live_sim["execution_fee_effective_bps"] == pytest.approx(1.0)
    assert live_sim["implied_slippage_cost_total"] == pytest.approx(1.6)
    assert live_sim["implied_slippage_effective_bps"] == pytest.approx(10.0)
    assert live_sim["estimated_execution_cost_total"] == pytest.approx(1.76004)
    assert live_sim["execution_realized_pnl_delta_total"] == pytest.approx(10.0)

    assert storage["model_artifact_file_count"] == 1
    assert storage["model_artifact_total_bytes"] == len(b"candidate-model")
    assert storage["managed_total_bytes"] >= len(b"candidate-model")
    assert storage["lineage_referenced_size_coverage_rate"] == pytest.approx(5.0 / 15.0)


def test_generate_project_metrics_is_deterministic(tmp_path: Path) -> None:
    _build_history_fixture(tmp_path)

    first_outputs = generate_project_metrics_report(project_root=tmp_path)
    first_runs = pd.read_parquet(first_outputs["runs_history_parquet"])
    first_models = pd.read_parquet(first_outputs["models_history_parquet"])
    first_report = json.loads(first_outputs["latest_report_json"].read_text(encoding="utf-8"))
    first_md = first_outputs["latest_report_md"].read_text(encoding="utf-8")

    second_outputs = generate_project_metrics_report(project_root=tmp_path)
    second_runs = pd.read_parquet(second_outputs["runs_history_parquet"])
    second_models = pd.read_parquet(second_outputs["models_history_parquet"])
    second_report = json.loads(second_outputs["latest_report_json"].read_text(encoding="utf-8"))
    second_md = second_outputs["latest_report_md"].read_text(encoding="utf-8")

    pd.testing.assert_frame_equal(first_runs, second_runs)
    pd.testing.assert_frame_equal(first_models, second_models)
    assert first_report == second_report
    assert first_md == second_md


def test_canonical_offline_pipeline_stage_declared_in_dvc_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "dvc.yaml").read_text(encoding="utf-8")
    assert "offline_pipeline:" in text
    assert "python -m src.pipeline.run --offline" in text
    assert "always_changed: true" in text

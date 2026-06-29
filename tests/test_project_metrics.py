from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def _write_lineage(root: Path, *, run_ts: str, include_promotion_refs: bool = True) -> None:
    lineage_dir = root / "artifacts" / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {
        "features_parquet": {"path": f"data/features/{run_ts}.parquet", "sha256": "features"},
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
        "# Future Metrics Roadmap\n\nplaceholder\n",
        encoding="utf-8",
    )


def _build_history_fixture(root: Path) -> None:
    _write_docs(root)
    _write_test_inventory_files(root)
    _write_live_sim(root)

    first_run = "20260101_000000Z"
    second_run = "20260102_000000Z"

    _write_lineage(root, run_ts=first_run, include_promotion_refs=False)
    _write_scorecard_artifacts(root, run_ts=first_run, active_rmse=0.30, baseline_rmse=0.55)
    _write_walkforward_artifacts(root, run_ts=first_run, active_sharpe=1.10, baseline_sharpe=0.80)

    _write_lineage(root, run_ts=second_run, include_promotion_refs=True)
    _write_scorecard_artifacts(root, run_ts=second_run, active_rmse=0.22, baseline_rmse=0.50)
    _write_walkforward_artifacts(root, run_ts=second_run, active_sharpe=1.60, baseline_sharpe=0.95)
    _write_promotion(root, run_ts=second_run, promoted=True)
    _write_regime_diagnostics(root, run_ts=second_run)

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
    assert second_row["walkforward_active_vs_baseline_sharpe_delta"] > 0

    active_second = models_df[
        (models_df["run_ts"] == "20260102_000000Z") & (models_df["model_name"] == "active")
    ].iloc[0]
    assert active_second["eval_rank_overall_rmse"] == 1
    assert active_second["wf_rank_mean_sharpe"] == 1
    assert active_second["wf_sharpe_mean"] > 1.0

    assert latest_report["subject_run_ts"] == "20260102_000000Z"
    assert latest_report["history_summary"]["run_count"] == 2
    assert latest_report["history_summary"]["has_promotion_json_rate"] == 0.5
    assert latest_report["inventory"]["test_case_count"] == 3
    assert latest_report["live_sim"]["trade_count"] == 2
    assert "20260102_000000Z" in latest_md
    assert "docs/future_metrics_roadmap.md" in latest_md

    assert manifest["subject_run_ts"] == "20260102_000000Z"
    assert manifest["history_run_ts"] == ["20260101_000000Z", "20260102_000000Z"]
    assert "latest_report.json" in manifest["output_hashes"]
    assert "docs/future_metrics_roadmap.md" in manifest["input_hashes"]


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


def test_report_metrics_stage_declared_in_dvc_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "dvc.yaml").read_text(encoding="utf-8")
    assert "report_metrics:" in text
    assert "tools/collect_project_metrics.py" in text
    assert "data/reporting" in text

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.dashboard.data import load_dashboard_snapshot

NOW = datetime(2026, 7, 15, 16, 10, tzinfo=UTC)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_live_state(root: Path, *, updated_at: str = "2026-07-15T16:05:00+00:00") -> None:
    _write_json(
        root / "data/live_sim/account_state.json",
        {
            "cash": 80_000.0,
            "position": 10.0,
            "last_price": 100.0,
            "portfolio_value": 81_000.0,
            "updated_at": updated_at,
        },
    )
    _write_json(
        root / "data/live_sim/heartbeat.json",
        {
            "updated_at_utc": updated_at,
            "bar_timestamp_utc": "2026-07-15T16:00:00+00:00",
            "regime": "bullish",
            "selected_model_id": "expert_bullish",
            "prediction": 0.01,
            "signal": "BUY",
        },
    )
    equity_path = root / "data/live_sim/equity_curve.parquet"
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-07-15T15:55:00Z", "2026-07-15T16:00:00Z"],
            "portfolio_value": [80_500.0, 81_000.0],
        }
    ).to_parquet(equity_path, index=False)


def test_loader_reads_live_persisted_state_without_mutating_files(tmp_path: Path) -> None:
    _write_live_state(tmp_path)
    (tmp_path / "data/live_sim/live_sim.lock").touch()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert snapshot.freshness.label == "LIVE"
    assert snapshot.overview["regime"] == "bullish"
    assert snapshot.overview["model"] == "expert_bullish"
    assert snapshot.overview["portfolio_value"] == 81_000.0
    assert snapshot.equity is not None
    assert before == after


def test_loader_starts_without_any_runtime_state(tmp_path: Path) -> None:
    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    assert snapshot.freshness.label == "UNAVAILABLE"
    assert snapshot.account is None
    assert snapshot.equity is None
    assert list(tmp_path.iterdir()) == []


def test_loader_labels_evaluation_only_artifacts_historical(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "data/scorecards/latest.json",
        {"overall": {"n": 2, "by_model": {"baseline": {"mae": 0.2, "rmse": 0.3}}}},
    )

    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    assert snapshot.freshness.label == "HISTORICAL"
    assert snapshot.scorecard is not None


def test_loader_marks_old_heartbeat_stale_and_tolerates_bad_optional_artifact(
    tmp_path: Path,
) -> None:
    _write_live_state(tmp_path, updated_at="2026-07-15T12:00:00+00:00")
    bad_trades = tmp_path / "data/live_sim/trades.parquet"
    bad_trades.write_text("not a parquet file", encoding="utf-8")

    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    assert snapshot.freshness.label == "STALE"
    assert snapshot.trades is None
    assert any("trade history" in warning for warning in snapshot.warnings)


def test_loader_tolerates_malformed_active_registry(tmp_path: Path) -> None:
    active = tmp_path / "registry/active_model.yaml"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text("active: [not valid", encoding="utf-8")

    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    assert snapshot.active_model is None
    assert any("active model registry" in warning for warning in snapshot.warnings)


def test_loader_reads_historical_and_registry_artifacts_independently(tmp_path: Path) -> None:
    regimes = tmp_path / "data/regimes/latest.parquet"
    regimes.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-07-14T00:00:00Z", "2026-07-15T00:00:00Z"],
            "regime": ["bearish", "sideways"],
        }
    ).to_parquet(regimes, index=False)
    predictions = tmp_path / "data/predictions/latest.parquet"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": [0, 1],
            "model_name": ["first", "expert_sideways"],
            "active_model_id": ["first", "expert_sideways"],
            "y_pred": [0.01, 0.02],
            "is_active": [True, True],
        }
    ).to_parquet(predictions, index=False)
    _write_json(
        tmp_path / "data/scorecards/latest.json",
        {"overall": {"n": 2, "by_model": {"expert_sideways": {"mae": 0.1, "rmse": 0.2}}}},
    )
    walkforward = tmp_path / "data/walkforward/latest.parquet"
    walkforward.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model_name": ["expert_sideways"], "sharpe": [1.2]}).to_parquet(
        walkforward, index=False
    )
    backtest = tmp_path / "artifacts/backtest/results_20260715.parquet"
    backtest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": [100.0, 101.0]}).to_parquet(backtest, index=False)
    active = tmp_path / "registry/active_model.yaml"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "active:\n  model_type: expert\n  model_id: expert_sideways\n  version: v1\n"
        "  artifact_path: models/experts/sideways/model.joblib\n  regime: sideways\n"
        "updated_at: '2026-07-15T00:00:00Z'\n",
        encoding="utf-8",
    )

    snapshot = load_dashboard_snapshot(tmp_path, now=NOW)

    assert snapshot.freshness.label == "LATEST KNOWN"
    assert snapshot.overview["regime"] == "sideways"
    assert snapshot.overview["model"] == "expert_sideways"
    assert snapshot.scorecard is not None
    assert snapshot.walkforward is not None
    assert snapshot.backtest is not None
    assert snapshot.active_model is not None
    assert snapshot.active_model.model_id == "expert_sideways"

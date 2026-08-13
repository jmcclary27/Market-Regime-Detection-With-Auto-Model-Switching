from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.experiment.engine import initial_state, process_daily_bar
from src.experiment.manifest import (
    ArtifactRef,
    FrozenExperimentManifest,
    ManifestError,
    freeze_manifest,
)
from src.experiment.reporting import build_dashboard_payload
from src.experiment.selection import select_static_model
from src.experiment.trading import TargetExposureConfig, prediction_to_target, rebalance_to_target
from src.trading.state import AccountState


def _ref(name: str, model_id: str = "") -> ArtifactRef:
    return ArtifactRef(path=name, sha256="a" * 64, version="v1", model_id=model_id)


def _manifest() -> FrozenExperimentManifest:
    return FrozenExperimentManifest(
        schema_version=1,
        experiment_id="daily-spy-test",
        created_at_utc="2026-08-13T00:00:00+00:00",
        data_cutoff="2026-08-12",
        symbols=("SPY", "QQQ"),
        traded_symbol="SPY",
        starting_cash=100_000,
        fee_bps=1,
        slippage_bps=2,
        exposure_thresholds=(-0.001, 0.001),
        static_model=_ref("models/static.joblib", "static-v1"),
        regime_models={
            "bullish": _ref("models/bull.joblib", "bull-v1"),
            "sideways": _ref("models/side.joblib", "side-v1"),
            "bearish": _ref("models/bear.joblib", "bear-v1"),
        },
        regime_detector=_ref("models/hmm.joblib"),
        feature_manifest=_ref("features.json"),
        git_commit="abc123",
    )


def _preds(static: float, bull: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"row_id": [7, 7], "model_name": ["static-v1", "bull-v1"], "y_pred": [static, bull]}
    )


def test_prediction_to_three_exposure_targets() -> None:
    assert prediction_to_target(-0.002, lower=-0.001, upper=0.001) == 0
    assert prediction_to_target(0, lower=-0.001, upper=0.001) == 0.5
    assert prediction_to_target(0.002, lower=-0.001, upper=0.001) == 1


def test_rebalance_fills_at_next_open_with_costs() -> None:
    state = AccountState(cash=100_000, portfolio_value=100_000)
    trade = rebalance_to_target(
        state=state, target_exposure=1.0, open_price=100, config=TargetExposureConfig()
    )
    assert trade["action"] == "BUY"
    assert trade["fill_price"] == pytest.approx(100.02)
    assert trade["fee"] > 0
    assert state.cash >= 0


def test_manifest_cannot_change_after_freeze(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    freeze_manifest(path, _manifest())
    changed = replace(_manifest(), starting_cash=90_000)
    with pytest.raises(ManifestError, match="cannot be changed"):
        freeze_manifest(path, changed)


def test_engine_queues_close_decisions_then_fills_at_next_open() -> None:
    manifest = _manifest()
    first, event_one = process_daily_bar(
        state=initial_state(manifest),
        manifest=manifest,
        predictions=_preds(0.0, 0.002),
        bar_timestamp_utc="2026-08-13T20:00:00Z",
        row_id=7,
        regime="bullish",
        regime_confidence=0.8,
        open_price=100,
        close_price=101,
    )
    assert all(fill is None for fill in event_one["fills"].values())
    assert first["pending_targets"]["buy_and_hold"]["target_exposure"] == 1
    assert first["pending_targets"]["static_ml"]["target_exposure"] == 0.5
    assert first["pending_targets"]["regime_ml"]["target_exposure"] == 1

    second, event_two = process_daily_bar(
        state=first,
        manifest=manifest,
        predictions=_preds(-0.002, -0.002),
        bar_timestamp_utc="2026-08-14T20:00:00Z",
        row_id=7,
        regime="bullish",
        regime_confidence=0.7,
        open_price=102,
        close_price=103,
    )
    assert event_two["fills"]["buy_and_hold"]["reference_open_price"] == 102
    assert event_two["fills"]["static_ml"]["action"] == "BUY"
    assert second["pending_targets"]["static_ml"]["target_exposure"] == 0


def test_dashboard_marks_short_sharpe_unavailable() -> None:
    manifest = _manifest()
    state, event = process_daily_bar(
        state=initial_state(manifest),
        manifest=manifest,
        predictions=_preds(0.0, 0.002),
        bar_timestamp_utc="2026-08-13T20:00:00Z",
        row_id=7,
        regime="bullish",
        regime_confidence=0.8,
        open_price=100,
        close_price=101,
    )
    del state
    payload = build_dashboard_payload(manifest=manifest, events=[event])
    assert payload["metrics"]["static_ml"]["sharpe"] is None
    assert "Paper trading" in payload["disclaimer"]


def test_static_selection_uses_predeclared_tie_breakers() -> None:
    scorecard = pd.DataFrame(
        {
            "model_id": ["ridge", "lightgbm"],
            "is_global_candidate": [True, True],
            "walk_forward_net_sharpe": [1.0, 1.0],
            "max_drawdown": [-0.20, -0.10],
            "cumulative_return": [0.20, 0.15],
        }
    )
    assert select_static_model(scorecard)["model_id"] == "lightgbm"

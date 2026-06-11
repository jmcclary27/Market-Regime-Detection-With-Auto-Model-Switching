from pathlib import Path

import pandas as pd
import yaml

from src.deploy.switcher import SwitchConfig, run_switcher


def _write_scorecard(path: Path, baseline_rmse: float, candidate_rmse: float) -> None:
    df = pd.DataFrame(
        {
            "scope": ["overall", "overall"],
            "regime": [None, None],
            "model_name": ["baseline", "expert_bullish"],
            "n": [100, 100],
            "rmse": [baseline_rmse, candidate_rmse],
            "mae": [baseline_rmse, candidate_rmse],
        }
    )
    df.to_parquet(path, index=False)


def test_switcher_promotes_and_updates_registry(tmp_path: Path) -> None:
    # Arrange
    data_dir = tmp_path / "data"
    (data_dir / "scorecards").mkdir(parents=True)
    (tmp_path / "registry").mkdir(parents=True)

    _write_scorecard(
        data_dir / "scorecards" / "latest.parquet", baseline_rmse=1.0, candidate_rmse=0.8
    )

    config = SwitchConfig(
        metric_name="rmse", promote_margin=0.0, rollback_margin=0.0, update_registry_on_promote=True
    )

    # Act
    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )

    # Assert event
    events = pd.read_parquet(data_dir / "deployments" / "events.parquet")
    row = events.iloc[0]
    assert row["decision"] == "promote"
    assert row["event_type"] == "promoted"
    assert row["active_model_id_after"] == "expert_bullish"

    # Assert registry pointer updated
    active_yaml = tmp_path / "registry" / "active_model.yaml"
    assert active_yaml.exists()
    payload = yaml.safe_load(active_yaml.read_text(encoding="utf-8"))
    assert payload["active"]["model_id"] == "expert_bullish"
    assert payload["active"]["model_type"] == "expert"
    assert payload["active"]["regime"] == "bullish"


def test_switcher_holds_when_within_margins(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "scorecards").mkdir(parents=True)
    (tmp_path / "registry").mkdir(parents=True)

    _write_scorecard(
        data_dir / "scorecards" / "latest.parquet", baseline_rmse=1.0, candidate_rmse=1.0
    )

    config = SwitchConfig(
        metric_name="rmse", promote_margin=0.1, rollback_margin=0.1, update_registry_on_promote=True
    )

    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )

    events = pd.read_parquet(data_dir / "deployments" / "events.parquet")
    row = events.iloc[0]
    assert row["decision"] == "hold"
    assert row["event_type"] == "hold"
    assert row["active_model_id_after"] == "baseline"


def test_switcher_rolls_back_when_candidate_worse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "scorecards").mkdir(parents=True)
    (tmp_path / "registry").mkdir(parents=True)

    _write_scorecard(
        data_dir / "scorecards" / "latest.parquet", baseline_rmse=1.0, candidate_rmse=1.3
    )

    config = SwitchConfig(
        metric_name="rmse", promote_margin=0.0, rollback_margin=0.1, update_registry_on_promote=True
    )

    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )

    events = pd.read_parquet(data_dir / "deployments" / "events.parquet")
    row = events.iloc[0]
    assert row["decision"] == "rollback"
    assert row["event_type"] == "rollback"
    assert row["active_model_id_after"] == "baseline"

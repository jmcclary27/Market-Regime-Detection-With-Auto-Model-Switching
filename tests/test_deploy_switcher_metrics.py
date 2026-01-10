from pathlib import Path

import pandas as pd

from src.deploy.switcher import SwitchConfig, run_switcher


def test_switcher_logs_metrics_from_scorecard(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scorecards_dir = data_dir / "scorecards"
    scorecards_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "scope": ["overall", "overall"],
            "regime": [None, None],
            "model_name": ["baseline", "expert_bullish"],
            "n": [100, 100],
            "rmse": [1.0, 0.9],
            "mae": [0.8, 0.7],
        }
    )
    df.to_parquet(scorecards_dir / "latest.parquet", index=False)

    config = SwitchConfig(metric_name="rmse", window_value=50)

    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline",
        candidate_model_id="expert_bullish",
    )

    events_path = data_dir / "deployments" / "events.parquet"
    events = pd.read_parquet(events_path)
    row = events.iloc[0]

    assert row["active_metric_value"] == 1.0
    assert row["candidate_metric_value"] == 0.9
    assert row["n"] == 100
    assert row["decision"] == "no_action"
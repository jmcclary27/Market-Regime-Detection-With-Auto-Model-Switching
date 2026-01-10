from pathlib import Path

import pandas as pd

from src.deploy.switcher import SwitchConfig, run_switcher


def test_switcher_logs_metrics_from_scorecard(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scorecards_dir = data_dir / "scorecards"
    scorecards_dir.mkdir(parents=True)

    # fake scorecard
    df = pd.DataFrame(
        {
            "model_id": ["baseline@v1", "candidate@v1"],
            "mse": [1.0, 0.9],
        }
    )
    df.to_parquet(scorecards_dir / "latest.parquet", index=False)

    config = SwitchConfig(metric_name="mse", window_value=50)

    run_switcher(
        data_dir=data_dir,
        config=config,
        active_model_id="baseline@v1",
        candidate_model_id="candidate@v1",
    )

    events_path = data_dir / "deployments" / "events.parquet"
    assert events_path.exists()

    events = pd.read_parquet(events_path)
    assert len(events) == 1

    row = events.iloc[0]
    assert row["active_metric_value"] == 1.0
    assert row["candidate_metric_value"] == 0.9
    assert row["decision"] == "no_action"
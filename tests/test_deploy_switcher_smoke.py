from pathlib import Path

import pandas as pd

from src.deploy.switcher import SwitchConfig, run_switcher


def test_switcher_writes_event(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = SwitchConfig(window_value=10)

    run_switcher(data_dir=data_dir, config=config)

    events_path = data_dir / "deployments" / "events.parquet"
    assert events_path.exists()

    df = pd.read_parquet(events_path)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["event_type"] == "canary_evaluated"
    assert row["decision"] == "no_action"
    assert row["window_value"] == 10
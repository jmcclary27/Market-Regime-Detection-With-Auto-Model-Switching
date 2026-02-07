from __future__ import annotations

import pandas as pd

from src.eval.metrics import EvalConfig, build_eval_frame, compute_metrics_table
from src.eval.run_evaluator import _ensure_time_sorted, _extract_market_time
from src.eval.walk_forward import walk_forward_splits


def test_walk_forward_eval_produces_split_metrics() -> None:
    n = 120
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")

    # Features and regimes, no row_id provided so _ensure_row_id will create it
    features = pd.DataFrame(
        {
            "timestamp": ts,
            # target column expected by default EvalConfig
            "log_return_1": [0.01] * n,
        }
    )
    regimes = pd.DataFrame(
        {
            "timestamp": ts,
            "regime": ["0"] * n,
        }
    )

    # Predictions are long-form by row_id
    preds = pd.DataFrame(
        {
            "row_id": list(range(n)),
            "model_name": ["baseline"] * n,
            "y_pred": [0.02] * n,
        }
    )

    cfg = EvalConfig(target_col="log_return_1", min_regime_n=1)

    eval_df = build_eval_frame(predictions=preds, features=features, regimes=regimes, cfg=cfg)

    eval_df_sorted = _ensure_time_sorted(eval_df, features)
    market_ts = pd.to_datetime(
        _extract_market_time(eval_df_sorted, features), utc=True, errors="raise"
    )

    splits = walk_forward_splits(
        market_ts,
        train_size=60,
        val_size=20,
        test_size=20,
        step_size=20,
        anchored=True,
    )
    assert len(splits) >= 1

    # Compute metrics on first split's test window, just like run_evaluator does
    s0 = splits[0]
    test_df = eval_df_sorted.iloc[s0.test.start : s0.test.stop].copy()
    assert len(test_df) == 20

    tbl = compute_metrics_table(test_df, cfg=cfg)
    assert not tbl.empty
    assert set(["scope", "model_name", "n"]).issubset(set(tbl.columns))
    assert "mae" in tbl.columns
    assert "rmse" in tbl.columns

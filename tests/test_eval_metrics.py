from __future__ import annotations

import pandas as pd

from src.eval.metrics import EvalConfig, build_eval_frame, compute_metrics_table, mae, rmse


def test_metrics_known_values() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 1.0, 5.0])
    # abs diffs: 0, 1, 2 -> MAE = 1
    assert mae(y_true, y_pred) == 1.0
    # squared diffs: 0,1,4 -> mean=5/3 -> rmse = sqrt(5/3)
    assert abs(rmse(y_true, y_pred) - (5.0 / 3.0) ** 0.5) < 1e-12


def test_build_eval_frame_reconstructs_row_id_and_joins() -> None:
    cfg = EvalConfig(target_col="log_return_1")

    # features/regimes intentionally missing row_id, must be reconstructed by sort(timestamp,symbol)
    features = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 09:40:00", "2024-01-01 09:39:00"],
            "symbol": ["TEST", "TEST"],
            "log_return_1": [0.2, 0.1],
        }
    )
    regimes = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 09:40:00", "2024-01-01 09:39:00"],
            "symbol": ["TEST", "TEST"],
            "regime": ["B", "A"],
            "regime_explanation": ["b", "a"],
        }
    )

    # predictions use row_id that should match sorted order:
    # sorted timestamps => 09:39 row_id=0, 09:40 row_id=1
    predictions = pd.DataFrame(
        {
            "row_id": [0, 1],
            "model_name": ["m1", "m1"],
            "y_pred": [0.05, 0.25],
        }
    )

    eval_df = build_eval_frame(
        predictions=predictions,
        features=features,
        regimes=regimes,
        cfg=cfg,
    )

    assert set(eval_df.columns) >= {"row_id", "model_name", "y_pred", "y_true", "regime"}
    # row_id=0 should map to 09:39 truth 0.1 and regime A
    r0 = eval_df.sort_values("row_id").iloc[0]
    assert r0["row_id"] == 0
    assert r0["y_true"] == 0.1
    assert r0["regime"] == "A"


def test_compute_metrics_table_has_overall_and_regime_rows() -> None:
    cfg = EvalConfig(target_col="log_return_1")
    eval_df = pd.DataFrame(
        {
            "row_id": [0, 1, 0, 1],
            "model_name": ["m1", "m1", "m2", "m2"],
            "y_pred": [0.0, 1.0, 0.5, 0.5],
            "y_true": [0.0, 2.0, 0.0, 2.0],
            "regime": ["A", "B", "A", "B"],
        }
    )

    tbl = compute_metrics_table(eval_df, cfg=cfg)

    assert {"scope", "regime", "model_name", "n", "mae", "rmse"} <= set(tbl.columns)
    assert (tbl["scope"] == "overall").any()
    assert (tbl["scope"] == "regime").any()

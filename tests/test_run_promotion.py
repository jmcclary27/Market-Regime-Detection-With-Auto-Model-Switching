from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.models.run_promotion import run_promotion
from src.registry.registry import ActiveModelRef


def _dummy_ref() -> ActiveModelRef:
    return ActiveModelRef(
        model_type="pretrained",
        model_id="unused",
        version="0",
        artifact_path=Path("models/pretrained/unused.joblib"),
        regime=None,
        metadata_path=None,
    )


def _write_promotion_inputs(
    repo_root: Path,
    *,
    run_ts: str,
    wf: pd.DataFrame,
    preds: pd.DataFrame | None = None,
) -> None:
    (repo_root / "artifacts" / "lineage").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "walkforward").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "predictions").mkdir(parents=True, exist_ok=True)

    lineage = {"run_ts": run_ts, "git_commit": "test", "config_sha256": "abc"}
    (repo_root / "artifacts" / "lineage" / "latest.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    wf.to_parquet(
        repo_root / "data" / "walkforward" / f"portfolio_metrics_{run_ts}.parquet",
        index=False,
    )

    if preds is not None:
        preds.to_parquet(repo_root / "data" / "predictions" / "latest.parquet", index=False)


def test_run_promotion_rejects_explicit_arima_challenger(tmp_path: Path) -> None:
    run_ts = "20260101_000000Z"
    wf = pd.DataFrame(
        {
            "model_name": ["baseline", "expert_arima_bullish"],
            "fold_id": [0, 0],
            "sharpe": [0.5, 1.5],
            "max_drawdown": [-0.2, -0.1],
        }
    )
    _write_promotion_inputs(tmp_path, run_ts=run_ts, wf=wf)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="non-promotable"):
            run_promotion(
                challenger_model_name="expert_arima_bullish",
                incumbent_model_name="baseline",
                challenger_ref=_dummy_ref(),
                write_pointer=False,
            )
    finally:
        os.chdir(old_cwd)


def test_run_promotion_auto_selection_ignores_arima_models(tmp_path: Path) -> None:
    run_ts = "20260101_000000Z"
    wf = pd.DataFrame(
        {
            "model_name": [
                "baseline",
                "expert_arima_bullish",
                "expert_lightgbm_bullish",
            ],
            "fold_id": [0, 0, 0],
            "sharpe": [0.5, 2.0, 1.2],
            "max_drawdown": [-0.2, -0.1, -0.1],
        }
    )
    preds = pd.DataFrame(
        {
            "model_name": ["expert_lightgbm_bullish"],
            "model_source": ["expert"],
            "model_path": ["models/experts/bullish/latest.joblib"],
        }
    )
    _write_promotion_inputs(tmp_path, run_ts=run_ts, wf=wf, preds=preds)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        out = run_promotion(
            challenger_model_name=None,
            incumbent_model_name="baseline",
            challenger_ref=_dummy_ref(),
            write_pointer=False,
        )
    finally:
        os.chdir(old_cwd)

    assert out["challenger_model_name"] == "expert_lightgbm_bullish"
    assert out["promoted"] is True
    assert out["promotable_models"] == ["baseline", "expert_lightgbm_bullish"]
    assert "expert_arima_*" in out["non_promotable_models"]

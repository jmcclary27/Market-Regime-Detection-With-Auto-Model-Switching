from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.features.stationary import augment_pairwise_stationary_features, summarize_feature_ranges
from src.models.run_promotion import is_non_promotable_model, run_promotion
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
    features_path: Path | None = None,
) -> None:
    (repo_root / "artifacts" / "lineage").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "walkforward").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "predictions").mkdir(parents=True, exist_ok=True)

    lineage: dict[str, object] = {"run_ts": run_ts, "git_commit": "test", "config_sha256": "abc"}
    if features_path is not None:
        lineage["artifacts"] = {
            "features_parquet": {
                "path": str(features_path),
                "sha256": "test",
            }
        }
    (repo_root / "artifacts" / "lineage" / "latest.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    wf.to_parquet(
        repo_root / "data" / "walkforward" / f"portfolio_metrics_{run_ts}.parquet",
        index=False,
    )

    if preds is not None:
        preds.to_parquet(repo_root / "data" / "predictions" / "latest.parquet", index=False)


def _write_expert_guard_inputs(
    repo_root: Path,
    *,
    candidate_model_name: str,
    y_pred: list[float],
) -> Path:
    features_path = repo_root / "data" / "features" / "current.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)

    features = pd.DataFrame(
        {
            "close_x": [100.0, 101.0, 102.0, 103.0],
            "log_return_1_x": [0.01, 0.011, 0.012, 0.013],
            "sma_10_x": [99.4, 100.0, 100.6, 101.1],
            "close_y": [50.0, 50.4, 50.8, 51.1],
            "log_return_1_y": [0.004, 0.005, 0.0055, 0.006],
            "sma_10_y": [49.5, 49.8, 50.1, 50.4],
        }
    )
    features.to_parquet(features_path, index=False)

    augmented, stationary_cols = augment_pairwise_stationary_features(features)
    metadata = {
        "model_type": "lightgbm",
        "model_name": candidate_model_name,
        "regime": "bullish",
        "feature_columns": list(augmented.columns),
        "stationary_feature_columns": stationary_cols,
        "feature_range_stats": summarize_feature_ranges(augmented, list(augmented.columns)),
    }
    metadata_path = repo_root / "models" / "experts" / "bullish" / "latest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "model_name": [candidate_model_name] * len(y_pred),
            "model_source": ["expert"] * len(y_pred),
            "model_path": ["models/experts/bullish/latest.joblib"] * len(y_pred),
            "features_path": [str(features_path)] * len(y_pred),
            "y_pred": y_pred,
        }
    )
    (repo_root / "data" / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_parquet(repo_root / "data" / "predictions" / "latest.parquet", index=False)
    return features_path


def test_arima_model_is_promotable_after_quality_validation() -> None:
    assert not is_non_promotable_model("expert_arima_bullish_arima_v1")


def test_run_promotion_auto_selection_can_select_quality_gated_arima(tmp_path: Path) -> None:
    run_ts = "20260101_000000Z"
    wf = pd.DataFrame(
        {
            "model_name": [
                "baseline",
                "expert_arima_bullish_arima_v1",
                "expert_lightgbm_bullish",
            ],
            "fold_id": [0, 0, 0],
            "sharpe": [0.5, 2.0, 1.2],
            "max_drawdown": [-0.2, -0.1, -0.1],
        }
    )
    features_path = _write_expert_guard_inputs(
        tmp_path,
        candidate_model_name="expert_lightgbm_bullish",
        y_pred=[0.01, 0.015, 0.02, 0.018],
    )
    arima_path = (
        tmp_path / "models" / "experts" / "bullish" / "arima" / "expert_arima_bullish_arima_v1.json"
    )
    arima_path.parent.mkdir(parents=True, exist_ok=True)
    arima_path.write_text(
        json.dumps(
            {
                "model_type": "arima",
                "model_id": "expert_arima_bullish_arima_v1",
                "regime": "bullish",
                "quality_gate": {"promotion_eligible": True},
            }
        ),
        encoding="utf-8",
    )
    preds = pd.concat(
        [
            pd.read_parquet(tmp_path / "data" / "predictions" / "latest.parquet"),
            pd.DataFrame(
                {
                    "model_name": ["expert_arima_bullish_arima_v1"] * 4,
                    "model_source": ["expert"] * 4,
                    "model_path": [str(arima_path)] * 4,
                    "features_path": [str(features_path)] * 4,
                    "row_id": [0, 1, 2, 3],
                    "y_pred": [0.01, 0.015, 0.02, 0.018],
                }
            ),
        ],
        ignore_index=True,
    )
    _write_promotion_inputs(
        tmp_path,
        run_ts=run_ts,
        wf=wf,
        preds=preds,
        features_path=features_path,
    )

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

    assert out["challenger_model_name"] == "expert_arima_bullish_arima_v1"
    assert out["promoted"] is True
    assert out["promotable_models"] == [
        "baseline",
        "expert_arima_bullish_arima_v1",
        "expert_lightgbm_bullish",
    ]
    assert out["non_promotable_models"] == ["active"]
    assert out["promotion_guard"]["allowed"] is True


def test_run_promotion_blocks_flat_candidate(tmp_path: Path) -> None:
    run_ts = "20260102_000000Z"
    wf = pd.DataFrame(
        {
            "model_name": ["baseline", "expert_lightgbm_bullish"],
            "fold_id": [0, 0],
            "sharpe": [0.4, 1.5],
            "max_drawdown": [-0.2, -0.1],
        }
    )
    features_path = _write_expert_guard_inputs(
        tmp_path,
        candidate_model_name="expert_lightgbm_bullish",
        y_pred=[6.963846733498323e-05] * 4,
    )
    _write_promotion_inputs(
        tmp_path,
        run_ts=run_ts,
        wf=wf,
        preds=pd.read_parquet(tmp_path / "data" / "predictions" / "latest.parquet"),
        features_path=features_path,
    )

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        out = run_promotion(
            challenger_model_name=None,
            incumbent_model_name="baseline",
            challenger_ref=_dummy_ref(),
            write_pointer=True,
        )
    finally:
        os.chdir(old_cwd)

    assert out["promoted"] is False
    assert out["pointer_written"] is False
    assert out["decision"]["promote"] is False
    assert "flat" in out["reason"] or "degenerate" in out["reason"]
    assert not (tmp_path / "registry" / "active_model.yaml").exists()


def test_run_promotion_records_no_challenger_hold_at_explicit_paths(tmp_path: Path) -> None:
    run_ts = "20260103_000000Z"
    wf = pd.DataFrame(
        {
            "model_name": ["baseline"],
            "fold_id": [0],
            "sharpe": [0.5],
            "max_drawdown": [-0.2],
        }
    )
    _write_promotion_inputs(tmp_path, run_ts=run_ts, wf=wf)

    out = run_promotion(
        challenger_model_name=None,
        incumbent_model_name="baseline",
        challenger_ref=_dummy_ref(),
        lineage_path=tmp_path / "artifacts" / "lineage" / "latest.json",
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        walkforward_dir=tmp_path / "data" / "walkforward",
        predictions_path=tmp_path / "data" / "predictions" / "latest.parquet",
        deployment_events_path=tmp_path / "data" / "deployments" / "events.parquet",
        registry_path=tmp_path / "registry" / "active_model.yaml",
    )

    assert out["promoted"] is False
    assert out["reason"] == "no_promotable_challenger"
    assert (tmp_path / "data" / "walkforward" / f"promotion_{run_ts}.json").exists()
    events = pd.read_parquet(tmp_path / "data" / "deployments" / "events.parquet")
    assert events.iloc[-1]["decision"] == "hold"
    assert not (tmp_path / "registry" / "active_model.yaml").exists()

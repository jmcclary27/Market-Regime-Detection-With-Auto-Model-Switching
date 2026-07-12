from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    DriftPolicy,
    build_drift_snapshot,
    default_drift_output,
    find_previous_successful_inference,
    write_drift_snapshot,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_run(
    root: Path,
    *,
    run_ts: str,
    timestamp: int,
    features_path: Path,
    predictions_path: Path,
    replay: bool = False,
) -> Path:
    name = f"run_replay_{run_ts}.json" if replay else f"run_{run_ts}.json"
    path = root / "data" / "runs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_type": "batch_inference",
                "timestamp": timestamp,
                "features_path": features_path.relative_to(root).as_posix(),
                "output_path": predictions_path.relative_to(root).as_posix(),
                "num_prediction_rows": 40,
                "num_models_succeeded": 1,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _features(*, shifted: bool, reversed_regimes: bool = False) -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, 40)
    if shifted:
        values = values + 10.0
    regimes = ["bullish"] * 30 + ["sideways"] * 10
    if reversed_regimes:
        regimes = ["bullish"] * 10 + ["sideways"] * 30
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"),
            "feature_x": values,
            "target": np.linspace(-0.01, 0.01, 40),
            "regime": regimes,
            "regime_explanation": ["test"] * 40,
        }
    )


def _predictions(*, shifted: bool) -> pd.DataFrame:
    values = np.linspace(-0.01, 0.01, 40)
    if shifted:
        values = values + 0.10
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"),
            "row_id": range(40),
            "model_source": ["expert"] * 40,
            "model_name": ["expert_regression"] * 40,
            "is_active": [True] * 40,
            "y_pred": values,
        }
    )


def _multi_model_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_id in range(40):
        for model_name, offset in (("model_a", 0.0), ("model_b", 0.02)):
            rows.append(
                {
                    "row_id": row_id,
                    "model_source": "expert",
                    "model_name": model_name,
                    "y_pred": (row_id / 10_000.0) + offset,
                }
            )
    return pd.DataFrame(rows)


def test_drift_snapshot_flags_regression_distribution_changes(tmp_path: Path) -> None:
    previous_features = tmp_path / "data" / "regimes" / "previous.parquet"
    previous_predictions = tmp_path / "data" / "predictions" / "previous.parquet"
    current_features = tmp_path / "data" / "features" / "current.parquet"
    current_regimes = tmp_path / "data" / "regimes" / "current.parquet"
    current_predictions = tmp_path / "data" / "predictions" / "current.parquet"
    _write_parquet(previous_features, _features(shifted=False))
    _write_parquet(previous_predictions, _predictions(shifted=False))
    _write_parquet(current_features, _features(shifted=True, reversed_regimes=True))
    _write_parquet(current_regimes, _features(shifted=True, reversed_regimes=True))
    _write_parquet(current_predictions, _predictions(shifted=True))
    _write_run(
        tmp_path,
        run_ts="20260101_000000Z",
        timestamp=20260101000000,
        features_path=previous_features,
        predictions_path=previous_predictions,
    )
    _write_run(
        tmp_path,
        run_ts="20260102_000000Z",
        timestamp=20260102000000,
        features_path=current_regimes,
        predictions_path=current_predictions,
    )

    snapshot = build_drift_snapshot(
        project_root=tmp_path,
        run_ts="20260102_000000Z",
        current_features_path=current_features,
        current_regimes_path=current_regimes,
        current_predictions_path=current_predictions,
        policy=DriftPolicy(window_size=40, min_samples=30),
    )

    assert snapshot["status"] == "warning"
    assert snapshot["action"] == "review_only"
    assert snapshot["reference"]["run_ts"] == "20260101_000000Z"
    assert snapshot["feature_drift"]["warning_features"] == ["feature_x"]
    assert snapshot["prediction_drift"]["warning_models"] == [
        {"model_source": "expert", "model_name": "expert_regression"}
    ]
    assert snapshot["regime_distribution_drift"]["total_variation_distance"] == 0.5
    assert {warning["kind"] for warning in snapshot["warnings"]} == {
        "feature_mean_shift",
        "prediction_mean_shift",
        "regime_distribution_shift",
    }

    output_path = default_drift_output(tmp_path, "20260102_000000Z")
    write_drift_snapshot(output_path, snapshot, latest_path=output_path.parent / "latest.json")
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "warning"
    assert json.loads((output_path.parent / "latest.json").read_text(encoding="utf-8")) == snapshot


def test_drift_snapshot_is_deterministic_and_ignores_replay_runs(tmp_path: Path) -> None:
    previous_features = tmp_path / "data" / "regimes" / "previous.parquet"
    previous_predictions = tmp_path / "data" / "predictions" / "previous.parquet"
    current_features = tmp_path / "data" / "features" / "current.parquet"
    current_regimes = tmp_path / "data" / "regimes" / "current.parquet"
    current_predictions = tmp_path / "data" / "predictions" / "current.parquet"
    replay_features = tmp_path / "data" / "regimes" / "replay.parquet"
    replay_predictions = tmp_path / "data" / "predictions" / "replay.parquet"
    _write_parquet(previous_features, _features(shifted=False))
    _write_parquet(previous_predictions, _predictions(shifted=False))
    _write_parquet(current_features, _features(shifted=False))
    _write_parquet(current_regimes, _features(shifted=False))
    _write_parquet(current_predictions, _predictions(shifted=False))
    _write_parquet(replay_features, _features(shifted=True))
    _write_parquet(replay_predictions, _predictions(shifted=True))
    _write_run(
        tmp_path,
        run_ts="20260101_000000Z",
        timestamp=20260101000000,
        features_path=previous_features,
        predictions_path=previous_predictions,
    )
    _write_run(
        tmp_path,
        run_ts="20260101_120000Z",
        timestamp=20260101120000,
        features_path=replay_features,
        predictions_path=replay_predictions,
        replay=True,
    )
    _write_run(
        tmp_path,
        run_ts="20260102_000000Z",
        timestamp=20260102000000,
        features_path=current_regimes,
        predictions_path=current_predictions,
    )

    reference = find_previous_successful_inference(
        project_root=tmp_path,
        runs_dir=tmp_path / "data" / "runs",
        current_predictions_path=current_predictions,
    )
    assert reference is not None
    assert reference.run_ts == "20260101_000000Z"

    first = build_drift_snapshot(
        project_root=tmp_path,
        run_ts="20260102_000000Z",
        current_features_path=current_features,
        current_regimes_path=current_regimes,
        current_predictions_path=current_predictions,
        policy=DriftPolicy(window_size=40, min_samples=30),
    )
    second = build_drift_snapshot(
        project_root=tmp_path,
        run_ts="20260102_000000Z",
        current_features_path=current_features,
        current_regimes_path=current_regimes,
        current_predictions_path=current_predictions,
        policy=DriftPolicy(window_size=40, min_samples=30),
    )
    assert first == second
    assert first["status"] == "ok"
    assert first["warnings"] == []


def test_drift_reference_excludes_partial_inference_runs(tmp_path: Path) -> None:
    previous_features = tmp_path / "data" / "regimes" / "previous.parquet"
    previous_predictions = tmp_path / "data" / "predictions" / "previous.parquet"
    partial_features = tmp_path / "data" / "regimes" / "partial.parquet"
    partial_predictions = tmp_path / "data" / "predictions" / "partial.parquet"
    current_features = tmp_path / "data" / "regimes" / "current.parquet"
    current_predictions = tmp_path / "data" / "predictions" / "current.parquet"
    for path in (previous_features, partial_features, current_features):
        _write_parquet(path, _features(shifted=False))
    for path in (previous_predictions, partial_predictions, current_predictions):
        _write_parquet(path, _predictions(shifted=False))

    _write_run(
        tmp_path,
        run_ts="20260101_000000Z",
        timestamp=20260101000000,
        features_path=previous_features,
        predictions_path=previous_predictions,
    )
    partial_metadata_path = _write_run(
        tmp_path,
        run_ts="20260101_120000Z",
        timestamp=20260101120000,
        features_path=partial_features,
        predictions_path=partial_predictions,
    )
    partial_metadata = json.loads(partial_metadata_path.read_text(encoding="utf-8"))
    partial_metadata["failed_models"] = [{"model_name": "failed_shadow"}]
    partial_metadata_path.write_text(json.dumps(partial_metadata), encoding="utf-8")
    _write_run(
        tmp_path,
        run_ts="20260102_000000Z",
        timestamp=20260102000000,
        features_path=current_features,
        predictions_path=current_predictions,
    )

    reference = find_previous_successful_inference(
        project_root=tmp_path,
        runs_dir=tmp_path / "data" / "runs",
        current_predictions_path=current_predictions,
    )

    assert reference is not None
    assert reference.run_ts == "20260101_000000Z"


def test_drift_snapshot_records_insufficient_history_without_prior_output(tmp_path: Path) -> None:
    features = tmp_path / "data" / "features" / "current.parquet"
    regimes = tmp_path / "data" / "regimes" / "current.parquet"
    predictions = tmp_path / "data" / "predictions" / "current.parquet"
    _write_parquet(features, _features(shifted=False))
    _write_parquet(regimes, _features(shifted=False))
    _write_parquet(predictions, _predictions(shifted=False))

    snapshot = build_drift_snapshot(
        project_root=tmp_path,
        run_ts="20260101_000000Z",
        current_features_path=features,
        current_regimes_path=regimes,
        current_predictions_path=predictions,
        policy=DriftPolicy(window_size=40, min_samples=30),
    )

    assert snapshot["status"] == "insufficient_history"
    assert snapshot["action"] == "none"
    assert snapshot["reference"] is None


def test_prediction_drift_uses_a_full_window_per_long_form_model(tmp_path: Path) -> None:
    previous_features = tmp_path / "data" / "regimes" / "previous.parquet"
    previous_predictions = tmp_path / "data" / "predictions" / "previous.parquet"
    current_features = tmp_path / "data" / "features" / "current.parquet"
    current_regimes = tmp_path / "data" / "regimes" / "current.parquet"
    current_predictions = tmp_path / "data" / "predictions" / "current.parquet"
    _write_parquet(previous_features, _features(shifted=False))
    _write_parquet(previous_predictions, _multi_model_predictions())
    _write_parquet(current_features, _features(shifted=False))
    _write_parquet(current_regimes, _features(shifted=False))
    _write_parquet(current_predictions, _multi_model_predictions())
    _write_run(
        tmp_path,
        run_ts="20260101_000000Z",
        timestamp=20260101000000,
        features_path=previous_features,
        predictions_path=previous_predictions,
    )
    _write_run(
        tmp_path,
        run_ts="20260102_000000Z",
        timestamp=20260102000000,
        features_path=current_regimes,
        predictions_path=current_predictions,
    )

    snapshot = build_drift_snapshot(
        project_root=tmp_path,
        run_ts="20260102_000000Z",
        current_features_path=current_features,
        current_regimes_path=current_regimes,
        current_predictions_path=current_predictions,
        policy=DriftPolicy(window_size=40, min_samples=30),
    )

    assert snapshot["prediction_drift"]["status"] == "ok"
    assert [model["reference_n"] for model in snapshot["prediction_drift"]["models"]] == [40, 40]

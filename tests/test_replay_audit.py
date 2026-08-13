from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring.replay_audit import build_replay_audit, write_replay_audit


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_parquet(path: Path, prediction: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": [0],
            "model_name": ["baseline"],
            "model_source": ["baseline"],
            "is_active": [True],
            "y_pred": [prediction],
        }
    ).to_parquet(path, index=False)


def test_build_replay_audit_passes_when_replay_matches(tmp_path: Path) -> None:
    run_ts = "20260101_000000Z"
    _write_json(
        tmp_path / "artifacts" / "lineage" / f"lineage_{run_ts}.json",
        {
            "run_ts": run_ts,
            "artifacts": {
                "raw_csv": {"path": f"data/raw/{run_ts}.csv", "sha256": "raw"},
                "features_parquet": {
                    "path": f"data/features/{run_ts}.parquet",
                    "sha256": "features",
                },
                "regimes_parquet": {"path": f"data/regimes/{run_ts}.parquet", "sha256": "regimes"},
                "predictions_parquet": {
                    "path": f"data/predictions/predictions_{run_ts}.parquet",
                    "sha256": "predictions",
                },
            },
        },
    )
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / f"{run_ts}.csv").write_text("timestamp\n", encoding="utf-8")
    _write_parquet(tmp_path / "data" / "features" / f"{run_ts}.parquet", 0.1)
    _write_parquet(tmp_path / "data" / "regimes" / f"{run_ts}.parquet", 0.1)
    original_predictions = tmp_path / "data" / "predictions" / f"predictions_{run_ts}.parquet"
    replay_predictions = tmp_path / "data" / "predictions" / f"predictions_replay_{run_ts}.parquet"
    _write_parquet(original_predictions, 0.1)
    _write_parquet(replay_predictions, 0.1)

    from src.data.lineage import sha256_file

    lineage = {
        "run_ts": run_ts,
        "artifacts": {
            "raw_csv": {
                "path": f"data/raw/{run_ts}.csv",
                "sha256": sha256_file(tmp_path / "data" / "raw" / f"{run_ts}.csv"),
            },
            "features_parquet": {
                "path": f"data/features/{run_ts}.parquet",
                "sha256": sha256_file(tmp_path / "data" / "features" / f"{run_ts}.parquet"),
            },
            "regimes_parquet": {
                "path": f"data/regimes/{run_ts}.parquet",
                "sha256": sha256_file(tmp_path / "data" / "regimes" / f"{run_ts}.parquet"),
            },
            "predictions_parquet": {
                "path": f"data/predictions/predictions_{run_ts}.parquet",
                "sha256": sha256_file(original_predictions),
            },
        },
    }

    summary = build_replay_audit(
        project_root=tmp_path,
        run_ts=run_ts,
        lineage=lineage,
        replay_artifacts={
            "features_parquet": tmp_path / "data" / "features" / f"{run_ts}.parquet",
            "regimes_parquet": tmp_path / "data" / "regimes" / f"{run_ts}.parquet",
            "predictions_parquet": replay_predictions,
        },
    )
    assert summary["status"] == "passed"
    assert summary["semantic_pass"] is True


def test_build_replay_audit_records_prediction_drift(tmp_path: Path) -> None:
    run_ts = "20260102_000000Z"
    raw_path = tmp_path / "data" / "raw" / f"{run_ts}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("timestamp\n", encoding="utf-8")
    features_path = tmp_path / "data" / "features" / f"{run_ts}.parquet"
    regimes_path = tmp_path / "data" / "regimes" / f"{run_ts}.parquet"
    original_predictions = tmp_path / "data" / "predictions" / f"predictions_{run_ts}.parquet"
    replay_predictions = tmp_path / "data" / "predictions" / f"predictions_replay_{run_ts}.parquet"
    _write_parquet(features_path, 0.1)
    _write_parquet(regimes_path, 0.1)
    _write_parquet(original_predictions, 0.1)
    _write_parquet(replay_predictions, 0.2)

    from src.data.lineage import sha256_file

    lineage = {
        "run_ts": run_ts,
        "artifacts": {
            "raw_csv": {"path": f"data/raw/{run_ts}.csv", "sha256": sha256_file(raw_path)},
            "features_parquet": {
                "path": f"data/features/{run_ts}.parquet",
                "sha256": sha256_file(features_path),
            },
            "regimes_parquet": {
                "path": f"data/regimes/{run_ts}.parquet",
                "sha256": sha256_file(regimes_path),
            },
            "predictions_parquet": {
                "path": f"data/predictions/predictions_{run_ts}.parquet",
                "sha256": sha256_file(original_predictions),
            },
        },
    }

    summary = build_replay_audit(
        project_root=tmp_path,
        run_ts=run_ts,
        lineage=lineage,
        replay_artifacts={
            "features_parquet": features_path,
            "regimes_parquet": regimes_path,
            "predictions_parquet": replay_predictions,
        },
    )
    out_path = tmp_path / "artifacts" / "replay" / f"replay_{run_ts}.json"
    write_replay_audit(out_path, summary)

    assert summary["status"] == "failed"
    assert summary["semantic_pass"] is False
    assert summary["max_prediction_drift"] == 0.1
    assert json.loads(out_path.read_text(encoding="utf-8"))["run_ts"] == run_ts

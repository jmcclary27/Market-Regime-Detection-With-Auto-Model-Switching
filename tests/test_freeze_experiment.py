from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.experiment.freeze import FreezeConfig, freeze_experiment
from src.experiment.frozen_inference import run_frozen_stage
from src.experiment.manifest import load_manifest
from src.features.manifest import FeatureManifest, dataframe_sha256, schema_from_df, write_manifest


def _fixture(tmp_path: Path) -> FreezeConfig:
    n = 105
    timestamps = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "log_return_1_x": [((index % 9) - 4) / 1000 for index in range(n)],
            "f1": [float(index) for index in range(n)],
            "f2": [float((index * 7) % 13) for index in range(n)],
        }
    )
    regimes = pd.DataFrame(
        {
            "timestamp": timestamps,
            "regime": (["bullish"] * 35) + (["sideways"] * 35) + (["bearish"] * 35),
        }
    )
    features_path = tmp_path / "features.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    features.to_parquet(features_path, index=False)
    regimes.to_parquet(regimes_path, index=False)
    from src.features.manifest import (
        FeatureManifest,
        dataframe_sha256,
        schema_from_df,
        write_manifest,
    )

    source_manifest = tmp_path / "features.manifest.json"
    write_manifest(
        FeatureManifest(
            timestamp="fixture",
            parquet_path=str(features_path),
            row_count=len(features),
            columns=schema_from_df(features),
            content_sha256=dataframe_sha256(features),
        ),
        source_manifest,
    )
    hmm = tmp_path / "hmm" / "latest"
    hmm.mkdir(parents=True)
    for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json"):
        (hmm / name).write_bytes(b"fixture")
    return FreezeConfig(
        experiment_id="frozen-daily-spy-test",
        official_start_date=date(2026, 5, 1),
        data_cutoff=date(2026, 4, 15),
        features_path=features_path,
        regimes_path=regimes_path,
        feature_manifest_path=source_manifest,
        hmm_artifacts_dir=hmm.parent,
        output_dir=tmp_path / "experiment",
        seed=17,
    )


def test_freeze_creates_immutable_bundle_without_live_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fixture(tmp_path)
    monkeypatch.setattr("src.experiment.freeze._validate_hmm", lambda *_args: None)
    active = tmp_path / "registry" / "active_model.yaml"
    active.parent.mkdir()
    active.write_text("user-owned pointer\n", encoding="utf-8")

    result = freeze_experiment(cfg)

    manifest = load_manifest(cfg.output_dir / "manifest.json")
    assert result["status"] == "frozen"
    assert manifest.schema_version == 2
    assert manifest.freeze_status == "frozen"
    assert set(manifest.regime_models) == {"bullish", "sideways", "bearish"}
    assert manifest.static_model.model_id in {"global_ridge", "global_lightgbm"}
    assert (cfg.output_dir / "model_bundle.tar.gz").is_file()
    assert not (cfg.output_dir / "candidates").exists()
    assert active.read_text(encoding="utf-8") == "user-owned pointer\n"

    original_manifest = (cfg.output_dir / "manifest.json").read_bytes()
    original_bundle = (cfg.output_dir / "model_bundle.tar.gz").read_bytes()
    repeated = freeze_experiment(cfg)
    assert repeated["status"] == "already_frozen"
    assert (cfg.output_dir / "manifest.json").read_bytes() == original_manifest
    assert (cfg.output_dir / "model_bundle.tar.gz").read_bytes() == original_bundle


def test_freeze_ignores_rows_after_the_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fixture(tmp_path)
    monkeypatch.setattr("src.experiment.freeze._validate_hmm", lambda *_args: None)
    freeze_experiment(cfg)
    changed = pd.read_parquet(cfg.features_path)
    changed.loc[len(changed)] = {
        "timestamp": pd.Timestamp("2026-04-16", tz="UTC"),
        "log_return_1_x": 0.01,
        "f1": 999.0,
        "f2": 1.0,
    }
    changed.to_parquet(cfg.features_path, index=False)
    write_manifest(
        FeatureManifest(
            "fixture",
            str(cfg.features_path),
            len(changed),
            schema_from_df(changed),
            dataframe_sha256(changed),
        ),
        cfg.feature_manifest_path,
    )
    assert freeze_experiment(cfg)["status"] == "already_frozen"


def test_frozen_bundle_emits_manifest_model_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fixture(tmp_path)
    monkeypatch.setattr("src.experiment.freeze._validate_hmm", lambda *_args: None)
    freeze_experiment(cfg)
    bundle_root = tmp_path / "extracted"
    with tarfile.open(cfg.output_dir / "model_bundle.tar.gz", "r:gz") as archive:
        archive.extractall(bundle_root)
    predictions = run_frozen_stage(
        features_path=cfg.features_path,
        bundle_root=bundle_root,
        output_dir=tmp_path / "predictions",
        runs_dir=tmp_path / "runs",
        inference_ts=1,
        output_name="predictions.parquet",
        run_meta_name="run.json",
        record_features_path="fixture",
    )
    manifest = load_manifest(cfg.output_dir / "manifest.json")
    frame = pd.read_parquet(predictions)
    assert set(frame["model_name"]) == {
        manifest.static_model.model_id,
        *(ref.model_id for ref in manifest.regime_models.values()),
    }


class _FakeS3:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def put_object(
        self, *, Bucket: str, Key: str, Body: object, **_kwargs: object
    ) -> dict[str, str]:
        assert Bucket == "fixture-bucket"
        assert hasattr(Body, "read")
        self.calls.append(Key)
        return {"VersionId": f"version-{len(self.calls)}"}


def test_publish_uploads_bundle_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fixture(tmp_path)
    cfg = FreezeConfig(
        **{
            **cfg.__dict__,
            "publish_s3": True,
            "s3_bucket": "fixture-bucket",
            "s3_bundle_key": "inference/model-bundles/frozen.tar.gz",
            "s3_manifest_key": "experiment/manifest.json",
        }
    )
    monkeypatch.setattr("src.experiment.freeze._validate_hmm", lambda *_args: None)
    s3 = _FakeS3()
    result = freeze_experiment(cfg, s3_client=s3)
    assert s3.calls == ["inference/model-bundles/frozen.tar.gz", "experiment/manifest.json"]
    assert result["published"]["bundle"]["version_id"] == "version-1"
    manifest = json.loads((cfg.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["s3_bundle"]["sha256"]
        == hashlib.sha256((cfg.output_dir / "model_bundle.tar.gz").read_bytes()).hexdigest()
    )

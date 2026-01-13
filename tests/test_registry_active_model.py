# tests/test_registry_active_model.py
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.registry.registry import (
    ActiveModelRef,
    RegistryError,
    load_active_model,
    read_active,
    write_active,
)


def test_registry_read_write_roundtrip(tmp_path: Path) -> None:
    active_file = tmp_path / "registry" / "active_model.yaml"

    ref = ActiveModelRef(
        model_type="expert",
        regime="bullish",
        model_id="ridge",
        version="1767565054",
        artifact_path=Path("models/experts/bullish/1767565054/model.joblib"),
        metadata_path=Path("models/experts/bullish/1767565054/metadata.json"),
        updated_at="2026-01-08T19:05:00Z",
    )

    write_active(ref, active_file=active_file)
    loaded = read_active(active_file=active_file)

    assert loaded.model_type == "expert"
    assert loaded.regime == "bullish"
    assert loaded.model_id == "ridge"
    assert loaded.version == "1767565054"
    assert loaded.artifact_path.as_posix() == "models/experts/bullish/1767565054/model.joblib"
    assert loaded.metadata_path is not None
    assert loaded.metadata_path.as_posix() == "models/experts/bullish/1767565054/metadata.json"
    assert loaded.updated_at == "2026-01-08T19:05:00Z"


def test_load_active_model_loads_joblib(tmp_path: Path) -> None:
    """
    Create a fake model artifact and point the registry at it,
    then ensure load_active_model returns it.
    """
    # Arrange fake filesystem layout
    repo_root = tmp_path
    (repo_root / "registry").mkdir(parents=True, exist_ok=True)
    model_dir = repo_root / "models" / "experts" / "bullish" / "123"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    dummy_model = {"hello": "world"}
    joblib.dump(dummy_model, model_path)

    # Create active pointer yaml referencing the dummy model
    active_file = repo_root / "registry" / "active_model.yaml"
    ref = ActiveModelRef(
        model_type="expert",
        regime="bullish",
        model_id="ridge",
        version="123",
        artifact_path=model_path,
        metadata_path=None,
        updated_at="2026-01-08T19:05:00Z",
    )
    write_active(ref, active_file=active_file)

    # Act: change cwd so relative paths work the same way as your repo
    # (pytest runs from repo root normally, this simulates that)
    import os

    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        model, metadata, loaded_ref = load_active_model(
            active_file=Path("registry/active_model.yaml")
        )
    finally:
        os.chdir(old_cwd)

    # Assert
    assert model == dummy_model
    assert metadata is None
    assert loaded_ref.model_type == "expert"
    assert loaded_ref.regime == "bullish"
    assert loaded_ref.version == "123"


def test_read_active_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        read_active(active_file=tmp_path / "registry" / "active_model.yaml")

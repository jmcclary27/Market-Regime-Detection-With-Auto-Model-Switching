"""Immutable manifest used to freeze an out-of-sample experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when an experiment manifest is incomplete or mutable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    version: str
    model_id: str = ""

    @classmethod
    def from_file(cls, path: Path, *, version: str, model_id: str = "") -> ArtifactRef:
        if not path.is_file():
            raise ManifestError(f"Artifact does not exist: {path}")
        return cls(
            path=path.as_posix(), sha256=sha256_file(path), version=version, model_id=model_id
        )


@dataclass(frozen=True)
class FrozenExperimentManifest:
    schema_version: int
    experiment_id: str
    created_at_utc: str
    data_cutoff: str
    symbols: tuple[str, str]
    traded_symbol: str
    starting_cash: float
    fee_bps: float
    slippage_bps: float
    exposure_thresholds: tuple[float, float]
    static_model: ArtifactRef
    regime_models: dict[str, ArtifactRef]
    regime_detector: ArtifactRef
    feature_manifest: ArtifactRef
    git_commit: str
    model_bundle: ArtifactRef | None = None
    # Schema v2 fields.  Defaults preserve the already-deployed v1 reader
    # contract while the freeze workflow requires all of them.
    official_start_date: str | None = None
    freeze_status: str | None = None
    input_fingerprint: str | None = None
    selection_scorecard: dict[str, Any] | None = None
    frozen_feature_manifest: ArtifactRef | None = None
    hmm_state_mapping: ArtifactRef | None = None
    s3_bundle: dict[str, str] | None = None

    def validate(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ManifestError("schema_version must equal 1 or 2")
        if not self.experiment_id.strip():
            raise ManifestError("experiment_id is required")
        if self.symbols != ("SPY", "QQQ") or self.traded_symbol != "SPY":
            raise ManifestError("the v1 experiment requires symbols (SPY, QQQ) and trades SPY")
        if self.starting_cash <= 0 or self.fee_bps < 0 or self.slippage_bps < 0:
            raise ManifestError(
                "cash and trading frictions must be non-negative, with positive cash"
            )
        low, high = self.exposure_thresholds
        if not low < high:
            raise ManifestError("exposure thresholds must be ordered")
        if set(self.regime_models) != {"bullish", "sideways", "bearish"}:
            raise ManifestError("regime_models must contain bullish, sideways, and bearish")
        for ref in [
            self.static_model,
            self.regime_detector,
            self.feature_manifest,
            *self.regime_models.values(),
        ]:
            if len(ref.sha256) != 64 or not ref.version:
                raise ManifestError("every frozen artifact requires a SHA-256 and version")
        if not self.static_model.model_id or any(
            not ref.model_id for ref in self.regime_models.values()
        ):
            raise ManifestError("static and regime model artifacts require immutable model ids")
        if self.schema_version == 2:
            if not self.official_start_date or not self.freeze_status:
                raise ManifestError("v2 manifests require official_start_date and freeze_status")
            if self.freeze_status != "frozen":
                raise ManifestError("v2 manifest freeze_status must equal 'frozen'")
            if not self.input_fingerprint or len(self.input_fingerprint) != 64:
                raise ManifestError("v2 manifests require a SHA-256 input_fingerprint")
            if not isinstance(self.selection_scorecard, dict):
                raise ManifestError("v2 manifests require the selected scorecard row")
            if self.frozen_feature_manifest is None or self.hmm_state_mapping is None:
                raise ManifestError("v2 manifests require frozen feature and HMM mapping refs")
            for ref in (self.frozen_feature_manifest, self.hmm_state_mapping):
                if len(ref.sha256) != 64 or not ref.version:
                    raise ManifestError("v2 frozen references require SHA-256 and version")
            if self.s3_bundle is not None:
                required = {"bucket", "key", "version_id", "sha256"}
                if set(self.s3_bundle) != required or len(self.s3_bundle["sha256"]) != 64:
                    raise ManifestError("s3_bundle must be a complete versioned object reference")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def freeze_manifest(path: Path, manifest: FrozenExperimentManifest) -> str:
    """Write a canonical manifest once; later writes must be byte-identical."""
    manifest.validate()
    payload = json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise ManifestError(f"Frozen manifest cannot be changed: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return sha256_file(path)


def load_manifest(path: Path) -> FrozenExperimentManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["symbols"] = tuple(raw["symbols"])
        raw["exposure_thresholds"] = tuple(raw["exposure_thresholds"])
        for key in (
            "static_model",
            "regime_detector",
            "feature_manifest",
            "model_bundle",
            "frozen_feature_manifest",
            "hmm_state_mapping",
        ):
            if raw.get(key) is not None:
                raw[key] = ArtifactRef(**raw[key])
        raw["regime_models"] = {
            key: ArtifactRef(**value) for key, value in raw["regime_models"].items()
        }
        manifest = FrozenExperimentManifest(**raw)
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Invalid manifest: {path}") from exc
    manifest.validate()
    return manifest


def utc_now() -> str:
    return datetime.now(UTC).isoformat()

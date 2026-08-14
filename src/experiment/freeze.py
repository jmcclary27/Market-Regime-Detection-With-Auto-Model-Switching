"""Deterministic, registry-free artifact freezing for the daily experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.experiment.manifest import ArtifactRef, FrozenExperimentManifest, sha256_file
from src.experiment.selection import select_static_model
from src.features.manifest import FeatureManifest, dataframe_sha256, schema_from_df, write_manifest
from src.regimes.hmm import label_regimes_hmm

REQUIRED_REGIMES = ("bullish", "sideways", "bearish")
TARGET_COL = "target_next_return"
RETURN_COL = "log_return_1_x"


@dataclass(frozen=True)
class FreezeConfig:
    experiment_id: str
    official_start_date: date
    data_cutoff: date
    features_path: Path
    regimes_path: Path
    feature_manifest_path: Path
    hmm_artifacts_dir: Path
    output_dir: Path
    seed: int = 42
    publish_s3: bool = False
    s3_bucket: str | None = None
    s3_bundle_key: str | None = None
    s3_manifest_key: str | None = None


class FreezeError(RuntimeError):
    """Raised when an experiment cannot safely be frozen."""


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_parquet(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FreezeError(f"{label} does not exist: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise FreezeError(f"{label} is not readable parquet: {path}") from exc


def _timestamps(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise FreezeError(f"{label} is missing timestamp")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any() or out["timestamp"].duplicated().any():
        raise FreezeError(f"{label} has invalid or duplicate timestamps")
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _load_feature_manifest(path: Path, features: pd.DataFrame) -> None:
    if not path.is_file():
        raise FreezeError(f"feature manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"feature manifest is unreadable: {path}") from exc
    expected = payload.get("content_sha256") if isinstance(payload, dict) else None
    actual = dataframe_sha256(features)
    if not isinstance(expected, str) or expected != actual:
        raise FreezeError("feature manifest content_sha256 does not match --features-path")


def _validate_hmm(root: Path, features: pd.DataFrame, supplied: pd.DataFrame) -> None:
    required = [
        root / "latest" / name
        for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json")
    ]
    if any(not path.is_file() for path in required):
        raise FreezeError("frozen HMM artifacts are incomplete")
    try:
        mapping = json.loads((root / "latest" / "state_mapping.json").read_text(encoding="utf-8"))
        if set(str(value) for value in mapping.values()) != set(REQUIRED_REGIMES):
            raise FreezeError("HMM state mapping must cover bullish, sideways, and bearish exactly")
        labeled = label_regimes_hmm(
            features, cfg={"regimes": {"hmm": {"artifacts_dir": str(root)}}}
        )
    except FreezeError:
        raise
    except Exception as exc:
        raise FreezeError("frozen HMM artifacts cannot label the cutoff features") from exc
    expected = supplied.set_index("timestamp")["regime"].astype("string").str.lower()
    actual = (
        pd.Series(labeled["regime"].to_numpy(), index=features["timestamp"])
        .astype("string")
        .str.lower()
    )
    shared = expected.reindex(actual.index)
    if shared.isna().any() or not shared.equals(actual):
        raise FreezeError("--regimes-path does not match labels from --hmm-artifacts-dir")


def _input_fingerprint(cfg: FreezeConfig, *, features: pd.DataFrame, regimes: pd.DataFrame) -> str:
    sources = {
        "experiment_id": cfg.experiment_id,
        "official_start_date": cfg.official_start_date.isoformat(),
        "data_cutoff": cfg.data_cutoff.isoformat(),
        "seed": cfg.seed,
        # Only cutoff-scoped data participates in the frozen identity. Appending
        # later market rows must not invalidate an already frozen study.
        "features": dataframe_sha256(features),
        "regimes": dataframe_sha256(regimes),
        "hmm": {
            name: sha256_file(cfg.hmm_artifacts_dir / "latest" / name)
            for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json")
        },
    }
    return _sha256_bytes(_canonical_json(sources))


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "regime", "regime_explanation", TARGET_COL, RETURN_COL}
    columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise FreezeError("no numeric feature columns remain after leakage-safe exclusions")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise FreezeError("feature inputs contain non-finite values")
    return columns


def _fit_model(kind: str, x: pd.DataFrame, y: pd.Series, seed: int) -> Any:
    if kind == "ridge":
        model: Any = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=1.0))]
        )
    elif kind == "lightgbm":
        model = LGBMRegressor(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            verbosity=-1,
        )
    else:
        raise FreezeError(f"unsupported candidate type: {kind}")
    model.fit(x, y)
    return model


def _portfolio_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise FreezeError("walk-forward predictions or targets are non-finite")
    returns = np.clip(predictions, -1.0, 1.0) * targets
    if len(returns) < 2:
        raise FreezeError("walk-forward fold is too short")
    std = float(np.std(returns, ddof=1))
    sharpe = 0.0 if std <= np.finfo(float).eps else float(np.mean(returns) / std * math.sqrt(252))
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "walk_forward_net_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "cumulative_return": float(equity[-1] - 1.0),
    }


def _walk_forward(
    frame: pd.DataFrame, columns: list[str], kind: str, seed: int
) -> dict[str, float]:
    n = len(frame)
    train_size = max(20, n // 2)
    test_size = max(5, n // 6)
    if n < train_size + test_size:
        raise FreezeError("insufficient rows for chronological walk-forward evaluation")
    predictions: list[float] = []
    targets: list[float] = []
    for test_start in range(train_size, n - test_size + 1, test_size):
        train = frame.iloc[:test_start]
        test = frame.iloc[test_start : test_start + test_size]
        model = _fit_model(kind, train[columns], train[TARGET_COL], seed)
        predictions.extend(np.asarray(model.predict(test[columns]), dtype=float))
        targets.extend(test[TARGET_COL].to_numpy(dtype=float))
    return _portfolio_metrics(np.asarray(predictions), np.asarray(targets))


def _write_candidate(
    *,
    root: Path,
    model_id: str,
    kind: str,
    regime: str | None,
    frame: pd.DataFrame,
    columns: list[str],
    seed: int,
    fingerprint: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if len(frame) < 25:
        raise FreezeError(f"candidate {model_id} has insufficient rows: {len(frame)}")
    metrics = _walk_forward(frame, columns, kind, seed)
    model = _fit_model(kind, frame[columns], frame[TARGET_COL], seed)
    candidate = root / model_id
    candidate.mkdir(parents=True, exist_ok=False)
    model_path = candidate / "model.joblib"
    joblib.dump({"model": model, "feature_cols": columns}, model_path)
    metadata = {
        "artifact_contract_version": 2,
        "candidate_only": True,
        "promotion_eligible": True,
        "model_id": model_id,
        "model_type": kind,
        "regime": regime,
        "version": fingerprint[:16],
        "feature_columns": columns,
        "target_col": TARGET_COL,
        "target_horizon": "next_period",
        "metrics": metrics,
    }
    metadata_path = candidate / "metadata.json"
    metadata_path.write_bytes(_canonical_json(metadata))
    return model_path, metadata_path, metrics


def _artifact_ref(path: Path, root: Path, *, version: str, model_id: str = "") -> ArtifactRef:
    return ArtifactRef(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        version=version,
        model_id=model_id,
    )


def _copy_frozen_candidate(source: Path, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    model_path = destination / "model.joblib"
    metadata_path = destination / "metadata.json"
    shutil.copy2(source / "model.joblib", model_path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["candidate_only"] = False
    metadata["publication_scope"] = "frozen_experiment"
    metadata_path.write_bytes(_canonical_json(metadata))
    return model_path, metadata_path


def _iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def _build_bundle(package_root: Path, output: Path) -> dict[str, Any]:
    staging = package_root / "bundle_root"
    files = _iter_files(staging)
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    (staging / "bundle_manifest.json").write_bytes(_canonical_json(manifest))
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in _iter_files(staging):
                    info = archive.gettarinfo(
                        str(path), arcname=path.relative_to(staging).as_posix()
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return {"path": output.name, "sha256": sha256_file(output), "size_bytes": output.stat().st_size}


def _s3_ref(s3: Any, *, bucket: str, key: str, path: Path, content_type: str) -> dict[str, str]:
    expected_sha256 = sha256_file(path)
    head_object = getattr(s3, "head_object", None)
    if callable(head_object):
        try:
            existing = head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise FreezeError(
                    f"could not verify existing S3 object s3://{bucket}/{key}"
                ) from exc
        else:
            metadata = existing.get("Metadata", {})
            version = existing.get("VersionId")
            if (
                metadata.get("sha256") != expected_sha256
                or not isinstance(version, str)
                or not version
            ):
                raise FreezeError("existing S3 key does not match the frozen object identity")
            return {"bucket": bucket, "key": key, "version_id": version, "sha256": expected_sha256}
    with path.open("rb") as handle:
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=handle,
            ContentType=content_type,
            Metadata={"sha256": expected_sha256},
        )
    version = response.get("VersionId")
    if not isinstance(version, str) or not version or version.lower() == "null":
        raise FreezeError("S3 bucket must return a non-null VersionId")
    return {"bucket": bucket, "key": key, "version_id": version, "sha256": expected_sha256}


def _publish(
    cfg: FreezeConfig, *, package: Path, manifest: FrozenExperimentManifest, s3: Any
) -> dict[str, dict[str, str]]:
    if not all((cfg.s3_bucket, cfg.s3_bundle_key, cfg.s3_manifest_key)):
        raise FreezeError(
            "--publish-s3 requires --s3-bucket, --s3-bundle-key, and --s3-manifest-key"
        )
    bundle = _s3_ref(
        s3,
        bucket=str(cfg.s3_bucket),
        key=str(cfg.s3_bundle_key),
        path=package / "model_bundle.tar.gz",
        content_type="application/gzip",
    )
    # The manifest is intentionally written after a versioned bundle reference exists.
    published = replace(manifest, s3_bundle=bundle)
    published.validate()
    manifest_path = package / "manifest.json"
    manifest_path.write_bytes(_canonical_json(published.as_dict()))
    manifest_ref = _s3_ref(
        s3,
        bucket=str(cfg.s3_bucket),
        key=str(cfg.s3_manifest_key),
        path=manifest_path,
        content_type="application/json",
    )
    return {"bundle": bundle, "manifest": manifest_ref}


def freeze_experiment(cfg: FreezeConfig, *, s3_client: Any | None = None) -> dict[str, Any]:
    if not cfg.experiment_id.strip() or cfg.official_start_date <= cfg.data_cutoff:
        raise FreezeError(
            "experiment_id is required and official_start_date must be after data_cutoff"
        )
    if cfg.publish_s3 and s3_client is None:
        try:
            import boto3

            s3_client = boto3.client("s3")
        except Exception as exc:  # pragma: no cover - boto3/environment dependent
            raise FreezeError("unable to create S3 client for --publish-s3") from exc
    features_all = _timestamps(_read_parquet(cfg.features_path, label="features"), label="features")
    regimes_all = _timestamps(_read_parquet(cfg.regimes_path, label="regimes"), label="regimes")
    if "regime" not in regimes_all.columns:
        raise FreezeError("regimes is missing regime")
    _load_feature_manifest(cfg.feature_manifest_path, features_all)
    cutoff = (
        pd.Timestamp(cfg.data_cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    )
    features = features_all.loc[features_all["timestamp"] <= cutoff].reset_index(drop=True)
    regimes = regimes_all.loc[
        regimes_all["timestamp"] <= cutoff, ["timestamp", "regime"]
    ].reset_index(drop=True)
    if features.empty or regimes.empty:
        raise FreezeError("cutoff removes all feature or regime rows")
    _validate_hmm(cfg.hmm_artifacts_dir, features, regimes)
    fingerprint = _input_fingerprint(cfg, features=features, regimes=regimes)
    if cfg.output_dir.exists():
        manifest_path = cfg.output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FreezeError(f"refusing to overwrite incomplete freeze output: {cfg.output_dir}")
        from src.experiment.manifest import load_manifest

        existing = load_manifest(manifest_path)
        if existing.input_fingerprint != fingerprint:
            raise FreezeError(
                "frozen output identity has different inputs, dates, or artifact hashes"
            )
        bundle = cfg.output_dir / "model_bundle.tar.gz"
        if (
            not bundle.is_file()
            or existing.model_bundle is None
            or sha256_file(bundle) != existing.model_bundle.sha256
        ):
            raise FreezeError("existing frozen bundle is missing or does not match its manifest")
        return {"status": "already_frozen", "manifest": str(manifest_path), "bundle": str(bundle)}

    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{cfg.output_dir.name}.freeze-", dir=cfg.output_dir.parent
    ) as temporary:
        package = Path(temporary) / cfg.output_dir.name
        candidates = package / "candidates"
        candidates.mkdir(parents=True)
        merged = features.merge(regimes, on="timestamp", how="inner", validate="one_to_one")
        merged[TARGET_COL] = pd.to_numeric(merged[RETURN_COL], errors="coerce").shift(-1)
        merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        if not np.isfinite(merged[TARGET_COL].to_numpy(dtype=float)).all():
            raise FreezeError("next-period target contains non-finite values")
        columns = _feature_columns(merged)
        definitions = [("global_ridge", "ridge", None), ("global_lightgbm", "lightgbm", None)] + [
            (f"expert_lightgbm_{regime}", "lightgbm", regime) for regime in REQUIRED_REGIMES
        ]
        score_rows: list[dict[str, Any]] = []
        for model_id, kind, regime in definitions:
            subset = (
                merged
                if regime is None
                else merged.loc[
                    merged["regime"].astype("string").str.lower() == regime
                ].reset_index(drop=True)
            )
            _, _, metrics = _write_candidate(
                root=candidates,
                model_id=model_id,
                kind=kind,
                regime=regime,
                frame=subset,
                columns=columns,
                seed=cfg.seed,
                fingerprint=fingerprint,
            )
            score_rows.append(
                {"model_id": model_id, "is_global_candidate": regime is None, **metrics}
            )
        scorecard = pd.DataFrame(score_rows)
        selected = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in select_static_model(scorecard).items()
            if key != "_drawdown_magnitude"
        }
        version = fingerprint[:16]
        root = package / "bundle_root"
        static_source = candidates / str(selected["model_id"])
        static_model, _ = _copy_frozen_candidate(
            static_source, root / "models" / "static" / str(selected["model_id"])
        )
        regime_refs: dict[str, ArtifactRef] = {}
        for regime in REQUIRED_REGIMES:
            model_id = f"expert_lightgbm_{regime}"
            model, _ = _copy_frozen_candidate(
                candidates / model_id, root / "models" / "experts" / regime / model_id
            )
            regime_refs[regime] = _artifact_ref(model, root, version=version, model_id=model_id)
        hmm_destination = root / "models" / "regimes" / "hmm" / "latest"
        hmm_destination.mkdir(parents=True)
        for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json"):
            shutil.copy2(cfg.hmm_artifacts_dir / "latest" / name, hmm_destination / name)
        frozen_features = root / "features" / "feature_manifest.json"
        frozen_features.parent.mkdir(parents=True)
        write_manifest(
            FeatureManifest(
                timestamp=cfg.data_cutoff.isoformat(),
                parquet_path="cutoff_features.parquet",
                row_count=len(features),
                columns=schema_from_df(features),
                content_sha256=dataframe_sha256(features),
            ),
            frozen_features,
        )
        bundle_descriptor = {
            "schema_version": 1,
            "experiment_id": cfg.experiment_id,
            "static_model_id": str(selected["model_id"]),
            "regime_model_ids": {regime: ref.model_id for regime, ref in regime_refs.items()},
        }
        (root / "experiment_bundle.json").write_bytes(_canonical_json(bundle_descriptor))
        bundle_info = _build_bundle(package, package / "model_bundle.tar.gz")
        static_ref = _artifact_ref(
            static_model, root, version=version, model_id=str(selected["model_id"])
        )
        hmm_ref = _artifact_ref(hmm_destination / "model.joblib", root, version=version)
        mapping_ref = _artifact_ref(hmm_destination / "state_mapping.json", root, version=version)
        feature_ref = _artifact_ref(frozen_features, root, version=version)
        manifest = FrozenExperimentManifest(
            schema_version=2,
            experiment_id=cfg.experiment_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            official_start_date=cfg.official_start_date.isoformat(),
            data_cutoff=cfg.data_cutoff.isoformat(),
            symbols=("SPY", "QQQ"),
            traded_symbol="SPY",
            starting_cash=100_000.0,
            fee_bps=1.0,
            slippage_bps=2.0,
            exposure_thresholds=(-0.001, 0.001),
            static_model=static_ref,
            regime_models=regime_refs,
            regime_detector=hmm_ref,
            feature_manifest=feature_ref,
            frozen_feature_manifest=feature_ref,
            hmm_state_mapping=mapping_ref,
            git_commit=_git_commit(),
            model_bundle=ArtifactRef(
                path=bundle_info["path"], sha256=str(bundle_info["sha256"]), version=version
            ),
            freeze_status="frozen",
            input_fingerprint=fingerprint,
            selection_scorecard=selected,
        )
        manifest.validate()
        (package / "selection_scorecard.json").write_bytes(_canonical_json(score_rows))
        (package / "manifest.json").write_bytes(_canonical_json(manifest.as_dict()))
        published = (
            _publish(cfg, package=package, manifest=manifest, s3=s3_client)
            if cfg.publish_s3
            else None
        )
        shutil.rmtree(candidates)
        package.rename(cfg.output_dir)
    return {
        "status": "frozen",
        "manifest": str(cfg.output_dir / "manifest.json"),
        "bundle": str(cfg.output_dir / "model_bundle.tar.gz"),
        "published": published,
    }

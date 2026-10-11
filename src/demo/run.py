"""Run the isolated deterministic end-to-end market-regime demonstration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.inference.batch_predict import discover_models, run_stage
from src.models.promotion import PromotionConfig, decide_promotion, summarize_walkforward
from src.registry.registry import ActiveModelRef, write_active

DEMO_TIMESTAMP = "20240201_160000Z"
PROMOTION_TIMESTAMP = "20240202_160000Z"
REJECTION_TIMESTAMP = "20240203_160000Z"
DEMO_SYMBOLS = ("SPY", "QQQ")


class DemoPrerequisiteError(RuntimeError):
    """The configured source artifacts cannot support a safe complete demo."""


def build_demo_bars(rows: int = 900) -> pd.DataFrame:
    """Return deterministic, autocorrelated synthetic daily bars for two symbols."""
    if rows < 600:
        raise ValueError("rows must be at least 600 for deterministic demo evaluation")
    dates = pd.date_range("2022-01-03", periods=rows, freq="B", tz="UTC")
    frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(DEMO_SYMBOLS):
        steps = np.arange(rows, dtype=float)
        innovations = 0.0007 * np.sin(steps / (11.0 + index))
        returns = 0.0012 * np.sin(steps / (5.0 + index)) + innovations
        returns[1:] += 0.45 * returns[:-1]
        close = (100.0 + index * 50.0) * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "close": close,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "volume": 1_000_000 + index * 100_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def prepare_demo_models(*, source_models: Path, runtime_models: Path) -> dict[str, object]:
    """Snapshot existing HMM/expert artifacts without touching their source."""
    source_models = source_models.resolve()
    runtime_models = runtime_models.resolve()
    hmm_source = source_models / "regimes" / "hmm" / "latest"
    required_hmm = [
        hmm_source / name
        for name in ("model.joblib", "scaler.joblib", "state_mapping.json", "metadata.json")
    ]
    missing_hmm = [path.name for path in required_hmm if not path.is_file()]
    if missing_hmm:
        raise DemoPrerequisiteError(
            f"HMM source artifacts are incomplete under {hmm_source}: {', '.join(missing_hmm)}"
        )
    experts_source = source_models / "experts"
    if not experts_source.is_dir():
        raise DemoPrerequisiteError(f"No existing expert artifacts found at {experts_source}")
    if runtime_models.exists():
        raise DemoPrerequisiteError(
            f"Refusing to reuse existing demo runtime models at {runtime_models}; use a fresh runtime root."
        )

    runtime_models.mkdir(parents=True)
    shutil.copytree(experts_source, runtime_models / "experts")
    shutil.copytree(source_models / "regimes", runtime_models / "regimes")
    try:
        discovered = discover_models(runtime_models, require_published_model_contract=True)
    except Exception as exc:
        raise DemoPrerequisiteError(f"Published expert validation failed: {exc}") from exc
    experts = [spec for spec in discovered if spec["model_source"] == "expert"]
    if len(experts) < 3:
        raise DemoPrerequisiteError(
            "The complete demo needs three published eligible experts (initial active, promotion "
            f"challenger, rejection challenger); found {len(experts)}."
        )
    return {
        "source_models": str(source_models),
        "runtime_models": str(runtime_models),
        "source_hashes": _tree_hashes(source_models),
        "runtime_hashes": _tree_hashes(runtime_models),
        "experts": experts,
    }


def _expert_ref(spec: dict[str, str]) -> ActiveModelRef:
    artifact_path = Path(spec["model_path"])
    metadata_path = (
        artifact_path if artifact_path.suffix == ".json" else artifact_path.parent / "latest.json"
    )
    if not metadata_path.is_file():
        raise DemoPrerequisiteError(f"Published expert metadata is missing: {metadata_path}")
    return ActiveModelRef(
        model_type="expert",
        model_id=spec["model_name"],
        version="demo-source",
        artifact_path=artifact_path,
        regime=spec.get("regime") or None,
        metadata_path=metadata_path,
    )


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_scenario(root: Path, name: str, payload: dict[str, object]) -> Path:
    path = root / "artifacts" / "demo" / "scenarios" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _pipeline_config(root: Path, run_ts: str, scenario: str):
    from src.pipeline.run import PipelineConfig

    return PipelineConfig(
        project_root=root,
        data_dir=root / "data",
        run_ts=run_ts,
        mode="pipeline",
        offline=True,
        demo=True,
        demo_scenario=scenario,
    )


def _promotion_path(root: Path, run_ts: str) -> Path:
    return root / "data" / "walkforward" / f"promotion_{run_ts}.json"


def _select_rejected_challenger(walkforward_path: Path, active_model_id: str) -> str:
    metrics = pd.read_parquet(walkforward_path)
    config = PromotionConfig()
    active = summarize_walkforward(metrics, model_name=active_model_id, cfg=config)
    for candidate in sorted(set(metrics["model_name"].astype(str)) - {active_model_id}):
        decision = decide_promotion(
            challenger_summary=summarize_walkforward(metrics, model_name=candidate, cfg=config),
            incumbent_summary=active,
            cfg=config,
        )
        if not decision.promote:
            return str(candidate)
    raise DemoPrerequisiteError(
        "No genuinely rejected eligible challenger was produced by the real walk-forward policy."
    )


def run() -> int:
    """Execute promotion, post-switch inference, then a genuine rejection scenario."""
    from src.pipeline.run import run_pipeline
    from src.registry.registry import read_active

    root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    source_models = Path(
        os.environ.get(
            "DEMO_MODEL_SOURCE_ROOT", str(Path(__file__).resolve().parents[2] / "models")
        )
    ).resolve()
    if source_models == root / "models":
        raise DemoPrerequisiteError(
            "DEMO_MODEL_SOURCE_ROOT must be separate from PROJECT_ROOT/models to protect real state."
        )
    prepared = prepare_demo_models(source_models=source_models, runtime_models=root / "models")
    experts = prepared["experts"]
    assert isinstance(experts, list)
    # Prefer a tabular published expert for the initial active model. ARIMA
    # experts remain eligible shadows, but can legitimately be flat until
    # their history reaches the artifact's training threshold.
    initial_spec = next(
        (spec for spec in experts if spec.get("expert_kind") == "sklearn"),
        experts[0],
    )
    initial = _expert_ref(initial_spec)
    write_active(
        initial,
        active_file=root / "registry" / "active_model.yaml",
        event_context={"source": "demo_initial_state", "run_ts": DEMO_TIMESTAMP},
    )

    fixed_eval = {
        "EVAL_WF_TRAIN": "300",
        "EVAL_WF_VAL": "75",
        "EVAL_WF_TEST": "75",
        "EVAL_WF_STEP": "75",
    }
    with _environment(fixed_eval | {"PROMOTION_INCUMBENT_MODEL": initial.model_id}):
        run_pipeline(_pipeline_config(root, PROMOTION_TIMESTAMP, "promotion"))
    promotion = _read_json(_promotion_path(root, PROMOTION_TIMESTAMP))
    if not bool(promotion.get("promoted")):
        raise DemoPrerequisiteError(
            "The supplied experts did not produce a genuine promotion under the existing policy: "
            f"{promotion.get('reason')}"
        )
    promoted_id = str(promotion["challenger_model_name"])
    promoted_ref = read_active(root / "registry" / "active_model.yaml")
    if promoted_ref.model_id != promoted_id:
        raise RuntimeError("Promotion decision and active registry pointer disagree")

    post_path = run_stage(
        features_path=root / "data" / "regimes" / "latest.parquet",
        active_file=root / "registry" / "active_model.yaml",
        models_dir=root / "models",
        output_dir=root / "data" / "predictions",
        runs_dir=root / "data" / "runs",
        inference_ts=20240202170000,
        output_name="predictions_post_promotion.parquet",
        latest_name="latest.parquet",
        run_meta_name="run_post_promotion.json",
        include_discovered_models=True,
    )
    post_predictions = pd.read_parquet(post_path)
    active_rows = post_predictions.loc[post_predictions["is_active"].astype(bool)]
    if active_rows.empty or set(active_rows["active_model_id"].astype(str)) != {promoted_id}:
        raise RuntimeError("Post-promotion inference did not serve the promoted registry model")
    _write_scenario(
        root,
        "promotion",
        {
            "schema_version": 1,
            "scenario": "promotion",
            "status": "completed",
            "synthetic_fixture_sha256": hashlib.sha256(
                build_demo_bars().to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "model_snapshot": prepared,
            "initial_active_model_id": initial.model_id,
            "promotion": promotion,
            "post_promotion_inference": {
                "predictions_path": str(post_path),
                "verified_active_model_id": promoted_id,
                "active_prediction_rows": int(len(active_rows)),
            },
        },
    )

    rejection_candidate = _select_rejected_challenger(
        root / "data" / "walkforward" / f"portfolio_metrics_{PROMOTION_TIMESTAMP}.parquet",
        promoted_id,
    )
    with _environment(
        fixed_eval
        | {
            "PROMOTION_INCUMBENT_MODEL": promoted_id,
            "PROMOTION_CHALLENGER_MODEL": rejection_candidate,
        }
    ):
        run_pipeline(_pipeline_config(root, REJECTION_TIMESTAMP, "rejection"))
    rejection = _read_json(_promotion_path(root, REJECTION_TIMESTAMP))
    if bool(rejection.get("promoted")):
        raise RuntimeError("The rejection scenario unexpectedly promoted its challenger")
    final_ref = read_active(root / "registry" / "active_model.yaml")
    if final_ref.model_id != promoted_id:
        raise RuntimeError("Rejected challenger changed the active registry pointer")
    _write_scenario(
        root,
        "rejection",
        {
            "schema_version": 1,
            "scenario": "rejection",
            "status": "completed",
            "promotion": rejection,
            "initial_active_model_id": promoted_id,
            "final_active_model_id": final_ref.model_id,
            "post_promotion_inference": {"verified_active_model_id": promoted_id},
        },
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

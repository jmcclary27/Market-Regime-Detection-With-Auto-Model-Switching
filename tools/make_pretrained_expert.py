"""Train a validated Ridge regime expert as a candidate artifact.

Historically this script wrote directly to ``models/pretrained`` and trained on
as few as two rows. It now writes versioned candidates by default and only
publishes to an inference-scanned directory when explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

REGIMES = {"bullish", "bearish", "sideways"}


@dataclass(frozen=True)
class TrainConfig:
    features_path: Path = Path("data/features/latest.parquet")
    regimes_path: Path = Path("data/regimes/latest.parquet")
    regime: str = "bullish"
    return_col: str = "log_return_1"
    target_col: str = "target_next_return"
    target_shift: int = -1
    ridge_alpha: float = 1.0
    min_regime_rows: int = 200
    min_train_rows: int = 100
    min_test_rows: int = 25
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    output_dir: Path = Path("models/candidates/pretrained")
    model_name: str = "expert_bullish_ridge"
    publish: bool = False
    publish_dir: Path | None = None
    publish_name: str | None = None


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a regime-specific Ridge candidate.")
    parser.add_argument("--features-path", default=str(TrainConfig.features_path))
    parser.add_argument("--regimes-path", default=str(TrainConfig.regimes_path))
    parser.add_argument("--regime", choices=sorted(REGIMES), default=TrainConfig.regime)
    parser.add_argument("--return-col", default=TrainConfig.return_col)
    parser.add_argument("--target-col", default=TrainConfig.target_col)
    parser.add_argument("--target-shift", type=int, default=TrainConfig.target_shift)
    parser.add_argument("--ridge-alpha", type=float, default=TrainConfig.ridge_alpha)
    parser.add_argument("--min-regime-rows", type=int, default=TrainConfig.min_regime_rows)
    parser.add_argument("--min-train-rows", type=int, default=TrainConfig.min_train_rows)
    parser.add_argument("--min-test-rows", type=int, default=TrainConfig.min_test_rows)
    parser.add_argument("--train-frac", type=float, default=TrainConfig.train_frac)
    parser.add_argument("--val-frac", type=float, default=TrainConfig.val_frac)
    parser.add_argument("--test-frac", type=float, default=TrainConfig.test_frac)
    parser.add_argument("--output-dir", default=str(TrainConfig.output_dir))
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Explicitly copy the validated artifact into --publish-dir for inference discovery.",
    )
    parser.add_argument(
        "--publish-dir",
        default=None,
        help="Required with --publish; typically models/pretrained after review.",
    )
    parser.add_argument(
        "--publish-name",
        default=None,
        help="Published filename stem. Defaults to --model-name.",
    )
    args = parser.parse_args()
    if args.publish and not args.publish_dir:
        parser.error("--publish requires --publish-dir so the production destination is explicit")

    return TrainConfig(
        features_path=Path(args.features_path),
        regimes_path=Path(args.regimes_path),
        regime=str(args.regime),
        return_col=str(args.return_col),
        target_col=str(args.target_col),
        target_shift=int(args.target_shift),
        ridge_alpha=float(args.ridge_alpha),
        min_regime_rows=int(args.min_regime_rows),
        min_train_rows=int(args.min_train_rows),
        min_test_rows=int(args.min_test_rows),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        output_dir=Path(args.output_dir),
        model_name=str(args.model_name),
        publish=bool(args.publish),
        publish_dir=Path(args.publish_dir) if args.publish_dir else None,
        publish_name=str(args.publish_name) if args.publish_name else None,
    )


def _resolve_col(df: pd.DataFrame, base: str) -> str:
    """Resolve a long-form or wide-pair feature column without ambiguous suffix guessing."""
    for candidate in (base, f"{base}_x", f"{base}_y"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Expected '{base}' (or _x/_y variant). Available columns: {list(df.columns)}")


def _load_regime_labels(features: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if "timestamp" not in features.columns:
        raise KeyError("Features must contain timestamp for regime-label alignment.")
    if "regime" in features.columns:
        return features.copy()
    if not cfg.regimes_path.exists():
        raise FileNotFoundError(f"Regimes path not found: {cfg.regimes_path}")

    labels = pd.read_parquet(cfg.regimes_path)
    missing = {"timestamp", "regime"} - set(labels.columns)
    if missing:
        raise KeyError(
            f"Regimes path is missing {sorted(missing)}. Available columns: {sorted(labels.columns)}"
        )
    labels = labels.loc[:, ["timestamp", "regime"]].drop_duplicates("timestamp", keep="last")
    return features.merge(labels, on="timestamp", how="left", validate="many_to_one")


def _time_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Split is too small: n={n}, train={n_train}, val={n_val}, test={n_test}")
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


def _finite_nunique(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    return int(np.unique(finite).size) if finite.size else 0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _zero_return_quality_gate(
    y_val: np.ndarray,
    val_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
) -> dict[str, Any]:
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    zero_return_val_rmse = float(np.sqrt(np.mean(np.square(y_val))))
    zero_return_test_rmse = float(np.sqrt(np.mean(np.square(y_test))))
    reasons: list[str] = []
    if val_rmse > zero_return_val_rmse:
        reasons.append("validation_rmse_exceeds_zero_return_baseline")
    if test_rmse > zero_return_test_rmse:
        reasons.append("test_rmse_exceeds_zero_return_baseline")
    return {
        "promotion_eligible": not reasons,
        "val_rmse": val_rmse,
        "zero_return_val_rmse": zero_return_val_rmse,
        "test_rmse": test_rmse,
        "zero_return_test_rmse": zero_return_test_rmse,
        "reason": "; ".join(reasons) if reasons else None,
        "reasons": reasons,
    }


def _target_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        raise ValueError("Expert target has no finite values.")
    quantiles = finite.quantile([0.01, 0.5, 0.99])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "p01": float(quantiles.loc[0.01]),
        "p50": float(quantiles.loc[0.5]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(finite.max()),
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def prepare_training_frame(cfg: TrainConfig) -> tuple[pd.DataFrame, str]:
    """Build next-period targets before filtering labels to preserve the stated horizon."""
    if not cfg.features_path.exists():
        raise FileNotFoundError(f"Features path not found: {cfg.features_path}")
    if cfg.regime.strip().lower() not in REGIMES:
        raise ValueError(f"Unknown regime '{cfg.regime}'. Expected one of {sorted(REGIMES)}")

    features = pd.read_parquet(cfg.features_path)
    resolved_return = _resolve_col(features, cfg.return_col)
    df = _load_regime_labels(features, cfg)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[cfg.target_col] = df[resolved_return].shift(cfg.target_shift)
    target_numeric = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df.loc[np.isfinite(target_numeric)].reset_index(drop=True)

    normalized = df["regime"].astype("string").str.strip().str.lower()
    return df.loc[normalized == cfg.regime.strip().lower()].reset_index(drop=True), resolved_return


def run(cfg: TrainConfig) -> Path:
    """Write a versioned candidate and publish only when explicitly requested."""
    if cfg.min_regime_rows <= 0:
        raise ValueError("min_regime_rows must be >= 1")
    if cfg.min_train_rows < 2:
        raise ValueError("min_train_rows must be >= 2")
    if cfg.min_test_rows < 1:
        raise ValueError("min_test_rows must be >= 1")
    if cfg.publish and cfg.publish_dir is None:
        raise ValueError("publish_dir must be set when publish=True")

    expert_df, resolved_return = prepare_training_frame(cfg)
    if len(expert_df) < cfg.min_regime_rows:
        raise ValueError(
            f"Not enough rows for pretrained '{cfg.regime}' Ridge expert: "
            f"{len(expert_df)} < {cfg.min_regime_rows}. No artifact was written."
        )

    train_df, val_df, test_df = _time_split(expert_df, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    if len(train_df) < cfg.min_train_rows:
        raise ValueError(
            f"Ridge expert training split has {len(train_df)} rows, below "
            f"min_train_rows={cfg.min_train_rows}. No artifact was written."
        )
    if len(test_df) < cfg.min_test_rows:
        raise ValueError(
            f"Ridge expert test split has {len(test_df)} rows, below "
            f"min_test_rows={cfg.min_test_rows}. No artifact was written."
        )

    excluded = {"timestamp", "symbol", "regime", "regime_explanation", cfg.target_col}
    feature_cols = [
        column
        for column in expert_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(expert_df[column])
    ]
    if not feature_cols:
        raise ValueError("No numeric features remain after excluding labels and target.")

    model: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=cfg.ridge_alpha, random_state=42)),
        ]
    )
    model.fit(train_df[feature_cols].to_numpy(), train_df[cfg.target_col].to_numpy())
    val_pred = model.predict(val_df[feature_cols].to_numpy())
    test_pred = model.predict(test_df[feature_cols].to_numpy())
    val_pred_nunique = _finite_nunique(np.asarray(val_pred, dtype=float))
    test_pred_nunique = _finite_nunique(np.asarray(test_pred, dtype=float))
    if val_pred_nunique <= 1 or test_pred_nunique <= 1:
        raise ValueError(
            "Refusing to save Ridge candidate because predictions collapsed to a constant "
            f"(val={val_pred_nunique}, test={test_pred_nunique})."
        )

    quality_gate = _zero_return_quality_gate(
        val_df[cfg.target_col].to_numpy(dtype=float),
        np.asarray(val_pred, dtype=float),
        test_df[cfg.target_col].to_numpy(dtype=float),
        np.asarray(test_pred, dtype=float),
    )
    promotion_eligible = bool(quality_gate["promotion_eligible"])

    run_ts = time.time_ns()
    versioned_name = f"{cfg.model_name}_{run_ts}"
    artifact: dict[str, Any] = {
        "artifact_contract_version": 2,
        "candidate_only": not (cfg.publish and promotion_eligible),
        "promotion_eligible": promotion_eligible,
        "quality_gate": quality_gate,
        "model": model,
        "feature_cols": feature_cols,
        "target_col": cfg.target_col,
        "timestamp_col": "timestamp",
        "return_col": cfg.return_col,
        "resolved_return_col": resolved_return,
        "regime": cfg.regime.strip().lower(),
        "model_name": versioned_name,
        "training_regime": cfg.regime.strip().lower(),
        "regime_filter_applied": True,
        "target_alignment": "current_features_to_next_period_target"
        if cfg.target_shift == -1
        else "configured_target_shift",
        "runtime_versions": _runtime_versions(),
    }
    metadata: dict[str, Any] = {
        "artifact_contract_version": 2,
        "model_type": "ridge",
        "candidate_only": not (cfg.publish and promotion_eligible),
        "publish_requested": cfg.publish,
        "promotion_eligible": promotion_eligible,
        "model_name": versioned_name,
        "features_path": str(cfg.features_path),
        "regimes_path": str(cfg.regimes_path),
        "regime": cfg.regime.strip().lower(),
        "training_regime": cfg.regime.strip().lower(),
        "regime_filter_applied": True,
        "target_col": cfg.target_col,
        "target_shift": cfg.target_shift,
        "target_alignment": artifact["target_alignment"],
        "resolved_return_col": resolved_return,
        "feature_cols": feature_cols,
        "n_rows_used": int(len(expert_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_summary": _target_summary(expert_df[cfg.target_col]),
        "val_metrics": _metrics(val_df[cfg.target_col].to_numpy(), val_pred),
        "test_metrics": _metrics(test_df[cfg.target_col].to_numpy(), test_pred),
        "quality_gate": quality_gate,
        "val_pred_nunique": val_pred_nunique,
        "test_pred_nunique": test_pred_nunique,
        "params": {"ridge_alpha": cfg.ridge_alpha},
        "runtime_versions": _runtime_versions(),
        "created_at_unix_ns": run_ts,
    }

    candidate_path = cfg.output_dir / f"{versioned_name}.joblib"
    metadata_path = cfg.output_dir / f"{versioned_name}.json"
    _atomic_joblib_dump(artifact, candidate_path)
    _atomic_write_json(metadata, metadata_path)

    if cfg.publish and promotion_eligible:
        assert cfg.publish_dir is not None  # narrowed above; keeps the destination explicit
        publish_name = cfg.publish_name or cfg.model_name
        _atomic_joblib_dump(artifact, cfg.publish_dir / f"{publish_name}.joblib")
        _atomic_write_json(metadata, cfg.publish_dir / f"{publish_name}.metadata.json")

    print("Wrote validated pretrained candidate:", candidate_path)
    if cfg.publish and promotion_eligible:
        print("Published pretrained artifact under:", cfg.publish_dir)
    if cfg.publish and not promotion_eligible:
        raise ValueError(
            "Ridge candidate failed the zero-return test gate; no publish destination was changed. "
            f"model_rmse={quality_gate['test_rmse']:.8f} "
            f"zero_return_rmse={quality_gate['zero_return_test_rmse']:.8f}"
        )
    return candidate_path


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

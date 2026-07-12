from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainHMMConfig:
    features_path: str
    obs_mode: str
    n_states: int
    covariance_type: str
    seed: int
    min_state_fraction: float

    time_col: str
    output_dir: str
    run_name: str


def _parse_args() -> TrainHMMConfig:
    p = argparse.ArgumentParser(description="Train a Gaussian HMM regime detector.")

    p.add_argument("--features-path", required=True, help="Path to features data, parquet or csv.")
    p.add_argument(
        "--obs-mode",
        default="rich",
        choices=["minimal", "rich"],
        help="Observation mode. minimal=returns+spread, rich=returns+spread+trend (close/sma_10).",
    )
    p.add_argument("--n-states", type=int, default=3, help="Number of hidden states.")
    p.add_argument(
        "--covariance-type",
        default="full",
        choices=["full", "diag", "tied", "spherical"],
        help="Gaussian covariance type.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-state-fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum fraction of valid training rows assigned to every hidden state. "
            "Rejects collapsed regime models."
        ),
    )

    p.add_argument("--time-col", default="timestamp", help="Time column for ordering.")
    p.add_argument("--output-dir", default="models/regimes/hmm", help="Local output folder.")
    p.add_argument("--run-name", default="", help="Optional run name. Defaults to timestamp.")

    args = p.parse_args()

    if not 0.0 < float(args.min_state_fraction) < 1.0:
        raise ValueError("--min-state-fraction must be between 0 and 1")

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"hmm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    return TrainHMMConfig(
        features_path=args.features_path,
        obs_mode=args.obs_mode,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        seed=args.seed,
        min_state_fraction=float(args.min_state_fraction),
        time_col=args.time_col,
        output_dir=args.output_dir,
        run_name=run_name,
    )


def _read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"features path not found: {path}")

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type: {p.suffix}, use parquet or csv")


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {sorted(df.columns)}")


def _build_observations(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (obs_df, obs_cols). obs_df is numeric-only columns used by the HMM.
    """
    if mode == "minimal":
        needed = ["log_return_1_x", "log_return_1_y"]
        _require_columns(df, needed)

        obs = pd.DataFrame(index=df.index)
        obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
        obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
        obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

        cols = ["ret_x", "ret_y", "spread_ret"]
        return obs[cols], cols

    if mode == "rich":
        needed = ["log_return_1_x", "log_return_1_y", "close_x", "sma_10_x", "close_y", "sma_10_y"]
        _require_columns(df, needed)

        obs = pd.DataFrame(index=df.index)
        obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
        obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
        obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

        close_x = pd.to_numeric(df["close_x"], errors="coerce")
        sma_x = pd.to_numeric(df["sma_10_x"], errors="coerce")
        close_y = pd.to_numeric(df["close_y"], errors="coerce")
        sma_y = pd.to_numeric(df["sma_10_y"], errors="coerce")

        obs["trend_x"] = (close_x / sma_x) - 1.0
        obs["trend_y"] = (close_y / sma_y) - 1.0

        cols = ["ret_x", "ret_y", "spread_ret", "trend_x", "trend_y"]
        return obs[cols], cols

    raise ValueError(f"Unsupported obs_mode: {mode}")


def _initial_hmm_means(Xz: np.ndarray, n_states: int) -> np.ndarray:
    """Deterministic spread-out HMM means without sklearn KMeans initialization."""
    if len(Xz) < n_states:
        raise ValueError(f"Need at least n_states rows for HMM initialization, got {len(Xz)}")

    # The first standardized observation is the primary market return.  Its
    # quantiles give stable, diverse starting states without invoking KMeans,
    # which is fragile in constrained Windows/BLAS environments.
    order = np.argsort(Xz[:, 0], kind="mergesort")
    positions = np.linspace(0, len(order) - 1, num=n_states, dtype=int)
    return np.asarray(Xz[order[positions]], dtype=np.float64)


def _initial_hmm_covariances(Xz: np.ndarray, cfg: TrainHMMConfig) -> np.ndarray:
    n_features = int(Xz.shape[1])
    full_cov = np.cov(Xz, rowvar=False, bias=True)
    full_cov = np.asarray(full_cov, dtype=np.float64).reshape(n_features, n_features)
    full_cov = full_cov + (np.eye(n_features, dtype=np.float64) * 1e-6)

    if cfg.covariance_type == "full":
        return np.repeat(full_cov[np.newaxis, :, :], cfg.n_states, axis=0)
    if cfg.covariance_type == "diag":
        diagonal = np.diag(full_cov)
        return np.repeat(diagonal[np.newaxis, :], cfg.n_states, axis=0)
    if cfg.covariance_type == "tied":
        return full_cov
    if cfg.covariance_type == "spherical":
        variance = float(np.mean(np.diag(full_cov)))
        return np.full(cfg.n_states, variance, dtype=np.float64)
    raise ValueError(f"Unsupported covariance_type: {cfg.covariance_type}")


def _fit_hmm(Xz: np.ndarray, cfg: TrainHMMConfig) -> GaussianHMM:
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=500,
        random_state=cfg.seed,
        # Avoid hmmlearn's default KMeans initialization.  It can fail in the
        # local Windows runtime when threadpoolctl cannot inspect BLAS.
        init_params="st",
    )
    model.means_ = _initial_hmm_means(Xz, cfg.n_states)
    model.covars_ = _initial_hmm_covariances(Xz, cfg)
    model.fit(Xz)
    return model


def _map_states_to_labels(
    hidden_states: np.ndarray,
    obs_df: pd.DataFrame,
) -> dict[int, str]:
    """
    Deterministic mapping:
      compute per-state mean of spread_ret
      lowest mean -> bearish, highest mean -> bullish, middle -> sideways
    """
    tmp = obs_df.copy()
    tmp["_state"] = hidden_states

    means = tmp.groupby("_state", sort=True)["spread_ret"].mean()

    # Order states by spread mean
    ordered_states = list(means.sort_values().index.astype(int))

    # Map to labels
    mapping: dict[int, str] = {}
    if len(ordered_states) == 1:
        mapping[ordered_states[0]] = "sideways"
        return mapping

    if len(ordered_states) == 2:
        mapping[ordered_states[0]] = "bearish"
        mapping[ordered_states[1]] = "bullish"
        return mapping

    # 3 or more: lowest bearish, highest bullish, everything else sideways
    mapping[ordered_states[0]] = "bearish"
    mapping[ordered_states[-1]] = "bullish"
    for s in ordered_states[1:-1]:
        mapping[s] = "sideways"
    return mapping


def main() -> None:
    cfg = _parse_args()

    df = _read_df(cfg.features_path)

    # Time ordering, same philosophy as your LightGBM script
    if cfg.time_col in df.columns:
        dt = _safe_to_datetime(df[cfg.time_col])
        df = df.assign(_dt=dt).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    obs_df, obs_cols = _build_observations(df, cfg.obs_mode)

    # Training rows must have full observations
    train_mask = obs_df.notna().all(axis=1)
    obs_train = obs_df[train_mask].reset_index(drop=True)

    if len(obs_train) < 50:
        raise ValueError(
            f"Not enough rows after dropping NaNs for HMM training, rows={len(obs_train)}"
        )

    scaler = StandardScaler()
    X = obs_train.to_numpy(dtype=np.float64)
    Xz = scaler.fit_transform(X)

    model = _fit_hmm(Xz, cfg)

    hidden_states = model.predict(Xz)
    state_counts = np.bincount(hidden_states.astype(int), minlength=cfg.n_states)
    state_fractions = state_counts / float(len(hidden_states))
    missing_states = [int(state) for state, count in enumerate(state_counts) if int(count) == 0]
    undersized_states = [
        int(state)
        for state, fraction in enumerate(state_fractions)
        if float(fraction) < cfg.min_state_fraction
    ]
    if missing_states or undersized_states:
        raise ValueError(
            "Refusing to save a collapsed HMM regime model. "
            f"state_counts={state_counts.tolist()} min_state_fraction={cfg.min_state_fraction} "
            f"missing_states={missing_states} undersized_states={undersized_states}"
        )
    state_mapping = _map_states_to_labels(hidden_states, obs_train)

    # Training diagnostics for metadata
    tmp = obs_train.copy()
    tmp["_state"] = hidden_states
    per_state = (
        tmp.groupby("_state", sort=True)
        .agg(
            n=("spread_ret", "size"),
            mean_ret_x=("ret_x", "mean"),
            mean_ret_y=("ret_y", "mean"),
            mean_spread=("spread_ret", "mean"),
            std_spread=("spread_ret", "std"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(scaler, run_dir / "scaler.joblib")

    (run_dir / "state_mapping.json").write_text(
        json.dumps({str(k): v for k, v in state_mapping.items()}, indent=2),
        encoding="utf-8",
    )

    metadata: dict[str, Any] = {
        "run_name": cfg.run_name,
        "trained_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features_path": cfg.features_path,
        "obs_mode": cfg.obs_mode,
        "obs_cols": obs_cols,
        "n_states": cfg.n_states,
        "covariance_type": cfg.covariance_type,
        "seed": cfg.seed,
        "min_state_fraction": cfg.min_state_fraction,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(obs_train)),
        "dropped_rows_due_to_nans": int(len(df) - len(obs_train)),
        "per_state_stats": per_state,
        "state_mapping": {str(k): v for k, v in state_mapping.items()},
        "state_counts": {str(index): int(count) for index, count in enumerate(state_counts)},
        "state_fractions": {
            str(index): float(fraction) for index, fraction in enumerate(state_fractions)
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Convenience copy for "latest" (simple, cross-platform)
    latest_dir = out_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, latest_dir / "model.joblib")
    joblib.dump(scaler, latest_dir / "scaler.joblib")
    (latest_dir / "state_mapping.json").write_text(
        json.dumps({str(k): v for k, v in state_mapping.items()}, indent=2),
        encoding="utf-8",
    )
    (latest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote: {run_dir}")
    print(f"Wrote: {latest_dir}")
    print("done")


if __name__ == "__main__":
    main()

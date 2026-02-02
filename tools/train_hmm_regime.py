from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainHMMConfig:
    features_path: str
    obs_mode: str
    n_states: int
    covariance_type: str
    seed: int

    time_col: str
    output_dir: str
    run_name: str


def _parse_args() -> TrainHMMConfig:
    p = argparse.ArgumentParser(description="Train a Gaussian HMM regime detector.")

    p.add_argument("--features-path", required=True, help="Path to features data, parquet or csv.")
    p.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal"],
        help="Observation mode. For now only 'minimal' is supported.",
    )
    p.add_argument("--n-states", type=int, default=3, help="Number of hidden states.")
    p.add_argument(
        "--covariance-type",
        default="full",
        choices=["full", "diag", "tied", "spherical"],
        help="Gaussian covariance type.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--time-col", default="timestamp", help="Time column for ordering.")
    p.add_argument("--output-dir", default="artifacts/regimes_hmm", help="Local output folder.")
    p.add_argument("--run-name", default="", help="Optional run name. Defaults to timestamp.")

    args = p.parse_args()

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"hmm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    return TrainHMMConfig(
        features_path=args.features_path,
        obs_mode=args.obs_mode,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        seed=args.seed,
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
    if mode != "minimal":
        raise ValueError(f"Unsupported obs_mode: {mode}")

    needed = ["log_return_1_x", "log_return_1_y"]
    _require_columns(df, needed)

    obs = pd.DataFrame(index=df.index)
    obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
    obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
    obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

    cols = ["ret_x", "ret_y", "spread_ret"]
    return obs[cols], cols


def _fit_hmm(Xz: np.ndarray, cfg: TrainHMMConfig) -> GaussianHMM:
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=500,
        random_state=cfg.seed,
    )
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
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(obs_train)),
        "dropped_rows_due_to_nans": int(len(df) - len(obs_train)),
        "per_state_stats": per_state,
        "state_mapping": {str(k): v for k, v in state_mapping.items()},
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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMArtifacts:
    model: Any
    scaler: Any
    state_to_label: dict[int, str]
    metadata: dict[str, Any]


def _default_artifacts_dir(cfg: dict[str, Any]) -> Path:
    """
    Resolve where HMM artifacts live.

    Priority:
      1) cfg["regimes"]["hmm"]["artifacts_dir"]
      2) "models/regimes/hmm"
    """
    reg_cfg = cfg.get("regimes", {})
    hmm_cfg = cast(dict[str, Any], reg_cfg.get("hmm", {})) if isinstance(reg_cfg, dict) else {}
    p = hmm_cfg.get("artifacts_dir", "models/regimes/hmm")
    return Path(str(p))


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_artifacts(artifacts_root: Path) -> HMMArtifacts:
    """
    Expects:
      <root>/latest/model.joblib
      <root>/latest/scaler.joblib
      <root>/latest/state_mapping.json
      <root>/latest/metadata.json
    """
    latest = artifacts_root / "latest"

    model_path = latest / "model.joblib"
    scaler_path = latest / "scaler.joblib"
    mapping_path = latest / "state_mapping.json"
    meta_path = latest / "metadata.json"

    missing = [p for p in [model_path, scaler_path, mapping_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HMM artifact files: "
            + ", ".join(str(p.as_posix()) for p in missing)
            + ". Train the HMM first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    raw_mapping = _load_json(mapping_path)
    # keys are strings in JSON, normalize to int -> str
    state_to_label = {int(k): str(v) for k, v in raw_mapping.items()}

    metadata = _load_json(meta_path)

    return HMMArtifacts(
        model=model,
        scaler=scaler,
        state_to_label=state_to_label,
        metadata=metadata,
    )


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {sorted(df.columns)}")


def _build_observations_minimal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Must match the training observation construction.

    Returns:
      obs_df with columns: ret_x, ret_y, spread_ret
    """
    needed = ["log_return_1_x", "log_return_1_y"]
    _require_columns(df, needed)

    obs = pd.DataFrame(index=df.index)
    obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
    obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
    obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

    cols = ["ret_x", "ret_y", "spread_ret"]
    return obs[cols], cols


def label_regimes_hmm(df: pd.DataFrame, *, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    HMM-based regime labeling.

    Contract:
      returns a dataframe with:
        - regime
        - regime_explanation

    NaN policy:
      - if any observation is NaN for a row -> regime="unknown"
    """
    artifacts_root = _default_artifacts_dir(cfg)
    art = _load_artifacts(artifacts_root)

    # Currently only minimal mode is implemented, but we still read it from metadata if present
    obs_mode = str(art.metadata.get("obs_mode", "minimal")).lower()
    if obs_mode != "minimal":
        raise ValueError(f"Unsupported obs_mode in metadata: {obs_mode}")

    obs_df, obs_cols = _build_observations_minimal(df)

    # Enforce we are consistent with training metadata (helps catch accidental schema drift)
    meta_cols = art.metadata.get("obs_cols")
    if isinstance(meta_cols, list) and [str(c) for c in meta_cols] != obs_cols:
        raise ValueError(
            f"Observation columns mismatch. metadata={meta_cols}, runtime={obs_cols}. "
            "Retrain or fix observation builder."
        )

    valid_mask = obs_df.notna().all(axis=1)
    n_valid = int(valid_mask.sum())

    regimes = pd.Series(index=df.index, dtype="string")
    explanations = pd.Series(index=df.index, dtype="string")

    # Default for invalid rows
    regimes.loc[~valid_mask] = "unknown"
    explanations.loc[~valid_mask] = "insufficient data for HMM observations"

    if n_valid == 0:
        return pd.DataFrame({"regime": regimes, "regime_explanation": explanations}, index=df.index)

    X = obs_df.loc[valid_mask].to_numpy(dtype=np.float64)
    Xz = art.scaler.transform(X)

    # Hidden states -> labels
    states = art.model.predict(Xz)

    labels: list[str] = []
    for s in states:
        s_int = int(s)
        labels.append(art.state_to_label.get(s_int, "unknown"))

    regimes.loc[valid_mask] = pd.Series(labels, index=obs_df.index[valid_mask], dtype="string")

    # Explanation: include state id + optional per-state mean spread from metadata
    per_state_stats = art.metadata.get("per_state_stats", [])
    mean_spread_by_state: dict[int, float] = {}
    if isinstance(per_state_stats, list):
        for rec in per_state_stats:
            if not isinstance(rec, dict):
                continue
            try:
                st = int(rec.get("_state"))
                ms = float(rec.get("mean_spread"))
                mean_spread_by_state[st] = ms
            except Exception:
                continue

    expl: list[str] = []
    for s, lab in zip(states, labels):
        s_int = int(s)
        ms = mean_spread_by_state.get(s_int)
        if ms is None or np.isnan(ms):
            expl.append(f"hmm state={s_int}, mapped={lab}")
        else:
            expl.append(f"hmm state={s_int}, mapped={lab}, train_mean_spread={ms:.6g}")

    explanations.loc[valid_mask] = pd.Series(expl, index=obs_df.index[valid_mask], dtype="string")

    out = pd.DataFrame(
        {
            "regime": regimes,
            "regime_explanation": explanations,
        },
        index=df.index,
    )
    return out

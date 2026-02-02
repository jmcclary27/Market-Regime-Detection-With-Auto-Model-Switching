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
    # Priority:
    # 1) cfg["regimes"]["hmm"]["artifacts_dir"]
    # 2) models/regimes/hmm
    reg_cfg = cfg.get("regimes", {})
    hmm_cfg: dict[str, Any] = {}
    if isinstance(reg_cfg, dict):
        hmm_cfg_val = reg_cfg.get("hmm", {})
        if isinstance(hmm_cfg_val, dict):
            hmm_cfg = cast(dict[str, Any], hmm_cfg_val)

    p = hmm_cfg.get("artifacts_dir", "models/regimes/hmm")
    return Path(str(p))


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_artifacts(root: Path) -> HMMArtifacts:
    latest = root / "latest"

    model_path = latest / "model.joblib"
    scaler_path = latest / "scaler.joblib"
    mapping_path = latest / "state_mapping.json"
    meta_path = latest / "metadata.json"

    missing = [p for p in [model_path, scaler_path, mapping_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HMM artifacts: "
            + ", ".join(str(p.as_posix()) for p in missing)
            + ". Did you run tools/train_hmm_regime.py?"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    raw_mapping = _load_json(mapping_path)
    state_to_label = {int(k): str(v) for k, v in raw_mapping.items()}

    metadata = _load_json(meta_path)

    return HMMArtifacts(model=model, scaler=scaler, state_to_label=state_to_label, metadata=metadata)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {sorted(df.columns)}")


def _build_observations(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
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


def label_regimes_hmm(df: pd.DataFrame, *, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Returns a dataframe with:
      - regime
      - regime_explanation

    NaN policy:
      - any NaN in observation vector -> regime="unknown"
    """
    artifacts_root = _default_artifacts_dir(cfg)
    art = _load_artifacts(artifacts_root)

    obs_mode = str(art.metadata.get("obs_mode", "minimal")).lower()
    obs_df, obs_cols = _build_observations(df, obs_mode)

    meta_cols = art.metadata.get("obs_cols")
    if isinstance(meta_cols, list) and [str(c) for c in meta_cols] != obs_cols:
        raise ValueError(
            f"Observation columns mismatch. metadata={meta_cols}, runtime={obs_cols}. Retrain HMM."
        )

    valid = obs_df.notna().all(axis=1)

    regimes = pd.Series(index=df.index, dtype="string")
    explanations = pd.Series(index=df.index, dtype="string")

    regimes.loc[~valid] = "unknown"
    explanations.loc[~valid] = "insufficient data for HMM observations"

    if int(valid.sum()) == 0:
        return pd.DataFrame({"regime": regimes, "regime_explanation": explanations}, index=df.index)

    X = obs_df.loc[valid].to_numpy(dtype=np.float64)
    Xz = art.scaler.transform(X)

    states = art.model.predict(Xz)
    labels: list[str] = [art.state_to_label.get(int(s), "unknown") for s in states]
    regimes.loc[valid] = pd.Series(labels, index=obs_df.index[valid], dtype="string")

    # Optional explanation enrichment using training metadata
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

    explanations.loc[valid] = pd.Series(expl, index=obs_df.index[valid], dtype="string")

    return pd.DataFrame({"regime": regimes, "regime_explanation": explanations}, index=df.index)

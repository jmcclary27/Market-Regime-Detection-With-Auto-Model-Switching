# src/regimes/hmm.py
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
    Priority:
      1) cfg["regimes"]["hmm"]["artifacts_dir"]
      2) models/regimes/hmm
    """
    reg_cfg = cfg.get("regimes")
    hmm_cfg: dict[str, Any] = {}

    if isinstance(reg_cfg, dict):
        maybe_hmm = reg_cfg.get("hmm")
        if isinstance(maybe_hmm, dict):
            hmm_cfg = cast(dict[str, Any], maybe_hmm)

    p = hmm_cfg.get("artifacts_dir", "models/regimes/hmm")
    return Path(str(p))


def _load_json_dict(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object (dict) in {path}, got {type(obj)}")
    return cast(dict[str, Any], obj)


def _load_artifacts(root: Path) -> HMMArtifacts:
    latest = root / "latest"

    model_path = latest / "model.joblib"
    scaler_path = latest / "scaler.joblib"
    mapping_path = latest / "state_mapping.json"
    meta_path = latest / "metadata.json"

    missing = [p for p in (model_path, scaler_path, mapping_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HMM artifacts: "
            + ", ".join(p.as_posix() for p in missing)
            + ". Did you run tools/train_hmm_regime.py --output-dir models/regimes/hmm ?"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    raw_mapping = _load_json_dict(mapping_path)
    state_to_label: dict[int, str] = {}
    for k, v in raw_mapping.items():
        # keys might be strings in JSON, values should be labels
        try:
            state_to_label[int(k)] = str(v)
        except Exception:
            continue

    metadata = _load_json_dict(meta_path)

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


def _build_observations(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    mode2 = mode.lower().strip()

    if mode2 == "minimal":
        needed = ["log_return_1_x", "log_return_1_y"]
        _require_columns(df, needed)

        obs = pd.DataFrame(index=df.index)
        obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
        obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
        obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

        cols = ["ret_x", "ret_y", "spread_ret"]
        return obs[cols], cols

    if mode2 == "rich":
        needed = [
            "log_return_1_x",
            "log_return_1_y",
            "close_x",
            "sma_10_x",
            "close_y",
            "sma_10_y",
        ]
        _require_columns(df, needed)

        obs = pd.DataFrame(index=df.index)
        obs["ret_x"] = pd.to_numeric(df["log_return_1_x"], errors="coerce")
        obs["ret_y"] = pd.to_numeric(df["log_return_1_y"], errors="coerce")
        obs["spread_ret"] = obs["ret_x"] - obs["ret_y"]

        close_x = pd.to_numeric(df["close_x"], errors="coerce")
        sma_x = pd.to_numeric(df["sma_10_x"], errors="coerce")
        close_y = pd.to_numeric(df["close_y"], errors="coerce")
        sma_y = pd.to_numeric(df["sma_10_y"], errors="coerce")

        # trend = (close / sma) - 1, will become NaN if sma is NaN or 0
        obs["trend_x"] = (close_x / sma_x) - 1.0
        obs["trend_y"] = (close_y / sma_y) - 1.0

        cols = ["ret_x", "ret_y", "spread_ret", "trend_x", "trend_y"]
        return obs[cols], cols

    raise ValueError(f"Unsupported obs_mode: {mode}")


def _mean_spread_map_from_metadata(metadata: dict[str, Any]) -> dict[int, float]:
    """
    metadata may include:
      per_state_stats: [{ "_state": 0, "mean_spread": -0.001, ...}, ...]
    Return: {state_id: mean_spread}
    """
    out: dict[int, float] = {}
    recs = metadata.get("per_state_stats")

    if not isinstance(recs, list):
        return out

    for rec in recs:
        if not isinstance(rec, dict):
            continue

        raw_state = rec.get("_state")
        raw_mean = rec.get("mean_spread")

        if raw_state is None or raw_mean is None:
            continue

        try:
            st = int(raw_state)
            ms = float(raw_mean)
        except Exception:
            continue

        if np.isfinite(ms):
            out[st] = ms

    return out


def label_regimes_hmm(df: pd.DataFrame, *, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Returns a dataframe with:
      - regime
      - regime_explanation

    NaN policy:
      - any NaN in the observation vector => regime="unknown"
    """
    artifacts_root = _default_artifacts_dir(cfg)
    art = _load_artifacts(artifacts_root)

    obs_mode = str(art.metadata.get("obs_mode", "minimal")).lower()
    obs_df, obs_cols = _build_observations(df, obs_mode)

    # Validate obs_cols match training metadata if present
    meta_cols = art.metadata.get("obs_cols")
    if isinstance(meta_cols, list):
        meta_cols_norm = [str(c) for c in meta_cols]
        if meta_cols_norm != obs_cols:
            raise ValueError(
                f"Observation columns mismatch. metadata={meta_cols_norm}, runtime={obs_cols}. "
                "Retrain HMM or use matching --obs-mode."
            )

    valid = obs_df.notna().all(axis=1)

    regimes = pd.Series(index=df.index, dtype="string")
    explanations = pd.Series(index=df.index, dtype="string")

    regimes.loc[~valid] = "unknown"
    explanations.loc[~valid] = "insufficient data for HMM observations"

    n_valid = int(valid.sum())
    if n_valid == 0:
        return pd.DataFrame(
            {"regime": regimes, "regime_explanation": explanations},
            index=df.index,
        )

    X = obs_df.loc[valid].to_numpy(dtype=np.float64)

    # scaler/model are Any, but runtime should work (mypy-safe)
    Xz = art.scaler.transform(X)
    states = art.model.predict(Xz)

    labels: list[str] = [art.state_to_label.get(int(s), "unknown") for s in states]
    regimes.loc[valid] = pd.Series(labels, index=obs_df.index[valid], dtype="string")

    mean_spread_by_state = _mean_spread_map_from_metadata(art.metadata)

    expl: list[str] = []
    for s, lab in zip(states, labels, strict=False):
        s_int = int(s)
        ms: float | None = mean_spread_by_state.get(s_int)

        if ms is None:
            expl.append(f"hmm state={s_int}, mapped={lab}")
        else:
            expl.append(f"hmm state={s_int}, mapped={lab}, train_mean_spread={ms:.6g}")

    explanations.loc[valid] = pd.Series(expl, index=obs_df.index[valid], dtype="string")

    return pd.DataFrame({"regime": regimes, "regime_explanation": explanations}, index=df.index)

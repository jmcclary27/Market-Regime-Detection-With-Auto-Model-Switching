# src/regimes/hmm.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from src.regimes.diagnostics import (
    RegimeConfidenceStats,
    RegimeDiagnostics,
    compute_durations,
    compute_entropy,
    compute_pct_time,
    compute_switches,
    compute_transition_counts,
    confidence_stats,
    normalize_rows,
)


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


def compute_hmm_diagnostics(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    run_ts: str,
) -> RegimeDiagnostics:
    """
    Compute run-level HMM regime diagnostics.

    Notes:
      - Uses only rows with valid observations (same policy as label_regimes_hmm).
      - Uses HMM hidden state sequence for transitions and durations.
      - If model supports predict_proba, records confidence stats as max posterior per step.
    """
    artifacts_root = _default_artifacts_dir(cfg)
    art = _load_artifacts(artifacts_root)

    obs_mode = str(art.metadata.get("obs_mode", "minimal")).lower()
    obs_df, obs_cols = _build_observations(df, obs_mode)

    meta_cols = art.metadata.get("obs_cols")
    if isinstance(meta_cols, list):
        meta_cols_norm = [str(c) for c in meta_cols]
        if meta_cols_norm != obs_cols:
            raise ValueError(
                f"Observation columns mismatch. metadata={meta_cols_norm}, runtime={obs_cols}. "
                "Retrain HMM or use matching --obs-mode."
            )

    valid = obs_df.notna().all(axis=1)
    n_steps = int(valid.sum())

    # Determine number of regimes (K)
    # Prefer metadata if present, else infer from mapping, else fallback to model attribute.
    n_regimes = None
    meta_k = art.metadata.get("n_states")
    if meta_k is not None:
        try:
            n_regimes = int(meta_k)
        except Exception:
            n_regimes = None

    if n_regimes is None and art.state_to_label:
        n_regimes = int(max(art.state_to_label.keys())) + 1

    if n_regimes is None:
        k_attr = getattr(art.model, "n_components", None)
        if k_attr is not None:
            try:
                n_regimes = int(k_attr)
            except Exception:
                n_regimes = None

    if n_regimes is None:
        raise ValueError("Could not determine n_regimes for HMM diagnostics")

    if n_steps == 0:
        # Empty diagnostics, stable schema
        return RegimeDiagnostics(
            run_ts=run_ts,
            n_steps=0,
            n_regimes=n_regimes,
            n_switches=0,
            switches_per_1000_steps=0.0,
            pct_time_regime=[0.0 for _ in range(n_regimes)],
            avg_regime_duration=float("nan"),
            regime_durations={},
            transition_counts=[[0 for _ in range(n_regimes)] for _ in range(n_regimes)],
            transition_probs=[[0.0 for _ in range(n_regimes)] for _ in range(n_regimes)],
            regime_entropy=float("nan"),
            confidence=None,
            model_version=str(art.metadata.get("version"))
            if art.metadata.get("version") is not None
            else None,
        )

    X = obs_df.loc[valid].to_numpy(dtype=np.float64)
    Xz = art.scaler.transform(X)
    states = art.model.predict(Xz).astype(int)

    n_switches = compute_switches(states)
    switches_per_1000 = (float(n_switches) / float(max(n_steps, 1))) * 1000.0

    pct = compute_pct_time(states, n_regimes=n_regimes)
    entropy = compute_entropy(np.array(pct, dtype=float))

    durations = compute_durations(states)
    all_durs: list[int] = [d for arr in durations.values() for d in arr]
    avg_dur = float(np.mean(all_durs)) if all_durs else float("nan")

    t_counts = compute_transition_counts(states, n_regimes=n_regimes)
    t_probs = normalize_rows(t_counts)

    # Confidence stats if available
    conf_out: RegimeConfidenceStats | None = None
    predict_proba = getattr(art.model, "predict_proba", None)
    if callable(predict_proba):
        try:
            post = predict_proba(Xz)
            post = np.asarray(post, dtype=float)
            if post.ndim == 2 and post.shape[0] == states.shape[0]:
                conf = np.max(post, axis=1)
                conf_out = confidence_stats(conf)
        except Exception:
            conf_out = None

    return RegimeDiagnostics(
        run_ts=run_ts,
        n_steps=n_steps,
        n_regimes=n_regimes,
        n_switches=n_switches,
        switches_per_1000_steps=float(switches_per_1000),
        pct_time_regime=pct,
        avg_regime_duration=float(avg_dur),
        regime_durations=durations,
        transition_counts=t_counts.tolist(),
        transition_probs=t_probs.tolist(),
        regime_entropy=float(entropy),
        confidence=conf_out,
        model_version=str(art.metadata.get("version"))
        if art.metadata.get("version") is not None
        else None,
    )

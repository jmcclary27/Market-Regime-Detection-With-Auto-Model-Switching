# src/backtest/adapters.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalAdapterResult:
    signals: pd.DataFrame
    chosen_model_name: str


def signals_from_predictions_long(
    preds: pd.DataFrame,
    *,
    features: pd.DataFrame,
    signal_col: str = "y_pred",
    asset: str = "portfolio",
) -> SignalAdapterResult:
    """
    Align active predictions to a timestamp index using the features table.

    - features must include a 'timestamp' column
    - preds must include: row_id, is_active, model_name, y_pred
    - row_id is interpreted as the integer row number into features (positional)
    - returns a DataFrame indexed by timestamp with a single column (default 'portfolio')
    """
    required_preds = {"row_id", "is_active", "model_name", signal_col}
    missing = required_preds - set(preds.columns)
    if missing:
        raise ValueError(f"preds missing columns: {sorted(missing)}")

    if "timestamp" not in features.columns:
        raise ValueError("features missing required column: timestamp")

    ts = pd.Series(features["timestamp"].values)
    if ts.isna().any():
        raise ValueError("features.timestamp contains NaNs")

    active = preds.loc[preds["is_active"] == True].copy()  # noqa: E712
    if active.empty:
        raise ValueError("No active predictions found (is_active==True)")

    if active["row_id"].duplicated().any():
        counts = active["row_id"].value_counts()
        bad = counts[counts > 1].head(10).to_dict()
        raise ValueError(f"Multiple active predictions for same row_id. Examples: {bad}")

    model_names = active["model_name"].dropna().unique()
    chosen = str(model_names[0]) if len(model_names) else "unknown"
    if len(model_names) > 1:
        chosen = "mixed_active_models"

    row_ids = active["row_id"].astype(int).to_numpy()
    if row_ids.min() < 0 or row_ids.max() >= len(features):
        raise ValueError(
            f"row_id out of bounds. row_id range [{row_ids.min()}, {row_ids.max()}], "
            f"features len={len(features)}"
        )

    # Build timestamp-indexed signal series
    sig = pd.Series(0.0, index=pd.Index(features["timestamp"], name="timestamp"))
    # Map each row_id to its timestamp position and fill
    sig.iloc[row_ids] = active.set_index("row_id")[signal_col].astype(float).values

    signals = pd.DataFrame({asset: sig})
    return SignalAdapterResult(signals=signals, chosen_model_name=chosen)


def signals_spy_from_predictions(
    preds: pd.DataFrame,
    *,
    features: pd.DataFrame,
    fallback_model_name: str | None = "baseline",
) -> pd.DataFrame:
    """
    Build SPY signals aligned to features.timestamp.

    Selection rule:
      1) If any rows have is_active==True, use those.
      2) Else if fallback_model_name is set and exists, use that model.
      3) Else choose the first model_name in sorted order (deterministic).
    """
    if "timestamp" not in features.columns:
        raise ValueError("features missing required column: timestamp")

    ts_index = pd.Index(features["timestamp"], name="timestamp")

    required = {"row_id", "model_name", "y_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"preds missing columns: {sorted(missing)}")

    # Prefer active predictions if present
    use = None
    if "is_active" in preds.columns:
        active = preds.loc[preds["is_active"] == True].copy()  # noqa: E712
        if not active.empty:
            use = active

    # Otherwise fall back to a deterministic model choice
    if use is None:
        if fallback_model_name is not None and (preds["model_name"] == fallback_model_name).any():
            use = preds.loc[preds["model_name"] == fallback_model_name].copy()
        else:
            # stable deterministic fallback: first model name alphabetically
            chosen = sorted(preds["model_name"].dropna().unique().tolist())
            if not chosen:
                raise ValueError("No predictions found (model_name empty)")
            use = preds.loc[preds["model_name"] == chosen[0]].copy()

    # Enforce one prediction per row_id
    if use["row_id"].duplicated().any():
        counts = use["row_id"].value_counts()
        bad = counts[counts > 1].head(10).to_dict()
        raise ValueError(f"Multiple predictions for same row_id in chosen set. Examples: {bad}")

    row_ids = use["row_id"].astype(int).to_numpy()
    if row_ids.min() < 0 or row_ids.max() >= len(features):
        raise ValueError(
            f"row_id out of bounds. row_id range [{row_ids.min()}, {row_ids.max()}], "
            f"features len={len(features)}"
        )

    use = use.sort_values("row_id")

    sig = pd.Series(0.0, index=ts_index, dtype="float64")
    sig.iloc[use["row_id"].astype(int).to_numpy()] = use["y_pred"].astype(float).to_numpy()

    return pd.DataFrame({"SPY": sig}, index=ts_index)

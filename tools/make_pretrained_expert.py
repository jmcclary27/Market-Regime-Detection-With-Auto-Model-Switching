# scripts/train_expert_bullish.py (or wherever you keep this)
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

FEATURES_PATH = Path("data/features/latest.parquet")
REGIMES_PATH = Path("data/regimes/latest.parquet")

TIMESTAMP_COL = "timestamp"
RETURN_COL = "log_return_1"
TARGET_COL = "target_next_return"


def _resolve_col(df: pd.DataFrame, base: str) -> str:
    """
    Resolve base / base_x / base_y, and also handle "double suffix" cases that can happen
    when you merge a wide features frame with another frame that accidentally shares names.

    Priority order:
      1) base
      2) base_x
      3) base_y
      4) any column that starts with base + "_" (pick the first in sorted order)
    """
    if base in df.columns:
        return base
    if f"{base}_x" in df.columns:
        return f"{base}_x"
    if f"{base}_y" in df.columns:
        return f"{base}_y"

    # Fallback for double-suffix or unexpected collisions:
    candidates = sorted([c for c in df.columns if c.startswith(f"{base}_")])
    if candidates:
        return candidates[0]

    raise KeyError(f"Expected column '{base}' (or suffixed) not found. cols={list(df.columns)}")


def _resolve_pair_feature_cols(feats: pd.DataFrame) -> list[str]:
    """
    Lock the expert to the canonical 6-feature contract:
      close_x, log_return_1_x, sma_10_x, close_y, log_return_1_y, sma_10_y

    If the data is long-form (no _x/_y), this will fall back to:
      close, log_return_1, sma_10
    """
    base_cols = ["close", "log_return_1", "sma_10"]

    # Wide (preferred): require all x/y
    wide = [f"{c}_x" for c in base_cols] + [f"{c}_y" for c in base_cols]
    if all(c in feats.columns for c in wide):
        return wide

    # Long fallback
    if all(c in feats.columns for c in base_cols):
        return base_cols

    raise KeyError(
        "Features do not match expected schema. Need either "
        f"{wide} (wide) or {base_cols} (long). cols={list(feats.columns)}"
    )


def main() -> None:
    feats = pd.read_parquet(FEATURES_PATH)
    regs = pd.read_parquet(REGIMES_PATH)

    # Consistently resolve return col from features schema (prefer _x over _y)
    resolved_return = _resolve_col(feats, RETURN_COL)
    if resolved_return != RETURN_COL:
        print(f"Resolved return col: {RETURN_COL} -> {resolved_return}")

    # Lock to the canonical feature set (prevents drift)
    feature_cols = _resolve_pair_feature_cols(feats)

    # Merge with explicit suffixes to avoid losing/resuffixing feature columns
    df = feats.merge(regs, on=TIMESTAMP_COL, how="inner", suffixes=("_feat", "_reg"))

    # Resolve again on the merged frame in case merge introduced suffix collisions
    resolved_return_df = _resolve_col(df, RETURN_COL)
    if resolved_return_df != resolved_return:
        print(f"Resolved return col after merge: {resolved_return} -> {resolved_return_df}")

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df[TARGET_COL] = df[resolved_return_df].shift(-1)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Expert is trained only on bullish regime rows (example)
    if "regime" not in df.columns:
        # If the merge applied suffixes because regs had a regime column name collision,
        # try a best-effort fallback.
        if "regime_reg" in df.columns:
            df = df.rename(columns={"regime_reg": "regime"})
        elif "regime_feat" in df.columns:
            df = df.rename(columns={"regime_feat": "regime"})
        else:
            raise KeyError(f"'regime' column missing after merge. cols={list(df.columns)}")

    expert_df = df[df["regime"] == "bullish"].copy()
    min_rows = 10  # set to 2-10 for dev smoke tests, raise later for real training
    if len(expert_df) < min_rows:
        print(
            f"WARNING: bullish expert has only {len(expert_df)} rows, min_rows={min_rows}. "
            "Training anyway for a smoke test."
        )
        if len(expert_df) < 2:
            raise ValueError(f"Not enough rows to fit a model at all, got {len(expert_df)}")

    # If merge introduced suffix collisions on feature columns, prefer the feature-side versions.
    # We keep your canonical list but map to existing columns in df.
    mapped_feature_cols: list[str] = []
    for c in feature_cols:
        if c in df.columns:
            mapped_feature_cols.append(c)
            continue
        if f"{c}_feat" in df.columns:
            mapped_feature_cols.append(f"{c}_feat")
            continue
        if f"{c}_reg" in df.columns:
            mapped_feature_cols.append(f"{c}_reg")
            continue
        raise KeyError(
            f"Expected feature column '{c}' not found after merge. cols={list(df.columns)}"
        )

    X = expert_df[mapped_feature_cols].to_numpy()
    y = expert_df[TARGET_COL].to_numpy()

    # Imputer makes it robust to NaNs from rolling features, etc.
    model: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X, y)

    out_dir = Path("models/pretrained")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expert_bullish_ridge_v0.joblib"

    artifact: dict[str, Any] = {
        "model": model,
        "feature_cols": mapped_feature_cols,
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "return_col": RETURN_COL,
        "resolved_return_col": resolved_return_df,
        "regime": "bullish",
        "model_name": "expert_bullish_ridge_v0",
        "notes": "pretrained expert trained on bullish rows only, stored as frozen artifact",
        "n_rows_used": int(len(expert_df)),
    }

    joblib.dump(artifact, out_path)

    print("Wrote pretrained expert:", out_path)
    print("Rows used:", len(expert_df))
    print("Feature cols:", mapped_feature_cols)


if __name__ == "__main__":
    main()

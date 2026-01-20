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
    """Resolve base / base_x / base_y (common after merges)."""
    if base in df.columns:
        return base
    if f"{base}_x" in df.columns:
        return f"{base}_x"
    if f"{base}_y" in df.columns:
        return f"{base}_y"
    raise KeyError(f"Expected column '{base}' (or suffixed) not found. cols={list(df.columns)}")


def main() -> None:
    feats = pd.read_parquet(FEATURES_PATH)
    regs = pd.read_parquet(REGIMES_PATH)

    df = feats.merge(regs, on=TIMESTAMP_COL, how="inner")

    # Normalize symbol if duplicated
    if "symbol_x" in df.columns and "symbol_y" in df.columns:
        df = df.drop(columns=["symbol_y"]).rename(columns={"symbol_x": "symbol"})
    elif "symbol_y" in df.columns and "symbol_x" not in df.columns:
        df = df.drop(columns=["symbol_y"])

    # Resolve return col (handles log_return_1_x / log_return_1_y)
    resolved_return = _resolve_col(df, RETURN_COL)
    if resolved_return != RETURN_COL:
        print(f"Resolved return col: {RETURN_COL} -> {resolved_return}")

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df[TARGET_COL] = df[resolved_return].shift(-1)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Expert is trained only on bullish regime rows (example)
    if "regime" not in df.columns:
        raise KeyError(f"'regime' column missing after merge. cols={list(df.columns)}")

    expert_df = df[df["regime"] == "bullish"].copy()
    min_rows = 10  # set to 2-10 for dev smoke tests, raise later for real training
    if len(expert_df) < min_rows:
        print(f"WARNING: bullish expert has only {len(expert_df)} rows, min_rows={min_rows}. "
            "Training anyway for a smoke test.")
        if len(expert_df) < 2:
            raise ValueError(f"Not enough rows to fit a model at all, got {len(expert_df)}")

    exclude = {
        TARGET_COL,
        TIMESTAMP_COL,
        "regime_explanation",
        "regime",
        "symbol",
        "symbol_x",
        "symbol_y",
    }
    feature_cols = [
        c for c in expert_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(expert_df[c])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns for expert. cols={list(expert_df.columns)}")

    X = expert_df[feature_cols].to_numpy()
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
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "return_col": RETURN_COL,
        "resolved_return_col": resolved_return,
        "regime": "bullish",
        "model_name": "expert_bullish_ridge_v0",
        "notes": "pretrained expert trained on bullish rows only, stored as frozen artifact",
        "n_rows_used": int(len(expert_df)),
    }

    joblib.dump(artifact, out_path)

    print("Wrote pretrained expert:", out_path)
    print("Rows used:", len(expert_df))
    print("Feature cols:", feature_cols)


if __name__ == "__main__":
    main()

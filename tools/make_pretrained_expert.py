from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

FEATURES_PATH = Path("data/features/test-run.parquet")
REGIMES_PATH = Path("data/regimes/test-run.parquet")

TIMESTAMP_COL = "timestamp"
RETURN_COL = "log_return_1"
TARGET_COL = "target_next_return"


def main() -> None:
    feats = pd.read_parquet(FEATURES_PATH)
    regs = pd.read_parquet(REGIMES_PATH)

    df = feats.merge(regs, on=TIMESTAMP_COL, how="inner")
    if "symbol_x" in df.columns and "symbol_y" in df.columns:
        df = df.drop(columns=["symbol_y"]).rename(columns={"symbol_x": "symbol"})

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df[TARGET_COL] = df[RETURN_COL].shift(-1)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Expert is trained only on bullish regime rows (example)
    expert_df = df[df["regime"] == "bullish"].copy()
    if len(expert_df) < 50:
        raise ValueError(f"Not enough rows for bullish expert, got {len(expert_df)}")

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
        c
        for c in expert_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(expert_df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns for expert")

    X = expert_df[feature_cols].to_numpy()
    y = expert_df[TARGET_COL].to_numpy()

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    out_dir = Path("models/pretrained")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expert_bullish_ridge_v0.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": TARGET_COL,
            "timestamp_col": TIMESTAMP_COL,
            "regime": "bullish",
            "model_name": "expert_bullish_ridge_v0",
            "notes": "placeholder pretrained expert, created once, loaded as frozen artifact",
        },
        out_path,
    )

    print("Wrote pretrained expert:", out_path)
    print("Feature cols:", feature_cols)
    print("Rows used:", len(expert_df))


if __name__ == "__main__":
    main()

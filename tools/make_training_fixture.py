from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    np.random.seed(42)

    n = 800  # enough for 70/15/15 split
    start = pd.Timestamp("2024-01-01 09:30:00")

    # 1-min timestamps, market-style
    ts = pd.date_range(start=start, periods=n, freq="T")

    # Simple random walk price
    rets = np.random.normal(loc=0.0, scale=0.001, size=n)  # small-ish returns
    close = 100.0 * np.exp(np.cumsum(rets))

    df_feat = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "TEST",
            "close": close,
        }
    )

    # Features similar to what you already have
    df_feat["log_return_1"] = np.log(df_feat["close"]).diff()
    df_feat["sma_10"] = df_feat["close"].rolling(10).mean()

    # Drop early NaNs from rolling/diff
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Create regimes based on volatility/trend-ish heuristics
    vol = df_feat["log_return_1"].rolling(20).std().fillna(method="bfill")
    trend = df_feat["close"].diff(20).fillna(0.0)

    regime = np.where(vol > vol.quantile(0.8), "high_vol", "low_vol")
    regime = np.where(trend > 0, "bullish", regime)
    regime = np.where(trend < 0, "bearish", regime)

    df_reg = pd.DataFrame(
        {
            "timestamp": df_feat["timestamp"],
            "symbol": df_feat["symbol"],
            "regime": regime,
            "regime_explanation": np.where(
                vol > vol.quantile(0.8),
                "rule: vol_high",
                np.where(trend > 0, "rule: trend_up", "rule: trend_down_or_flat"),
            ),
        }
    )

    out_feat = Path("data/features/test-run.parquet")
    out_reg = Path("data/regimes/test-run.parquet")
    out_feat.parent.mkdir(parents=True, exist_ok=True)
    out_reg.parent.mkdir(parents=True, exist_ok=True)

    df_feat.to_parquet(out_feat, index=False)
    df_reg.to_parquet(out_reg, index=False)

    print("Wrote:", out_feat, "rows:", len(df_feat), "cols:", len(df_feat.columns))
    print("Wrote:", out_reg, "rows:", len(df_reg), "cols:", len(df_reg.columns))
    print("Regime counts:", df_reg["regime"].value_counts().to_dict())


if __name__ == "__main__":
    main()

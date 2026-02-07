from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.metrics import (
    EvalConfig,
    build_eval_frame,
    build_scorecard,
    compute_metrics_table,
    make_timestamp_id,
    write_scorecard_artifacts,
)
from src.eval.walk_forward import walk_forward_splits


def run() -> None:
    main()


def _extract_market_time(eval_df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Return a tz-aware UTC timestamp Series aligned to eval_df rows.

    Priority:
      1) eval_df["timestamp"]
      2) eval_df.index if DatetimeIndex
      3) join via row_id -> features["timestamp"]
    """
    if "timestamp" in eval_df.columns:
        return pd.to_datetime(eval_df["timestamp"], utc=True, errors="raise")

    if isinstance(eval_df.index, pd.DatetimeIndex):
        # Ensure UTC
        idx = eval_df.index
        if idx.tz is None:
            return pd.to_datetime(idx, utc=True, errors="raise")
        return pd.to_datetime(idx, utc=True, errors="raise")

    if "row_id" in eval_df.columns and "timestamp" in features.columns:
        # Build row_id -> timestamp mapping from features row order
        # Assumption, row_id refers to the row position of features (common in your adapters)
        fmap = features[["timestamp"]].copy()
        fmap = fmap.reset_index(drop=True)
        fmap["row_id"] = fmap.index.astype(int)

        merged = eval_df[["row_id"]].merge(fmap, on="row_id", how="left", validate="many_to_one")
        if merged["timestamp"].isna().any():
            missing = int(merged["timestamp"].isna().sum())
            raise ValueError(
                f"Walk-forward requires market time, could not resolve timestamp for {missing} rows via row_id."
            )
        return pd.to_datetime(merged["timestamp"], utc=True, errors="raise")

    raise ValueError(
        "Walk-forward requires market time, but eval_df has no 'timestamp' column, "
        "eval_df index is not DatetimeIndex, and cannot map via row_id. "
        f"eval_df columns={list(eval_df.columns)}; features columns={list(features.columns)}"
    )


def _ensure_time_sorted(eval_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure eval_df is sorted by market time and has a stable positional order for iloc slicing.
    """
    ts = _extract_market_time(eval_df, features)

    out = eval_df.copy()
    out["_market_ts"] = ts
    out = out.sort_values("_market_ts").reset_index(drop=True)
    out = out.drop(columns=["_market_ts"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/latest.parquet")
    ap.add_argument("--regimes", default="data/regimes/latest.parquet")
    ap.add_argument("--predictions", default="data/predictions/latest.parquet")
    ap.add_argument("--target-col", default="log_return_1")
    ap.add_argument("--min-regime-n", type=int, default=30)
    ap.add_argument("--out-dir", default="data/scorecards")

    # ---- walk-forward options ----
    ap.add_argument(
        "--walk-forward",
        action="store_true",
        help="If set, compute per-split metrics on the test window using walk-forward splits.",
    )
    ap.add_argument(
        "--run-ts",
        default=None,
        help="If set, use this id for artifact naming (recommended for pipeline replay).",
    )
    ap.add_argument("--wf-train", type=int, default=252 * 2, help="Train window size in bars.")
    ap.add_argument("--wf-val", type=int, default=252 // 2, help="Validation window size in bars.")
    ap.add_argument("--wf-test", type=int, default=252 // 2, help="Test window size in bars.")
    ap.add_argument("--wf-step", type=int, default=252 // 2, help="Step size in bars.")
    ap.add_argument(
        "--wf-anchored",
        action="store_true",
        help="If set, train window expands (anchored). If not set, train window rolls (fixed size).",
    )

    args, _unknown = ap.parse_known_args()

    features_path = Path(args.features)
    regimes_path = Path(args.regimes)
    preds_path = Path(args.predictions)

    features = pd.read_parquet(features_path)
    regimes = pd.read_parquet(regimes_path)
    preds = pd.read_parquet(preds_path)

    cfg = EvalConfig(target_col=args.target_col, min_regime_n=args.min_regime_n)

    eval_df = build_eval_frame(
        predictions=preds,
        features=features,
        regimes=regimes,
        cfg=cfg,
    )

    # Base, full-period metrics and scorecard (existing behavior)
    metrics_tbl = compute_metrics_table(eval_df, cfg=cfg)
    ts = str(args.run_ts) if args.run_ts else make_timestamp_id()
    scorecard = build_scorecard(
        eval_df,
        cfg=cfg,
        features_path=features_path,
        regimes_path=regimes_path,
        predictions_path=preds_path,
        timestamp=ts,
    )
    json_path, parquet_path = write_scorecard_artifacts(
        scorecard=scorecard,
        metrics_table=metrics_tbl,
        out_dir=args.out_dir,
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {parquet_path}")

    # ---- walk-forward metrics ----
    if args.walk_forward:
        eval_df_sorted = _ensure_time_sorted(eval_df, features)

        # Resolve market time ONCE, even if eval_df has no 'timestamp' column.
        market_ts = pd.to_datetime(_extract_market_time(eval_df_sorted, features), utc=True, errors="raise")

        splits = walk_forward_splits(
            market_ts,
            train_size=int(args.wf_train),
            val_size=int(args.wf_val),
            test_size=int(args.wf_test),
            step_size=int(args.wf_step),
            anchored=bool(args.wf_anchored),
        )

        if len(splits) == 0:
            print("Walk-forward: no splits produced (not enough data for requested windows).")
            return

        wf_rows: list[pd.DataFrame] = []

        for s in splits:
            # Evaluate on TEST window only
            test_df = eval_df_sorted.iloc[s.test.start : s.test.stop].copy()
            if test_df.empty:
                continue

            test_metrics = compute_metrics_table(test_df, cfg=cfg).copy()

            # Split metadata from resolved market_ts, not from a 'timestamp' column
            test_ts = market_ts.iloc[s.test.start : s.test.stop]
            split_start = pd.Timestamp(test_ts.iloc[0])
            split_end = pd.Timestamp(test_ts.iloc[-1])

            test_metrics.insert(0, "split_id", s.split_id)
            test_metrics.insert(1, "split_start", split_start)
            test_metrics.insert(2, "split_end", split_end)
            test_metrics.insert(3, "n_test_rows", int(len(test_df)))

            wf_rows.append(test_metrics)

        if not wf_rows:
            print("Walk-forward: splits existed, but no test windows produced metrics.")
            return

        wf_metrics = pd.concat(wf_rows, axis=0, ignore_index=True)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        wf_parquet = out_dir / f"walk_forward_metrics_{ts}.parquet"
        wf_csv = out_dir / f"walk_forward_metrics_{ts}.csv"

        wf_metrics.to_parquet(wf_parquet, index=False)
        wf_metrics.to_csv(wf_csv, index=False)

        print(f"Wrote: {wf_parquet}")
        print(f"Wrote: {wf_csv}")


if __name__ == "__main__":
    main()

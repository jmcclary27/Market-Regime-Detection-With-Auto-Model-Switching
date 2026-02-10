# src/eval/run_evaluator.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import compute_portfolio_metrics
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
        idx = eval_df.index
        return pd.to_datetime(idx, utc=True, errors="raise")

    if "row_id" in eval_df.columns and "timestamp" in features.columns:
        fmap = features[["timestamp"]].copy().reset_index(drop=True)
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


def _row_id_to_prices_spy(*, features: pd.DataFrame, row_ids: pd.Series) -> pd.DataFrame:
    """
    Build SPY prices indexed by timestamp for the given row_ids.

    Assumes:
      - features has 'timestamp'
      - wide schema uses 'close_x' as SPY proxy (matches your pipeline backtest step)
      - row_id refers to the row position of features after deterministic ordering
    """
    if "timestamp" not in features.columns:
        raise ValueError("features missing 'timestamp'")
    if "close_x" not in features.columns:
        raise ValueError("features missing 'close_x' (SPY proxy)")

    fmap = features[["timestamp", "close_x"]].copy().reset_index(drop=True)
    fmap["row_id"] = fmap.index.astype(int)

    rid = pd.to_numeric(row_ids, errors="coerce").dropna().astype(int)
    sub = fmap[fmap["row_id"].isin(rid)].copy()

    if sub.empty:
        return pd.DataFrame(columns=["SPY"], index=pd.DatetimeIndex([], tz="UTC"))

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True, errors="raise")
    sub = sub.sort_values("timestamp")

    prices = (
        sub.rename(columns={"close_x": "SPY"})
        .set_index("timestamp")[["SPY"]]
        .astype(float)
        .sort_index()
    )
    return prices


def _signals_df_from_eval_test_window(
    *,
    test_df: pd.DataFrame,
    features: pd.DataFrame,
    model_name: str,
    traded_col: str = "SPY",
) -> pd.DataFrame:
    """
    Create a signals DataFrame indexed by timestamp for ONE model on the test window.

    Contract with src.backtest.engine.run_backtest:
      - signals is a DataFrame
      - signals.index exactly equals prices.index
      - signals column overlaps prices column, for v1 we use ["SPY"]

    We use y_pred directly as the target position signal.
    """
    need = {"row_id", "model_name", "y_pred"}
    missing = need - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df missing columns: {sorted(missing)}")

    g = test_df.loc[test_df["model_name"] == model_name, ["row_id", "y_pred"]].copy()
    if g.empty:
        # caller will handle empties
        return pd.DataFrame(columns=[traded_col], index=pd.DatetimeIndex([], tz="UTC"))

    fmap = features[["timestamp"]].copy().reset_index(drop=True)
    fmap["row_id"] = fmap.index.astype(int)

    g = g.merge(fmap, on="row_id", how="left", validate="many_to_one")
    if g["timestamp"].isna().any():
        missing_ts = int(g["timestamp"].isna().sum())
        raise ValueError(f"could not map {missing_ts} row_id values to timestamp for signals")

    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True, errors="raise")
    g = g.sort_values("timestamp")

    sig = pd.Series(g["y_pred"].to_numpy(dtype=float), index=g["timestamp"])
    sig = sig.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.DataFrame({traded_col: sig})


def main(argv: list[str] | None = None) -> None:
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

    args, _unknown = ap.parse_known_args(argv)

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

    # ---- base scorecard artifacts (existing behavior) ----
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
    if not args.walk_forward:
        return

    eval_df_sorted = _ensure_time_sorted(eval_df, features)
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
    wf_port_rows: list[dict[str, object]] = []

    bt_cfg = BacktestConfig(
        initial_cash=float(os.getenv("BACKTEST_INITIAL_CASH", "100000")),
        max_leverage=float(os.getenv("BACKTEST_MAX_LEVERAGE", "1.0")),
        clip_signal=float(os.getenv("BACKTEST_CLIP_SIGNAL", "1.0")),
        fee_bps=float(os.getenv("BACKTEST_FEE_BPS", "0")),
        spread_bps=float(os.getenv("BACKTEST_SPREAD_BPS", "0")),
        slippage_bps=float(os.getenv("BACKTEST_SLIPPAGE_BPS", "0")),
        seed=int(os.getenv("BACKTEST_SEED", "0")),
    )

    for s in splits:
        test_df = eval_df_sorted.iloc[s.test.start : s.test.stop].copy()
        if test_df.empty:
            continue

        test_metrics = compute_metrics_table(test_df, cfg=cfg).copy()

        test_ts = market_ts.iloc[s.test.start : s.test.stop]
        split_start = pd.Timestamp(test_ts.iloc[0])
        split_end = pd.Timestamp(test_ts.iloc[-1])

        test_metrics.insert(0, "split_id", s.split_id)
        test_metrics.insert(1, "split_start", split_start)
        test_metrics.insert(2, "split_end", split_end)
        test_metrics.insert(3, "n_test_rows", int(len(test_df)))

        wf_rows.append(test_metrics)

        # ---- portfolio metrics on TEST window (per model) ----
        row_ids = test_df["row_id"].drop_duplicates()
        prices = _row_id_to_prices_spy(features=features, row_ids=row_ids)
        if prices.empty:
            continue

        model_names = sorted(test_df["model_name"].unique().tolist())
        for model_name in model_names:
            sig_df = _signals_df_from_eval_test_window(
                test_df=test_df,
                features=features,
                model_name=model_name,
                traded_col="SPY",
            )

            # Align to prices index and satisfy backtest contract
            sig_df = sig_df.reindex(prices.index).fillna(0.0)
            if sig_df.empty:
                continue

            res = run_backtest(prices=prices, signals=sig_df, cfg=bt_cfg)

            results_df = pd.DataFrame(
                {
                    "equity": res.equity_curve,
                    "returns_gross": res.returns_gross,
                    "returns_net": res.returns_net,
                },
                index=prices.index,
            )

            pm = compute_portfolio_metrics(
                results_df=results_df,
                trades_df=res.trades,
                periods_per_year=252,
            )

            wf_port_rows.append(
                {
                    "run_ts": ts,
                    "split_id": s.split_id,
                    "split_start": split_start,
                    "split_end": split_end,
                    "n_bars": int(len(prices)),
                    "model_name": model_name,
                    "cagr": pm.cagr,
                    "sharpe": pm.sharpe,
                    "sortino": pm.sortino,
                    "max_drawdown": pm.max_drawdown,  # negative number (your convention)
                    "turnover": pm.turnover,
                    "profit_factor": pm.profit_factor,
                }
            )

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

    # ---- portfolio metrics artifact for promotion (PR14) ----
    wf_port_out_dir = Path("data/walkforward")
    wf_port_out_dir.mkdir(parents=True, exist_ok=True)

    wf_port = pd.DataFrame(wf_port_rows)
    port_parquet = wf_port_out_dir / f"portfolio_metrics_{ts}.parquet"
    wf_port.to_parquet(port_parquet, index=False)
    wf_port.to_parquet(wf_port_out_dir / "latest.parquet", index=False)

    print(f"Wrote: {port_parquet}")


if __name__ == "__main__":
    main()

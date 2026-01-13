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


def run() -> None:
    main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/latest.parquet")
    ap.add_argument("--regimes", default="data/regimes/latest.parquet")
    ap.add_argument("--predictions", default="data/predictions/latest.parquet")
    ap.add_argument("--target-col", default="log_return_1")
    ap.add_argument("--min-regime-n", type=int, default=30)
    ap.add_argument("--out-dir", default="data/scorecards")
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
    metrics_tbl = compute_metrics_table(eval_df, cfg=cfg)

    ts = make_timestamp_id()
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


if __name__ == "__main__":
    main()

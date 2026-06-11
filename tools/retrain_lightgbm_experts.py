from __future__ import annotations

import argparse
from datetime import datetime

try:
    from tools.train_lightgbm_expert import TrainConfig, run
except ModuleNotFoundError:
    from train_lightgbm_expert import TrainConfig, run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain all LightGBM regime experts from the regime-labeled parquet."
    )
    p.add_argument(
        "--features-path",
        default="data/regimes/latest.parquet",
        help="Regime-labeled parquet used as the training source of truth.",
    )
    p.add_argument(
        "--output-dir",
        default="models/experts",
        help="Root directory for expert artifacts.",
    )
    p.add_argument("--experiment-name", default="market-regime")
    p.add_argument("--model-name", default="lightgbm_expert")
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--min-regime-rows", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for regime in ("bullish", "bearish", "sideways"):
        cfg = TrainConfig(
            features_path=args.features_path,
            regimes_path=None,
            target_col="log_return_1_x",
            target_expr=None,
            target_shift=-1,
            group_col=None,
            vol_window=None,
            min_regime_rows=int(args.min_regime_rows),
            regime=regime,
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            run_name=f"{args.model_name}_{regime}_{ts}",
            output_dir=args.output_dir,
            id_cols=["timestamp"],
            drop_cols=["regime", "regime_explanation"],
            time_col="timestamp",
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            early_stopping_rounds=50,
            num_boost_round=2000,
            seed=int(args.seed),
            params_json=None,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
        out_dir = run(cfg)
        print(f"Retrained {regime}: {out_dir}")


if __name__ == "__main__":
    main()

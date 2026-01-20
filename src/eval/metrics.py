# src/eval/metrics.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# -------------------------
# Metrics
# -------------------------
def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    diff = (y_true - y_pred).abs()
    diff = diff.dropna()
    if diff.empty:
        return float("nan")
    return float(diff.mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    err2 = (y_true - y_pred) ** 2
    err2 = err2.dropna()
    if err2.empty:
        return float("nan")
    return float(math.sqrt(err2.mean()))


_METRIC_FNS = {
    "mae": mae,
    "rmse": rmse,
}


# -------------------------
# Schema helpers
# -------------------------
def _resolve_col_df(df: pd.DataFrame, base: str) -> str:
    """
    Resolve base / base_x / base_y in a dataframe.

    Priority:
      1) base
      2) base_x
      3) base_y

    This lets eval work for both long (single symbol) and wide (x/y) schemas.
    """
    if base in df.columns:
        return base
    if f"{base}_x" in df.columns:
        return f"{base}_x"
    if f"{base}_y" in df.columns:
        return f"{base}_y"
    raise ValueError(
        f"Missing required column '{base}' (or suffixed). Available: {list(df.columns)}"
    )


# -------------------------
# IO helpers
# -------------------------
def _ensure_row_id(
    df: pd.DataFrame, *, sort_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure a deterministic row_id exists. If not present, create it by:
    sort -> reset_index(drop=True) -> row_id = index

    Backward compatible across schemas:
      - long schema: expects ["timestamp","symbol"] (or configured)
      - wide schema: symbol often does not exist, so we fall back to ["timestamp"]
    """
    if "row_id" in df.columns:
        return df, sort_cols

    effective_sort_cols = list(sort_cols)

    # Wide schema compatibility: allow ("timestamp","symbol") configs even if symbol is absent.
    if "symbol" in effective_sort_cols and "symbol" not in df.columns:
        effective_sort_cols = [c for c in effective_sort_cols if c != "symbol"]

    missing = [c for c in effective_sort_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot create row_id, missing sort cols: {missing}")

    out = df.sort_values(effective_sort_cols, kind="mergesort").reset_index(drop=True).copy()
    out["row_id"] = out.index.astype(int)
    return out, effective_sort_cols


def load_latest_parquet(dirpath: str | Path) -> tuple[pd.DataFrame, Path]:
    dirpath = Path(dirpath)
    p = dirpath / "latest.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_parquet(p), p


# -------------------------
# Core evaluator
# -------------------------
@dataclass(frozen=True)
class EvalConfig:
    target_col: str = "log_return_1"
    regime_col: str = "regime"
    model_col: str = "model_name"
    pred_col: str = "y_pred"
    min_regime_n: int = 30
    metrics: tuple[str, ...] = ("mae", "rmse")
    lower_is_better: bool = True

    # how we reconstruct row_id for frames that do not have it
    row_id_sort_cols: tuple[str, ...] = ("timestamp", "symbol")


def build_eval_frame(
    *,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    cfg: EvalConfig,
) -> pd.DataFrame:
    """
    Returns a single dataframe with (row_id, model_name, y_pred, y_true, regime, ...)
    """
    # Predictions already have row_id
    required_pred = {"row_id", cfg.model_col, cfg.pred_col}
    missing_pred = required_pred - set(predictions.columns)
    if missing_pred:
        raise ValueError(f"Predictions missing columns: {sorted(missing_pred)}")

    feat, feat_sort_cols = _ensure_row_id(features, sort_cols=list(cfg.row_id_sort_cols))
    reg, reg_sort_cols = _ensure_row_id(regimes, sort_cols=list(cfg.row_id_sort_cols))

    # Resolve target column across schemas (log_return_1 vs log_return_1_x/log_return_1_y)
    effective_target_col = _resolve_col_df(feat, cfg.target_col)

    if cfg.regime_col not in reg.columns:
        raise ValueError(
            f"Regimes missing regime_col={cfg.regime_col}. Available: {list(reg.columns)}"
        )

    base = predictions[["row_id", cfg.model_col, cfg.pred_col]].copy()
    base = (
        base.merge(
            feat[["row_id", effective_target_col]],
            on="row_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            reg[["row_id", cfg.regime_col]],
            on="row_id",
            how="left",
            validate="many_to_one",
        )
        .rename(columns={effective_target_col: "y_true"})
    )

    # Store what rules actually got used (useful for debugging and scorecard honesty)
    base.attrs["feat_row_id_sort_cols"] = feat_sort_cols
    base.attrs["reg_row_id_sort_cols"] = reg_sort_cols
    base.attrs["effective_target_col"] = effective_target_col

    return base


def _rank_models(metric_by_model: dict[str, float], *, lower_is_better: bool) -> list[str]:
    # NaNs should go to the bottom
    def key(item: tuple[str, float]) -> tuple[int, float, str]:
        name, val = item
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return (1, float("inf"), name)
        return (0, val if lower_is_better else -val, name)

    return [name for name, _ in sorted(metric_by_model.items(), key=key)]


def compute_metrics_table(
    eval_df: pd.DataFrame,
    *,
    cfg: EvalConfig,
) -> pd.DataFrame:
    """
    Returns a long-form table:
    scope in {"overall","regime"}
    columns: scope, regime(optional), model_name, n, mae, rmse, ...
    """
    for m in cfg.metrics:
        if m not in _METRIC_FNS:
            raise ValueError(f"Unknown metric '{m}'. Supported: {sorted(_METRIC_FNS)}")

    rows: list[dict[str, Any]] = []

    # Overall
    for model_name, g in eval_df.groupby(cfg.model_col, sort=True):
        r: dict[str, Any] = {
            "scope": "overall",
            "regime": None,
            cfg.model_col: model_name,
            "n": int(g["y_true"].notna().sum()),
        }
        for m in cfg.metrics:
            r[m] = _METRIC_FNS[m](g["y_true"], g[cfg.pred_col])
        rows.append(r)

    # By regime
    for regime_val, rg in eval_df.groupby(cfg.regime_col, sort=True):
        for model_name, g in rg.groupby(cfg.model_col, sort=True):
            r = {
                "scope": "regime",
                "regime": str(regime_val),
                cfg.model_col: model_name,
                "n": int(g["y_true"].notna().sum()),
            }
            for m in cfg.metrics:
                r[m] = _METRIC_FNS[m](g["y_true"], g[cfg.pred_col])
            rows.append(r)

    return pd.DataFrame(rows)


def build_scorecard(
    eval_df: pd.DataFrame,
    *,
    cfg: EvalConfig,
    features_path: str | Path,
    regimes_path: str | Path,
    predictions_path: str | Path,
    timestamp: str,
) -> dict[str, Any]:
    metrics_tbl = compute_metrics_table(eval_df, cfg=cfg)

    # Overall section
    overall = metrics_tbl[metrics_tbl["scope"] == "overall"].copy()
    overall_by_model: dict[str, dict[str, float]] = {}
    for _, r in overall.iterrows():
        model = str(r[cfg.model_col])
        overall_by_model[model] = {m: float(r[m]) for m in cfg.metrics}

    # Choose rank metric = first metric in cfg.metrics
    primary = cfg.metrics[0]
    overall_rank = _rank_models(
        {k: v.get(primary, float("nan")) for k, v in overall_by_model.items()},
        lower_is_better=cfg.lower_is_better,
    )

    # By regime section
    by_regime: dict[str, Any] = {}
    regime_tbl = metrics_tbl[metrics_tbl["scope"] == "regime"].copy()
    for regime_val in sorted(regime_tbl["regime"].dropna().unique().tolist()):
        reg_slice = regime_tbl[regime_tbl["regime"] == regime_val]
        n_regime = int(eval_df.loc[eval_df[cfg.regime_col] == regime_val, "y_true"].notna().sum())

        by_model: dict[str, dict[str, float]] = {}
        for _, r in reg_slice.iterrows():
            model = str(r[cfg.model_col])
            by_model[model] = {m: float(r[m]) for m in cfg.metrics}

        rank = _rank_models(
            {k: v.get(primary, float("nan")) for k, v in by_model.items()},
            lower_is_better=cfg.lower_is_better,
        )

        by_regime[str(regime_val)] = {
            "n": n_regime,
            "by_model": by_model,
            "rank": rank,
        }

    feat_cols_used = eval_df.attrs.get("feat_row_id_sort_cols", list(cfg.row_id_sort_cols))
    reg_cols_used = eval_df.attrs.get("reg_row_id_sort_cols", list(cfg.row_id_sort_cols))
    effective_target_col = eval_df.attrs.get("effective_target_col", cfg.target_col)

    scorecard: dict[str, Any] = {
        "timestamp": timestamp,
        "data": {
            "features_path": str(features_path),
            "regimes_path": str(regimes_path),
            "predictions_path": str(predictions_path),
        },
        "target": {
            "y_true_col": str(effective_target_col),
            "requested_target_col": str(cfg.target_col),
        },
        "metrics": list(cfg.metrics),
        "overall": {
            "n": int(eval_df["y_true"].notna().sum()),
            "by_model": overall_by_model,
            "rank": overall_rank,
        },
        "by_regime": by_regime,
        "notes": {
            "min_regime_n": int(cfg.min_regime_n),
            "scoring_rule": "lower_is_better" if cfg.lower_is_better else "higher_is_better",
            "row_id_rule": (
                f"features: sort {feat_cols_used} then row_id=index; "
                f"regimes: sort {reg_cols_used} then row_id=index"
            ),
        },
    }
    return scorecard


def write_scorecard_artifacts(
    *,
    scorecard: dict[str, Any],
    metrics_table: pd.DataFrame,
    out_dir: str | Path = "data/scorecards",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = scorecard["timestamp"]
    json_path = out_dir / f"scorecard_{ts}.json"
    latest_json = out_dir / "latest.json"

    parquet_path = out_dir / f"scorecard_{ts}.parquet"
    latest_parquet = out_dir / "latest.parquet"

    json_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True))
    latest_json.write_text(json.dumps(scorecard, indent=2, sort_keys=True))

    metrics_table.to_parquet(parquet_path, index=False)
    metrics_table.to_parquet(latest_parquet, index=False)

    return json_path, parquet_path


def make_timestamp_id() -> str:
    # matches your prior PR style using epoch seconds
    return str(int(time.time()))

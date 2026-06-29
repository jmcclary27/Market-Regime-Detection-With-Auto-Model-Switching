from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.lineage import sha256_file, try_git_commit
from src.reporting.roadmap import parse_future_metrics_roadmap

LINEAGE_ARTIFACT_KEYS = (
    "raw_csv",
    "features_manifest",
    "features_parquet",
    "regimes_parquet",
    "predictions_parquet",
    "data_quality_audit_json",
    "pipeline_run_json",
    "walkforward_portfolio_metrics_parquet",
    "promotion_decision_json",
)

BOOL_FIELDS = (
    "has_git_commit",
    "has_config_sha256",
    "has_lineage_artifacts",
    "lineage_complete",
    "has_scorecard_json",
    "has_scorecard_parquet",
    "has_walk_forward_metrics_parquet",
    "has_walkforward_portfolio_metrics_parquet",
    "has_promotion_json",
    "has_regime_diagnostics_json",
    "has_data_quality_audit_json",
    "has_pipeline_run_json",
    "has_replay_audit_json",
)


@dataclass(frozen=True)
class HistoryRun:
    run_ts: str
    lineage_path: Path
    lineage: dict[str, Any]


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(os.environ.get("PROJECT_ROOT", str(Path.cwd()))).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def discover_lineage_runs(project_root: str | Path | None = None) -> list[HistoryRun]:
    root = _project_root(project_root)
    lineage_dir = root / "artifacts" / "lineage"
    runs: list[HistoryRun] = []
    for path in sorted(lineage_dir.glob("lineage_*.json")):
        data = _read_json(path)
        run_ts = str(data.get("run_ts", "")).strip() or path.stem.replace("lineage_", "", 1)
        runs.append(HistoryRun(run_ts=run_ts, lineage_path=path, lineage=data))
    runs.sort(key=lambda item: (item.run_ts, item.lineage_path.name))
    return runs


def resolve_subject_run_ts(
    runs: list[HistoryRun],
    *,
    project_root: str | Path | None = None,
    subject_run_ts: str = "latest",
) -> str:
    if not runs:
        raise FileNotFoundError("No lineage-backed runs found under artifacts/lineage")

    if subject_run_ts != "latest":
        valid = {run.run_ts for run in runs}
        if subject_run_ts not in valid:
            raise ValueError(
                f"Requested subject_run_ts={subject_run_ts!r} was not found. Available={sorted(valid)}"
            )
        return subject_run_ts

    root = _project_root(project_root)
    latest_path = root / "artifacts" / "lineage" / "latest.json"
    if latest_path.exists():
        data = _read_json(latest_path)
        latest_run_ts = str(data.get("run_ts", "")).strip()
        if latest_run_ts and latest_run_ts in {run.run_ts for run in runs}:
            return latest_run_ts

    return runs[-1].run_ts


def _lineage_artifact_paths(project_root: Path, run: HistoryRun) -> dict[str, Path]:
    raw_artifacts = run.lineage.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        return {}

    out: dict[str, Path] = {}
    for name, raw_value in raw_artifacts.items():
        if not isinstance(raw_value, dict):
            continue
        resolved = _resolve_path(project_root, cast(str | None, raw_value.get("path")))
        if resolved is not None:
            out[str(name)] = resolved
    return out


def _first_defined_path(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    for path in paths:
        if path is not None:
            return path
    return None


def _companion_paths(project_root: Path, run: HistoryRun) -> dict[str, Path | None]:
    run_ts = run.run_ts
    lineage_paths = _lineage_artifact_paths(project_root, run)
    return {
        "scorecard_json": project_root / "data" / "scorecards" / f"scorecard_{run_ts}.json",
        "scorecard_parquet": project_root / "data" / "scorecards" / f"scorecard_{run_ts}.parquet",
        "walk_forward_metrics_parquet": project_root
        / "data"
        / "scorecards"
        / f"walk_forward_metrics_{run_ts}.parquet",
        "walkforward_portfolio_metrics_parquet": _first_defined_path(
            [
                lineage_paths.get("walkforward_portfolio_metrics_parquet"),
                project_root / "data" / "walkforward" / f"portfolio_metrics_{run_ts}.parquet",
            ]
        ),
        "promotion_json": _first_defined_path(
            [
                lineage_paths.get("promotion_decision_json"),
                project_root / "data" / "walkforward" / f"promotion_{run_ts}.json",
            ]
        ),
        "data_quality_audit_json": _first_defined_path(
            [
                lineage_paths.get("data_quality_audit_json"),
                project_root / "artifacts" / "data_quality" / f"data_quality_{run_ts}.json",
            ]
        ),
        "pipeline_run_json": _first_defined_path(
            [
                lineage_paths.get("pipeline_run_json"),
                project_root / "artifacts" / "pipeline_runs" / f"pipeline_run_{run_ts}.json",
            ]
        ),
        "replay_audit_json": project_root / "artifacts" / "replay" / f"replay_{run_ts}.json",
        "regime_diagnostics_json": project_root
        / "artifacts"
        / "regimes"
        / f"diagnostics_{run_ts}.json",
    }


def _maybe_read_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return _read_json(path), None
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, str(exc)


def _maybe_read_parquet(path: Path | None) -> tuple[pd.DataFrame | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return pd.read_parquet(path), None
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, str(exc)


def _metric_value(frame: pd.DataFrame, *, model_name: str, metric: str) -> float | None:
    if metric not in frame.columns:
        return None
    rows = frame[frame["model_name"] == model_name]
    if rows.empty:
        return None
    return _finite_float(rows.iloc[0][metric])


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _ranked_models(frame: pd.DataFrame, metric: str, *, lower_is_better: bool) -> list[str]:
    if metric not in frame.columns or "model_name" not in frame.columns:
        return []
    tmp = frame[["model_name", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.replace([float("inf"), float("-inf")], pd.NA)
    tmp = tmp.dropna(subset=[metric])
    if tmp.empty:
        return []
    ascending = [lower_is_better, True]
    tmp = tmp.sort_values([metric, "model_name"], ascending=ascending, kind="mergesort")
    return [str(name) for name in tmp["model_name"].tolist()]


def _scorecard_json_to_table(scorecard: dict[str, Any]) -> pd.DataFrame:
    metrics = [str(metric) for metric in scorecard.get("metrics", [])]
    rows: list[dict[str, Any]] = []

    overall = scorecard.get("overall")
    if isinstance(overall, dict):
        by_model = overall.get("by_model")
        overall_n = overall.get("n")
        if isinstance(by_model, dict):
            for model_name, metric_map in by_model.items():
                if not isinstance(metric_map, dict):
                    continue
                row: dict[str, Any] = {
                    "scope": "overall",
                    "regime": None,
                    "model_name": str(model_name),
                    "n": overall_n,
                }
                for metric in metrics:
                    row[metric] = metric_map.get(metric)
                rows.append(row)

    by_regime = scorecard.get("by_regime")
    if isinstance(by_regime, dict):
        for regime_name, payload in by_regime.items():
            if not isinstance(payload, dict):
                continue
            regime_n = payload.get("n")
            by_model = payload.get("by_model")
            if not isinstance(by_model, dict):
                continue
            for model_name, metric_map in by_model.items():
                if not isinstance(metric_map, dict):
                    continue
                row = {
                    "scope": "regime",
                    "regime": str(regime_name),
                    "model_name": str(model_name),
                    "n": regime_n,
                }
                for metric in metrics:
                    row[metric] = metric_map.get(metric)
                rows.append(row)

    return pd.DataFrame(rows)


def _ensure_model_row(
    model_rows: dict[str, dict[str, Any]],
    *,
    run_ts: str,
    model_name: str,
) -> dict[str, Any]:
    row = model_rows.setdefault(
        model_name,
        {
            "run_ts": run_ts,
            "model_name": model_name,
        },
    )
    return row


def _apply_scorecard_metrics(
    *,
    run_ts: str,
    run_row: dict[str, Any],
    model_rows: dict[str, dict[str, Any]],
    scorecard_df: pd.DataFrame,
    scorecard_json: dict[str, Any] | None,
) -> None:
    if scorecard_df.empty or "model_name" not in scorecard_df.columns:
        return

    frame = scorecard_df.copy()
    frame["model_name"] = frame["model_name"].astype(str)
    if "scope" in frame.columns:
        frame["scope"] = frame["scope"].astype(str)
    if "regime" in frame.columns:
        frame["regime"] = frame["regime"].where(frame["regime"].isna(), frame["regime"].astype(str))
    for numeric_col in ("n", "mae", "rmse"):
        if numeric_col in frame.columns:
            frame[numeric_col] = pd.to_numeric(frame[numeric_col], errors="coerce")

    overall = (
        frame[frame["scope"] == "overall"].copy() if "scope" in frame.columns else frame.copy()
    )
    regime_rows = (
        frame[frame["scope"] == "regime"].copy()
        if "scope" in frame.columns
        else pd.DataFrame(columns=frame.columns)
    )

    metrics = [metric for metric in ("mae", "rmse") if metric in overall.columns]
    run_row["scorecard_model_count"] = int(overall["model_name"].nunique())
    run_row["scorecard_overall_n_max"] = (
        _finite_int(overall["n"].max()) if "n" in overall.columns else None
    )
    run_row["scorecard_regime_count"] = (
        int(regime_rows["regime"].dropna().nunique()) if "regime" in regime_rows.columns else 0
    )

    regime_win_counts: dict[str, dict[str, int]] = {metric: {} for metric in metrics}
    if not regime_rows.empty and "regime" in regime_rows.columns:
        for _regime_name, regime_frame in regime_rows.groupby("regime", sort=True, dropna=False):
            for metric in metrics:
                ranked = _ranked_models(regime_frame, metric, lower_is_better=True)
                if ranked:
                    winner = ranked[0]
                    regime_win_counts[metric][winner] = regime_win_counts[metric].get(winner, 0) + 1

    for metric in metrics:
        ranked = _ranked_models(overall, metric, lower_is_better=True)
        if ranked:
            run_row[f"best_model_{metric}"] = ranked[0]
            run_row[f"best_{metric}"] = _metric_value(overall, model_name=ranked[0], metric=metric)

        active_value = _metric_value(overall, model_name="active", metric=metric)
        baseline_value = _metric_value(overall, model_name="baseline", metric=metric)
        run_row[f"active_{metric}"] = active_value
        run_row[f"baseline_{metric}"] = baseline_value
        if active_value is not None and baseline_value is not None:
            run_row[f"active_vs_baseline_{metric}_delta"] = active_value - baseline_value
            run_row[f"active_beats_baseline_{metric}"] = active_value < baseline_value
        else:
            run_row[f"active_vs_baseline_{metric}_delta"] = None
            run_row[f"active_beats_baseline_{metric}"] = None

        rank_map = {model_name: idx + 1 for idx, model_name in enumerate(ranked)}
        for _, record in overall.iterrows():
            model_name = str(record["model_name"])
            row = _ensure_model_row(model_rows, run_ts=run_ts, model_name=model_name)
            row["overall_n"] = _finite_int(record.get("n"))
            row[f"overall_{metric}"] = _finite_float(record.get(metric))
            row[f"eval_rank_overall_{metric}"] = rank_map.get(model_name)
            row[f"eval_regime_win_count_{metric}"] = regime_win_counts[metric].get(model_name, 0)

    if not regime_rows.empty:
        for model_name in sorted(regime_rows["model_name"].astype(str).unique().tolist()):
            row = _ensure_model_row(model_rows, run_ts=run_ts, model_name=model_name)
            row["eval_regime_count"] = int(
                regime_rows.loc[regime_rows["model_name"].astype(str) == model_name, "regime"]
                .dropna()
                .nunique()
            )

    if scorecard_json is not None:
        target = scorecard_json.get("target")
        notes = scorecard_json.get("notes")
        if isinstance(target, dict):
            run_row["scorecard_requested_target_col"] = target.get("requested_target_col")
            run_row["scorecard_resolved_target_col"] = target.get("y_true_col")
        if isinstance(notes, dict):
            run_row["scorecard_scoring_rule"] = notes.get("scoring_rule")
            run_row["scorecard_row_id_rule"] = notes.get("row_id_rule")


def _fold_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("split_id", "fold_id"):
        if candidate in frame.columns:
            return candidate
    return None


def _apply_walkforward_metrics(
    *,
    run_ts: str,
    run_row: dict[str, Any],
    model_rows: dict[str, dict[str, Any]],
    portfolio_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
) -> None:
    if portfolio_df.empty or "model_name" not in portfolio_df.columns:
        return

    frame = portfolio_df.copy()
    frame["model_name"] = frame["model_name"].astype(str)
    for numeric_col in (
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "turnover",
        "profit_factor",
    ):
        if numeric_col in frame.columns:
            frame[numeric_col] = pd.to_numeric(frame[numeric_col], errors="coerce")

    fold_col = _fold_column(frame)
    if fold_col is not None:
        run_row["walkforward_fold_count"] = int(frame[fold_col].dropna().nunique())
    run_row["walkforward_model_count"] = int(frame["model_name"].nunique())

    grouped = frame.groupby("model_name", sort=True)
    sharpe_summary: dict[str, float] = {}

    for model_name, group in grouped:
        row = _ensure_model_row(model_rows, run_ts=run_ts, model_name=str(model_name))
        if fold_col is not None:
            row["wf_fold_count"] = int(group[fold_col].dropna().nunique())

        sharpe_series = (
            pd.to_numeric(group.get("sharpe"), errors="coerce")
            if "sharpe" in group.columns
            else pd.Series(dtype=float)
        )
        row["wf_sharpe_mean"] = (
            _finite_float(sharpe_series.mean()) if not sharpe_series.empty else None
        )
        row["wf_sharpe_median"] = (
            _finite_float(sharpe_series.median()) if not sharpe_series.empty else None
        )
        row["wf_sharpe_std"] = (
            _finite_float(sharpe_series.std(ddof=0)) if not sharpe_series.empty else None
        )
        row["wf_sharpe_finite_fold_count"] = int(sharpe_series.dropna().shape[0])
        row["wf_sharpe_positive_fold_count"] = int((sharpe_series.dropna() > 0.0).sum())

        for prefix in ("cagr", "sortino"):
            if prefix not in group.columns:
                continue
            series = pd.to_numeric(group[prefix], errors="coerce")
            row[f"wf_{prefix}_mean"] = _finite_float(series.mean()) if not series.empty else None
            row[f"wf_{prefix}_median"] = (
                _finite_float(series.median()) if not series.empty else None
            )
            row[f"wf_{prefix}_std"] = (
                _finite_float(series.std(ddof=0)) if not series.empty else None
            )

        if "max_drawdown" in group.columns:
            max_dd = pd.to_numeric(group["max_drawdown"], errors="coerce")
            row["wf_max_drawdown_worst"] = _finite_float(max_dd.min()) if not max_dd.empty else None
            row["wf_max_drawdown_mean"] = _finite_float(max_dd.mean()) if not max_dd.empty else None

        if "turnover" in group.columns:
            turnover = pd.to_numeric(group["turnover"], errors="coerce")
            row["wf_turnover_mean"] = _finite_float(turnover.mean()) if not turnover.empty else None

        if "profit_factor" in group.columns:
            profit_factor = pd.to_numeric(group["profit_factor"], errors="coerce")
            row["wf_profit_factor_mean"] = (
                _finite_float(profit_factor.replace([float("inf"), float("-inf")], pd.NA).mean())
                if not profit_factor.empty
                else None
            )

        if row.get("wf_sharpe_mean") is not None:
            sharpe_summary[str(model_name)] = cast(float, row["wf_sharpe_mean"])

    ranked = sorted(
        sharpe_summary.items(),
        key=lambda item: (-item[1], item[0]),
    )
    rank_map = {model_name: idx + 1 for idx, (model_name, _value) in enumerate(ranked)}
    for model_name, row in model_rows.items():
        row["wf_rank_mean_sharpe"] = rank_map.get(model_name)

    if ranked:
        run_row["walkforward_best_model_by_mean_sharpe"] = ranked[0][0]
        run_row["walkforward_best_mean_sharpe"] = ranked[0][1]

    active_sharpe = sharpe_summary.get("active")
    baseline_sharpe = sharpe_summary.get("baseline")
    run_row["walkforward_active_mean_sharpe"] = active_sharpe
    run_row["walkforward_baseline_mean_sharpe"] = baseline_sharpe
    if active_sharpe is not None and baseline_sharpe is not None:
        run_row["walkforward_active_vs_baseline_sharpe_delta"] = active_sharpe - baseline_sharpe
        run_row["walkforward_active_beats_baseline_sharpe"] = active_sharpe > baseline_sharpe
    else:
        run_row["walkforward_active_vs_baseline_sharpe_delta"] = None
        run_row["walkforward_active_beats_baseline_sharpe"] = None

    active_row = model_rows.get("active", {})
    baseline_row = model_rows.get("baseline", {})
    active_dd = _finite_float(active_row.get("wf_max_drawdown_worst"))
    baseline_dd = _finite_float(baseline_row.get("wf_max_drawdown_worst"))
    run_row["walkforward_active_worst_max_drawdown"] = active_dd
    run_row["walkforward_baseline_worst_max_drawdown"] = baseline_dd
    if active_dd is not None and baseline_dd is not None:
        run_row["walkforward_active_vs_baseline_max_drawdown_delta"] = active_dd - baseline_dd
    else:
        run_row["walkforward_active_vs_baseline_max_drawdown_delta"] = None

    if eval_df is not None and not eval_df.empty:
        eval_frame = eval_df.copy()
        fold_col_eval = _fold_column(eval_frame)
        run_row["walk_forward_eval_row_count"] = int(len(eval_frame))
        if fold_col_eval is not None:
            run_row["walk_forward_eval_fold_count"] = int(
                eval_frame[fold_col_eval].dropna().nunique()
            )

    baseline_for_delta = model_rows.get("baseline", {})
    baseline_sharpe_mean = _finite_float(baseline_for_delta.get("wf_sharpe_mean"))
    baseline_dd_worst = _finite_float(baseline_for_delta.get("wf_max_drawdown_worst"))
    for _model_name, row in model_rows.items():
        model_sharpe = _finite_float(row.get("wf_sharpe_mean"))
        model_dd = _finite_float(row.get("wf_max_drawdown_worst"))
        if baseline_sharpe_mean is not None and model_sharpe is not None:
            row["wf_delta_vs_baseline_sharpe_mean"] = model_sharpe - baseline_sharpe_mean
        if baseline_dd_worst is not None and model_dd is not None:
            row["wf_delta_vs_baseline_max_drawdown_worst"] = model_dd - baseline_dd_worst


def _apply_promotion_metrics(run_row: dict[str, Any], promotion: dict[str, Any]) -> None:
    run_row["promoted"] = bool(promotion.get("promoted"))
    run_row["promotion_pointer_written"] = bool(promotion.get("pointer_written"))
    run_row["challenger_model_name"] = promotion.get("challenger_model_name")
    run_row["incumbent_model_name"] = promotion.get("incumbent_model_name")
    run_row["promotion_reason"] = promotion.get("reason")

    decision = promotion.get("decision")
    if isinstance(decision, dict):
        run_row["promotion_decision_promote"] = decision.get("promote")
        run_row["promotion_decision_reason"] = decision.get("reason")
        deltas = decision.get("deltas")
        if isinstance(deltas, dict):
            run_row["promotion_sharpe_delta"] = _finite_float(deltas.get("sharpe"))
            run_row["promotion_max_drawdown_delta"] = _finite_float(deltas.get("max_drawdown"))

    promotion_guard = promotion.get("promotion_guard")
    if isinstance(promotion_guard, dict):
        run_row["promotion_guard_allowed"] = promotion_guard.get("allowed")
        run_row["promotion_guard_reason"] = promotion_guard.get("reason")

    non_promotable = promotion.get("non_promotable_models")
    if isinstance(non_promotable, list):
        items = sorted(str(item) for item in non_promotable)
        run_row["non_promotable_models"] = "|".join(items)
        run_row["non_promotable_model_count"] = len(items)


def _apply_regime_diagnostics(run_row: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    for field in (
        "avg_regime_duration",
        "n_regimes",
        "n_steps",
        "n_switches",
        "regime_entropy",
        "switches_per_1000_steps",
    ):
        value = diagnostics.get(field)
        run_row[field] = (
            _finite_float(value)
            if field.startswith("avg_") or field.endswith("entropy") or field.endswith("steps")
            else value
        )

    pct_time = diagnostics.get("pct_time_regime")
    if isinstance(pct_time, list):
        for idx, value in enumerate(pct_time):
            run_row[f"pct_time_regime_{idx}"] = _finite_float(value)

    confidence = diagnostics.get("confidence")
    if isinstance(confidence, dict):
        for field in ("mean", "p10", "p50", "p90", "min", "max"):
            run_row[f"confidence_{field}"] = _finite_float(confidence.get(field))


def _apply_data_quality_metrics(run_row: dict[str, Any], audit: dict[str, Any]) -> None:
    run_row["data_quality_status"] = audit.get("status")
    for field in (
        "duplicate_bar_rate",
        "missing_bar_rate",
        "late_data_rate",
        "provider_failure_rate",
        "stale_bar_p95_seconds",
    ):
        run_row[field] = _finite_float(audit.get(field))
    for field in (
        "duplicate_bar_count",
        "missing_bar_count",
        "late_data_count",
        "provider_failure_count",
        "provider_attempt_count",
        "row_count",
    ):
        run_row[field] = _finite_int(audit.get(field))
    run_row["data_quality_interval"] = audit.get("interval")


def _apply_pipeline_run_metrics(run_row: dict[str, Any], pipeline_run: dict[str, Any]) -> None:
    run_row["pipeline_status"] = pipeline_run.get("status")
    run_row["pipeline_duration_seconds"] = _finite_float(pipeline_run.get("duration_seconds"))
    run_row["pipeline_started_at_utc"] = pipeline_run.get("started_at_utc")
    run_row["pipeline_finished_at_utc"] = pipeline_run.get("finished_at_utc")

    steps = pipeline_run.get("steps")
    if not isinstance(steps, list):
        return

    run_row["pipeline_step_count"] = len(steps)
    run_row["pipeline_failed_step_count"] = sum(
        1 for step in steps if isinstance(step, dict) and step.get("status") == "failed"
    )

    for step in steps:
        if not isinstance(step, dict):
            continue
        name = str(step.get("name", "")).strip()
        if not name:
            continue
        run_row[f"pipeline_step_{name}_status"] = step.get("status")
        run_row[f"pipeline_step_{name}_duration_seconds"] = _finite_float(
            step.get("duration_seconds")
        )


def _apply_replay_metrics(run_row: dict[str, Any], replay_audit: dict[str, Any]) -> None:
    run_row["replay_status"] = replay_audit.get("status")
    run_row["replay_exact_pass"] = replay_audit.get("exact_pass")
    run_row["replay_semantic_pass"] = replay_audit.get("semantic_pass")
    run_row["replay_max_prediction_drift"] = _finite_float(replay_audit.get("max_prediction_drift"))
    failures = replay_audit.get("failure_breakdown")
    if isinstance(failures, list):
        run_row["replay_failure_count"] = len(failures)


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return _finite_float(series.quantile(quantile))


def _replay_summary(runs_df: pd.DataFrame) -> dict[str, Any]:
    if "has_replay_audit_json" not in runs_df.columns:
        return {"audit_count": 0}
    frame = runs_df[runs_df["has_replay_audit_json"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return {"audit_count": 0}
    out: dict[str, Any] = {"audit_count": int(len(frame))}
    if "replay_exact_pass" in frame.columns:
        out["exact_pass_rate"] = float(frame["replay_exact_pass"].fillna(False).astype(bool).mean())
    if "replay_semantic_pass" in frame.columns:
        out["semantic_pass_rate"] = float(
            frame["replay_semantic_pass"].fillna(False).astype(bool).mean()
        )
    if "replay_max_prediction_drift" in frame.columns:
        out["max_prediction_drift_worst"] = _finite_float(frame["replay_max_prediction_drift"].max())
    return out


def _data_quality_summary(runs_df: pd.DataFrame) -> dict[str, Any]:
    if "has_data_quality_audit_json" not in runs_df.columns:
        return {"audit_count": 0}
    frame = runs_df[runs_df["has_data_quality_audit_json"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return {"audit_count": 0}
    out: dict[str, Any] = {"audit_count": int(len(frame))}
    if "data_quality_status" in frame.columns:
        out["ok_rate"] = float((frame["data_quality_status"] == "ok").mean())
    for field in (
        "duplicate_bar_rate",
        "missing_bar_rate",
        "late_data_rate",
        "provider_failure_rate",
        "stale_bar_p95_seconds",
    ):
        if field in frame.columns:
            out[f"{field}_max"] = _finite_float(pd.to_numeric(frame[field], errors="coerce").max())
    return out


def _pipeline_summary(runs_df: pd.DataFrame) -> dict[str, Any]:
    if "has_pipeline_run_json" not in runs_df.columns:
        return {"run_count": 0}
    frame = runs_df[runs_df["has_pipeline_run_json"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return {"run_count": 0}
    out: dict[str, Any] = {"run_count": int(len(frame))}
    if "pipeline_status" in frame.columns:
        out["success_rate"] = float((frame["pipeline_status"] == "completed").mean())
    if "pipeline_duration_seconds" in frame.columns:
        durations = pd.to_numeric(frame["pipeline_duration_seconds"], errors="coerce")
        out["duration_p50_seconds"] = _series_quantile(durations, 0.50)
        out["duration_p95_seconds"] = _series_quantile(durations, 0.95)

    if {"pipeline_status", "pipeline_finished_at_utc"}.issubset(frame.columns):
        ordered = frame.copy()
        ordered["pipeline_finished_at_utc"] = pd.to_datetime(
            ordered["pipeline_finished_at_utc"], utc=True, errors="coerce"
        )
        ordered = ordered.sort_values("pipeline_finished_at_utc", kind="mergesort")
        recoveries: list[float] = []
        pending_failure: pd.Timestamp | None = None
        for _, row in ordered.iterrows():
            finished_at = row.get("pipeline_finished_at_utc")
            if pd.isna(finished_at):
                continue
            status = row.get("pipeline_status")
            if status == "failed":
                pending_failure = cast(pd.Timestamp, finished_at)
            elif status == "completed" and pending_failure is not None:
                recoveries.append(max((cast(pd.Timestamp, finished_at) - pending_failure).total_seconds(), 0.0))
                pending_failure = None
        if recoveries:
            out["mean_recovery_time_seconds"] = float(sum(recoveries) / len(recoveries))
    return out


def _deployment_and_registry_summary(
    project_root: Path,
) -> tuple[dict[str, Any] | None, set[Path], str | None]:
    events_path = project_root / "data" / "deployments" / "events.parquet"
    history_path = project_root / "registry" / "history.parquet"
    consumed: set[Path] = set()
    out: dict[str, Any] = {}

    events_df, events_error = _maybe_read_parquet(events_path)
    if events_df is not None:
        consumed.add(events_path)
        out["event_count"] = int(len(events_df))
        if "decision" in events_df.columns:
            decisions = events_df["decision"].fillna("unknown").astype(str).value_counts()
            out["promote_count"] = int(decisions.get("promote", 0))
            out["rollback_count"] = int(decisions.get("rollback", 0))
            out["hold_count"] = int(decisions.get("hold", 0))
            out["blocked_count"] = int(decisions.get("blocked", 0))
            completed = sum(int(decisions.get(name, 0)) for name in ("promote", "rollback", "hold", "blocked"))
            out["canary_completion_rate"] = completed / float(max(len(events_df), 1))
            out["promotion_precision"] = int(decisions.get("promote", 0)) / float(
                max(int(decisions.get("promote", 0)) + int(decisions.get("rollback", 0)), 1)
            )
        if "pointer_written" in events_df.columns:
            out["pointer_written_rate"] = float(
                events_df["pointer_written"].fillna(False).astype(bool).mean()
            )
    elif events_error is not None:
        return None, consumed, events_error

    history_df, history_error = _maybe_read_parquet(history_path)
    if history_df is not None:
        consumed.add(history_path)
        out["registry_change_count"] = int(len(history_df))
        if "ts" in history_df.columns and not history_df.empty:
            hist = history_df.copy()
            hist["ts"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce")
            hist = hist.sort_values("ts", kind="mergesort")
            tenures: list[float] = []
            for idx in range(1, len(hist)):
                prev_ts = hist.iloc[idx - 1]["ts"]
                cur_ts = hist.iloc[idx]["ts"]
                if pd.isna(prev_ts) or pd.isna(cur_ts):
                    continue
                tenures.append(max((cur_ts - prev_ts).total_seconds(), 0.0))
            if tenures:
                out["active_model_tenure_mean_seconds"] = float(sum(tenures) / len(tenures))
            if len(hist) > 1:
                first_ts = hist.iloc[0]["ts"]
                last_ts = hist.iloc[-1]["ts"]
                if not pd.isna(first_ts) and not pd.isna(last_ts):
                    window_seconds = max((last_ts - first_ts).total_seconds(), 0.0)
                    if window_seconds > 0.0:
                        out["registry_churn_rate_per_day"] = (len(hist) - 1) / (
                            window_seconds / 86400.0
                        )
    elif history_error is not None:
        return None, consumed, history_error

    if not out:
        return None, consumed, None
    return out, consumed, None


def _test_inventory(project_root: Path) -> tuple[dict[str, Any], set[Path]]:
    tests_dir = project_root / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))
    consumed = set(test_files)
    test_function_count = 0
    pattern = re.compile(r"^def test_", re.MULTILINE)
    for path in test_files:
        text = path.read_text(encoding="utf-8")
        test_function_count += len(pattern.findall(text))

    return (
        {
            "test_file_count": len(test_files),
            "test_case_count": test_function_count,
            "saved_scorecard_count": len(
                list((project_root / "data" / "scorecards").glob("scorecard_*.parquet"))
            ),
            "saved_lineage_count": len(
                list((project_root / "artifacts" / "lineage").glob("lineage_*.json"))
            ),
            "saved_replay_audit_count": len(
                list((project_root / "artifacts" / "replay").glob("replay_*.json"))
            ),
            "saved_data_quality_audit_count": len(
                list((project_root / "artifacts" / "data_quality").glob("data_quality_*.json"))
            ),
            "saved_pipeline_run_count": len(
                list((project_root / "artifacts" / "pipeline_runs").glob("pipeline_run_*.json"))
            ),
        },
        consumed,
    )


def _live_sim_summary(project_root: Path) -> tuple[dict[str, Any] | None, set[Path], str | None]:
    live_dir = project_root / "data" / "live_sim"
    account_path = live_dir / "account_state.json"
    equity_path = live_dir / "equity_curve.parquet"
    trades_path = live_dir / "trades.parquet"

    if not account_path.exists() and not equity_path.exists() and not trades_path.exists():
        return None, set(), None

    consumed: set[Path] = set()
    out: dict[str, Any] = {"scope": "project_current"}

    account_json, account_error = _maybe_read_json(account_path)
    if account_json is not None:
        consumed.add(account_path)
        for field in (
            "cash",
            "position",
            "portfolio_value",
            "unrealized_pnl",
            "total_pnl",
            "last_price",
        ):
            out[f"latest_{field}"] = _finite_float(account_json.get(field))
        out["account_updated_at"] = account_json.get("updated_at")
    elif account_error is not None:
        return None, consumed, account_error

    equity_df, equity_error = _maybe_read_parquet(equity_path)
    if equity_df is not None and not equity_df.empty:
        consumed.add(equity_path)
        eq = equity_df.copy()
        if "timestamp" in eq.columns:
            eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
        if "portfolio_value" in eq.columns:
            values = pd.to_numeric(eq["portfolio_value"], errors="coerce")
            first_value = _finite_float(values.iloc[0]) if not values.empty else None
            last_value = _finite_float(values.iloc[-1]) if not values.empty else None
            out["start_portfolio_value"] = first_value
            out["end_portfolio_value"] = last_value
            if first_value is not None and first_value != 0.0 and last_value is not None:
                out["total_return_pct"] = ((last_value / first_value) - 1.0) * 100.0

            peak = values.cummax()
            drawdown = (values / peak) - 1.0
            out["max_drawdown_pct"] = (
                _finite_float(drawdown.min() * 100.0) if not drawdown.empty else None
            )

        if "timestamp" in eq.columns and not eq["timestamp"].dropna().empty:
            out["start_timestamp"] = str(eq["timestamp"].dropna().iloc[0].isoformat())
            out["end_timestamp"] = str(eq["timestamp"].dropna().iloc[-1].isoformat())
        if "regime" in eq.columns:
            out["unique_regimes_seen"] = int(eq["regime"].dropna().astype(str).nunique())
        if "active_model_id" in eq.columns:
            out["unique_active_models_seen"] = int(
                eq["active_model_id"].dropna().astype(str).nunique()
            )
    elif equity_error is not None:
        return None, consumed, equity_error

    trades_df, trades_error = _maybe_read_parquet(trades_path)
    if trades_df is not None:
        consumed.add(trades_path)
        out["trade_count"] = int(len(trades_df))
        if "action" in trades_df.columns:
            action_counts = trades_df["action"].fillna("UNKNOWN").astype(str).value_counts()
            out["buy_count"] = int(action_counts.get("BUY", 0))
            out["sell_count"] = int(action_counts.get("SELL", 0))
            out["none_count"] = int(action_counts.get("NONE", 0))
    elif trades_error is not None:
        return None, consumed, trades_error

    return out, consumed, None


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _sanitize_json(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    if value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _format_number(value: Any, digits: int = 4) -> str:
    val = _finite_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def _format_int(value: Any) -> str:
    val = _finite_int(value)
    if val is None:
        return "n/a"
    return f"{val}"


def _format_bool(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return "yes" if bool(value) else "no"


def _markdown_summary(
    *,
    subject_run_ts: str,
    history_summary: dict[str, Any],
    subject_run: dict[str, Any],
    subject_models: list[dict[str, Any]],
    live_sim: dict[str, Any] | None,
    inventory: dict[str, Any],
    replay_summary: dict[str, Any],
    deployments_summary: dict[str, Any] | None,
    data_quality_summary: dict[str, Any],
    pipeline_summary: dict[str, Any],
    roadmap_summary: dict[str, Any],
    roadmap_path_display: str,
) -> str:
    top_models = subject_models[:5]
    lines = [
        "# Project Metrics Report",
        "",
        "## Summary",
        f"- Subject run: `{subject_run_ts}`",
        f"- Canonical lineage-backed runs: `{_format_int(history_summary.get('run_count'))}`",
        f"- History window: `{history_summary.get('earliest_run_ts', 'n/a')}` to `{history_summary.get('latest_run_ts', 'n/a')}`",
        f"- Roadmap: `{roadmap_path_display}`",
        "",
        "## Headline Metrics",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Models evaluated | {_format_int(subject_run.get('model_count'))} |",
        f"| Walk-forward folds | {_format_int(subject_run.get('walkforward_fold_count'))} |",
        f"| Active RMSE | {_format_number(subject_run.get('active_rmse'))} |",
        f"| Baseline RMSE | {_format_number(subject_run.get('baseline_rmse'))} |",
        f"| Active vs baseline RMSE delta | {_format_number(subject_run.get('active_vs_baseline_rmse_delta'))} |",
        f"| Active mean Sharpe | {_format_number(subject_run.get('walkforward_active_mean_sharpe'))} |",
        f"| Baseline mean Sharpe | {_format_number(subject_run.get('walkforward_baseline_mean_sharpe'))} |",
        f"| Active vs baseline Sharpe delta | {_format_number(subject_run.get('walkforward_active_vs_baseline_sharpe_delta'))} |",
        f"| Promotion written | {_format_bool(subject_run.get('promoted'))} |",
        f"| Promotion guard allowed | {_format_bool(subject_run.get('promotion_guard_allowed'))} |",
        "",
        "## Top Models",
        "| Model | Overall RMSE | Overall MAE | Mean Sharpe | Worst Max Drawdown | Eval Rank RMSE | WF Rank Sharpe |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in top_models:
        lines.append(
            "| "
            + f"{row.get('model_name', 'n/a')} | "
            + f"{_format_number(row.get('overall_rmse'))} | "
            + f"{_format_number(row.get('overall_mae'))} | "
            + f"{_format_number(row.get('wf_sharpe_mean'))} | "
            + f"{_format_number(row.get('wf_max_drawdown_worst'))} | "
            + f"{_format_int(row.get('eval_rank_overall_rmse'))} | "
            + f"{_format_int(row.get('wf_rank_mean_sharpe'))} |"
        )

    lines.extend(
        [
            "",
            "## Regime Health",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Regime entropy | {_format_number(subject_run.get('regime_entropy'))} |",
            f"| Average regime duration | {_format_number(subject_run.get('avg_regime_duration'))} |",
            f"| Switches per 1,000 steps | {_format_number(subject_run.get('switches_per_1000_steps'))} |",
            f"| Confidence p10 / p50 / p90 | {_format_number(subject_run.get('confidence_p10'))} / {_format_number(subject_run.get('confidence_p50'))} / {_format_number(subject_run.get('confidence_p90'))} |",
            "",
        ]
    )

    if live_sim is not None:
        lines.extend(
            [
                "## Live Sim",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Start portfolio value | {_format_number(live_sim.get('start_portfolio_value'), 2)} |",
                f"| End portfolio value | {_format_number(live_sim.get('end_portfolio_value'), 2)} |",
                f"| Total return % | {_format_number(live_sim.get('total_return_pct'), 2)} |",
                f"| Max drawdown % | {_format_number(live_sim.get('max_drawdown_pct'), 2)} |",
                f"| Trade count | {_format_int(live_sim.get('trade_count'))} |",
                f"| Unique regimes seen | {_format_int(live_sim.get('unique_regimes_seen'))} |",
                f"| Unique active models seen | {_format_int(live_sim.get('unique_active_models_seen'))} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Inventory and Versioning",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Lineage completeness rate | {_format_number(history_summary.get('lineage_completeness_rate'))} |",
            f"| Git commit presence rate | {_format_number(history_summary.get('git_commit_presence_rate'))} |",
            f"| Config hash presence rate | {_format_number(history_summary.get('config_sha256_presence_rate'))} |",
            f"| Saved scorecard count | {_format_int(inventory.get('saved_scorecard_count'))} |",
            f"| Saved lineage count | {_format_int(inventory.get('saved_lineage_count'))} |",
            f"| Test file count | {_format_int(inventory.get('test_file_count'))} |",
            f"| Test case count | {_format_int(inventory.get('test_case_count'))} |",
            f"| Replay audits saved | {_format_int(inventory.get('saved_replay_audit_count'))} |",
            f"| Data-quality audits saved | {_format_int(inventory.get('saved_data_quality_audit_count'))} |",
            f"| Pipeline summaries saved | {_format_int(inventory.get('saved_pipeline_run_count'))} |",
            "",
        ]
    )
    lines.extend(
        [
            "## Replay",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Replay audit count | {_format_int(replay_summary.get('audit_count'))} |",
            f"| Exact replay pass rate | {_format_number(replay_summary.get('exact_pass_rate'))} |",
            f"| Semantic replay pass rate | {_format_number(replay_summary.get('semantic_pass_rate'))} |",
            f"| Worst max prediction drift | {_format_number(replay_summary.get('max_prediction_drift_worst'))} |",
            "",
            "## Data Quality",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Data-quality audit count | {_format_int(data_quality_summary.get('audit_count'))} |",
            f"| Data-quality ok rate | {_format_number(data_quality_summary.get('ok_rate'))} |",
            f"| Max missing-bar rate | {_format_number(data_quality_summary.get('missing_bar_rate_max'))} |",
            f"| Max duplicate-bar rate | {_format_number(data_quality_summary.get('duplicate_bar_rate_max'))} |",
            f"| Max late-data rate | {_format_number(data_quality_summary.get('late_data_rate_max'))} |",
            "",
            "## Pipeline Telemetry",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Pipeline summaries | {_format_int(pipeline_summary.get('run_count'))} |",
            f"| Pipeline success rate | {_format_number(pipeline_summary.get('success_rate'))} |",
            f"| Runtime p50 seconds | {_format_number(pipeline_summary.get('duration_p50_seconds'))} |",
            f"| Runtime p95 seconds | {_format_number(pipeline_summary.get('duration_p95_seconds'))} |",
            f"| Mean recovery time seconds | {_format_number(pipeline_summary.get('mean_recovery_time_seconds'))} |",
            "",
        ]
    )

    if deployments_summary is not None:
        lines.extend(
            [
                "## Deployments",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Deployment events | {_format_int(deployments_summary.get('event_count'))} |",
                f"| Promote count | {_format_int(deployments_summary.get('promote_count'))} |",
                f"| Rollback count | {_format_int(deployments_summary.get('rollback_count'))} |",
                f"| Canary completion rate | {_format_number(deployments_summary.get('canary_completion_rate'))} |",
                f"| Registry change count | {_format_int(deployments_summary.get('registry_change_count'))} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Roadmap Coverage",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Implemented capabilities | {_format_int(roadmap_summary.get('implemented_count'))} |",
            f"| Partial capabilities | {_format_int(roadmap_summary.get('partial_count'))} |",
            f"| Planned capabilities | {_format_int(roadmap_summary.get('planned_count'))} |",
            f"| Remaining capabilities | {_format_int(len(cast(list[Any], roadmap_summary.get('remaining_capabilities', []))))} |",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _history_summary(runs_df: pd.DataFrame, inventory: dict[str, Any]) -> dict[str, Any]:
    if runs_df.empty:
        return {
            "run_count": 0,
            "earliest_run_ts": None,
            "latest_run_ts": None,
            **inventory,
        }

    out: dict[str, Any] = {
        "run_count": int(len(runs_df)),
        "earliest_run_ts": str(runs_df["run_ts"].min()),
        "latest_run_ts": str(runs_df["run_ts"].max()),
        **inventory,
    }

    if "lineage_complete" in runs_df.columns:
        out["lineage_completeness_rate"] = float(
            runs_df["lineage_complete"].fillna(False).astype(bool).mean()
        )
    if "has_git_commit" in runs_df.columns:
        out["git_commit_presence_rate"] = float(
            runs_df["has_git_commit"].fillna(False).astype(bool).mean()
        )
    if "has_config_sha256" in runs_df.columns:
        out["config_sha256_presence_rate"] = float(
            runs_df["has_config_sha256"].fillna(False).astype(bool).mean()
        )

    for column in sorted(
        col
        for col in runs_df.columns
        if col.startswith("has_") and col not in {"has_git_commit", "has_config_sha256"}
    ):
        out[f"{column}_rate"] = float(runs_df[column].fillna(False).astype(bool).mean())

    return out


def _subject_models_rows(models_df: pd.DataFrame, subject_run_ts: str) -> list[dict[str, Any]]:
    if models_df.empty:
        return []
    frame = models_df[models_df["run_ts"] == subject_run_ts].copy()
    if frame.empty:
        return []

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for column in ("wf_rank_mean_sharpe", "eval_rank_overall_rmse"):
        if column in frame.columns:
            sort_cols.append(column)
            ascending.append(True)
    sort_cols.append("model_name")
    ascending.append(True)

    frame = frame.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last")
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _order_columns(df: pd.DataFrame, leading: list[str]) -> pd.DataFrame:
    available_leading = [column for column in leading if column in df.columns]
    remaining = sorted(column for column in df.columns if column not in set(available_leading))
    return df[available_leading + remaining]


def generate_project_metrics_report(
    *,
    project_root: str | Path | None = None,
    subject_run_ts: str = "latest",
    history_source: str = "lineage",
    out_dir: str | Path = "data/reporting",
    roadmap_path: str | Path = "docs/future_metrics_roadmap.md",
) -> dict[str, Path]:
    root = _project_root(project_root)
    if history_source != "lineage":
        raise ValueError(f"Unsupported history_source={history_source!r}. Expected 'lineage'.")

    runs = discover_lineage_runs(root)
    resolved_subject_run_ts = resolve_subject_run_ts(
        runs, project_root=root, subject_run_ts=subject_run_ts
    )

    output_dir = _resolve_path(root, str(out_dir))
    if output_dir is None:
        raise ValueError("out_dir could not be resolved")
    output_dir.mkdir(parents=True, exist_ok=True)

    roadmap = _resolve_path(root, str(roadmap_path))
    if roadmap is None:
        raise ValueError("roadmap_path could not be resolved")

    consumed_paths: set[Path] = set()
    run_rows: list[dict[str, Any]] = []
    model_rows_out: list[dict[str, Any]] = []

    for run in runs:
        run_row: dict[str, Any] = {
            "run_ts": run.run_ts,
            "git_commit": run.lineage.get("git_commit"),
            "config_sha256": run.lineage.get("config_sha256"),
            "has_git_commit": bool(str(run.lineage.get("git_commit", "")).strip()),
            "has_config_sha256": bool(str(run.lineage.get("config_sha256", "")).strip()),
            "has_lineage_artifacts": bool(run.lineage.get("artifacts")),
        }
        run_row["lineage_complete"] = (
            run_row["has_git_commit"]
            and run_row["has_config_sha256"]
            and run_row["has_lineage_artifacts"]
        )
        consumed_paths.add(run.lineage_path)

        recorded_artifacts = _lineage_artifact_paths(root, run)
        run_row["lineage_artifact_ref_count"] = len(recorded_artifacts)
        for key in LINEAGE_ARTIFACT_KEYS:
            run_row[f"has_recorded_{key}"] = key in recorded_artifacts

        companions = _companion_paths(root, run)
        for key, path in companions.items():
            run_row[f"has_{key}"] = bool(path is not None and path.exists())

        model_rows: dict[str, dict[str, Any]] = {}

        scorecard_json, scorecard_json_error = _maybe_read_json(companions.get("scorecard_json"))
        if scorecard_json is not None and companions.get("scorecard_json") is not None:
            consumed_paths.add(cast(Path, companions["scorecard_json"]))
        elif scorecard_json_error is not None:
            run_row["scorecard_json_error"] = scorecard_json_error

        scorecard_df, scorecard_parquet_error = _maybe_read_parquet(
            companions.get("scorecard_parquet")
        )
        if scorecard_df is not None and companions.get("scorecard_parquet") is not None:
            consumed_paths.add(cast(Path, companions["scorecard_parquet"]))
        elif scorecard_parquet_error is not None:
            run_row["scorecard_parquet_error"] = scorecard_parquet_error

        if scorecard_df is None and scorecard_json is not None:
            scorecard_df = _scorecard_json_to_table(scorecard_json)
        if scorecard_df is not None:
            _apply_scorecard_metrics(
                run_ts=run.run_ts,
                run_row=run_row,
                model_rows=model_rows,
                scorecard_df=scorecard_df,
                scorecard_json=scorecard_json,
            )

        walk_forward_eval_df, walk_forward_eval_error = _maybe_read_parquet(
            companions.get("walk_forward_metrics_parquet")
        )
        if (
            walk_forward_eval_df is not None
            and companions.get("walk_forward_metrics_parquet") is not None
        ):
            consumed_paths.add(cast(Path, companions["walk_forward_metrics_parquet"]))
        elif walk_forward_eval_error is not None:
            run_row["walk_forward_metrics_error"] = walk_forward_eval_error

        portfolio_df, portfolio_error = _maybe_read_parquet(
            companions.get("walkforward_portfolio_metrics_parquet")
        )
        if (
            portfolio_df is not None
            and companions.get("walkforward_portfolio_metrics_parquet") is not None
        ):
            consumed_paths.add(cast(Path, companions["walkforward_portfolio_metrics_parquet"]))
        elif portfolio_error is not None:
            run_row["walkforward_portfolio_metrics_error"] = portfolio_error

        if portfolio_df is not None:
            _apply_walkforward_metrics(
                run_ts=run.run_ts,
                run_row=run_row,
                model_rows=model_rows,
                portfolio_df=portfolio_df,
                eval_df=walk_forward_eval_df,
            )

        promotion_json, promotion_error = _maybe_read_json(companions.get("promotion_json"))
        if promotion_json is not None and companions.get("promotion_json") is not None:
            consumed_paths.add(cast(Path, companions["promotion_json"]))
            _apply_promotion_metrics(run_row, promotion_json)
        elif promotion_error is not None:
            run_row["promotion_json_error"] = promotion_error

        data_quality_json, data_quality_error = _maybe_read_json(
            companions.get("data_quality_audit_json")
        )
        if (
            data_quality_json is not None
            and companions.get("data_quality_audit_json") is not None
        ):
            consumed_paths.add(cast(Path, companions["data_quality_audit_json"]))
            _apply_data_quality_metrics(run_row, data_quality_json)
        elif data_quality_error is not None:
            run_row["data_quality_audit_error"] = data_quality_error

        pipeline_run_json, pipeline_run_error = _maybe_read_json(companions.get("pipeline_run_json"))
        if pipeline_run_json is not None and companions.get("pipeline_run_json") is not None:
            consumed_paths.add(cast(Path, companions["pipeline_run_json"]))
            _apply_pipeline_run_metrics(run_row, pipeline_run_json)
        elif pipeline_run_error is not None:
            run_row["pipeline_run_error"] = pipeline_run_error

        replay_audit_json, replay_audit_error = _maybe_read_json(companions.get("replay_audit_json"))
        if replay_audit_json is not None and companions.get("replay_audit_json") is not None:
            consumed_paths.add(cast(Path, companions["replay_audit_json"]))
            _apply_replay_metrics(run_row, replay_audit_json)
        elif replay_audit_error is not None:
            run_row["replay_audit_error"] = replay_audit_error

        diagnostics_json, diagnostics_error = _maybe_read_json(
            companions.get("regime_diagnostics_json")
        )
        if diagnostics_json is not None and companions.get("regime_diagnostics_json") is not None:
            consumed_paths.add(cast(Path, companions["regime_diagnostics_json"]))
            _apply_regime_diagnostics(run_row, diagnostics_json)
        elif diagnostics_error is not None:
            run_row["regime_diagnostics_error"] = diagnostics_error

        run_row["model_count"] = len(model_rows)

        for _model_name, row in sorted(model_rows.items(), key=lambda item: item[0]):
            row["has_eval_metrics"] = (
                row.get("overall_rmse") is not None or row.get("overall_mae") is not None
            )
            row["has_walkforward_metrics"] = row.get("wf_fold_count") is not None
            model_rows_out.append(row)

        run_rows.append(run_row)

    runs_df = pd.DataFrame(run_rows).sort_values("run_ts", kind="mergesort").reset_index(drop=True)
    models_df = (
        pd.DataFrame(model_rows_out)
        .sort_values(["run_ts", "model_name"], kind="mergesort")
        .reset_index(drop=True)
        if model_rows_out
        else pd.DataFrame(columns=["run_ts", "model_name"])
    )

    for field in BOOL_FIELDS:
        if field in runs_df.columns:
            runs_df[field] = runs_df[field].fillna(False).astype(bool)
    for field in ("has_eval_metrics", "has_walkforward_metrics"):
        if field in models_df.columns:
            models_df[field] = models_df[field].fillna(False).astype(bool)

    runs_df = _order_columns(
        runs_df,
        [
            "run_ts",
            "git_commit",
            "config_sha256",
            "has_git_commit",
            "has_config_sha256",
            "has_lineage_artifacts",
            "lineage_complete",
            "model_count",
        ],
    )
    models_df = _order_columns(
        models_df,
        [
            "run_ts",
            "model_name",
            "overall_n",
            "overall_mae",
            "overall_rmse",
            "eval_rank_overall_mae",
            "eval_rank_overall_rmse",
            "wf_fold_count",
            "wf_sharpe_mean",
            "wf_rank_mean_sharpe",
            "has_eval_metrics",
            "has_walkforward_metrics",
        ],
    )

    inventory, test_paths = _test_inventory(root)
    consumed_paths.update(test_paths)

    live_sim, live_sim_paths, live_sim_error = _live_sim_summary(root)
    consumed_paths.update(live_sim_paths)
    deployments_summary, deployment_paths, deployments_error = _deployment_and_registry_summary(root)
    consumed_paths.update(deployment_paths)

    if roadmap.exists():
        consumed_paths.add(roadmap)

    history_summary = _history_summary(runs_df, inventory)
    if live_sim_error is not None:
        history_summary["live_sim_error"] = live_sim_error
    if deployments_error is not None:
        history_summary["deployments_error"] = deployments_error

    replay_summary = _replay_summary(runs_df)
    data_quality_summary = _data_quality_summary(runs_df)
    pipeline_summary = _pipeline_summary(runs_df)
    roadmap_summary = parse_future_metrics_roadmap(roadmap)

    roadmap_display = (
        roadmap.relative_to(root).as_posix() if roadmap.is_relative_to(root) else roadmap.as_posix()
    )

    subject_run_frame = runs_df[runs_df["run_ts"] == resolved_subject_run_ts]
    if subject_run_frame.empty:
        raise RuntimeError(
            f"Subject run {resolved_subject_run_ts!r} disappeared during report build"
        )
    subject_run = cast(dict[str, Any], subject_run_frame.iloc[0].to_dict())
    subject_models = _subject_models_rows(models_df, resolved_subject_run_ts)

    report_json = {
        "subject_run_ts": resolved_subject_run_ts,
        "history_source": history_source,
        "history_summary": history_summary,
        "subject": {
            "run": subject_run,
            "models": subject_models,
            "topline": {
                "model_count": subject_run.get("model_count"),
                "walkforward_fold_count": subject_run.get("walkforward_fold_count"),
                "active_rmse": subject_run.get("active_rmse"),
                "baseline_rmse": subject_run.get("baseline_rmse"),
                "active_vs_baseline_rmse_delta": subject_run.get("active_vs_baseline_rmse_delta"),
                "walkforward_active_mean_sharpe": subject_run.get("walkforward_active_mean_sharpe"),
                "walkforward_baseline_mean_sharpe": subject_run.get(
                    "walkforward_baseline_mean_sharpe"
                ),
                "walkforward_active_vs_baseline_sharpe_delta": subject_run.get(
                    "walkforward_active_vs_baseline_sharpe_delta"
                ),
                "promoted": subject_run.get("promoted"),
                "challenger_model_name": subject_run.get("challenger_model_name"),
                "incumbent_model_name": subject_run.get("incumbent_model_name"),
            },
        },
        "live_sim": live_sim,
        "replay": replay_summary,
        "deployments": deployments_summary,
        "data_quality": data_quality_summary,
        "pipeline": pipeline_summary,
        "inventory": inventory,
        "roadmap": {**roadmap_summary, "path": roadmap_display, "exists": roadmap.exists()},
    }
    report_json = cast(dict[str, Any], _sanitize_json(report_json))

    markdown_text = _markdown_summary(
        subject_run_ts=resolved_subject_run_ts,
        history_summary=history_summary,
        subject_run=subject_run,
        subject_models=subject_models,
        live_sim=live_sim,
        inventory=inventory,
        replay_summary=replay_summary,
        deployments_summary=deployments_summary,
        data_quality_summary=data_quality_summary,
        pipeline_summary=pipeline_summary,
        roadmap_summary=roadmap_summary,
        roadmap_path_display=roadmap_display,
    )

    runs_history_path = output_dir / "runs_history.parquet"
    models_history_path = output_dir / "models_history.parquet"
    latest_report_json_path = output_dir / "latest_report.json"
    latest_report_md_path = output_dir / "latest_report.md"
    report_manifest_path = output_dir / "report_manifest.json"

    runs_df.to_parquet(runs_history_path, index=False)
    models_df.to_parquet(models_history_path, index=False)
    latest_report_json_path.write_text(
        json.dumps(report_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    latest_report_md_path.write_text(markdown_text, encoding="utf-8")

    manifest = {
        "subject_run_ts": resolved_subject_run_ts,
        "history_source": history_source,
        "history_run_ts": [run.run_ts for run in runs],
        "collector_args": {
            "subject_run_ts": subject_run_ts,
            "history_source": history_source,
            "out_dir": str(out_dir),
            "roadmap_path": str(roadmap_path),
        },
        "git_commit": try_git_commit(root),
        "input_hashes": {
            (
                path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
            ): sha256_file(path)
            for path in sorted(consumed_paths)
            if path.exists() and path.is_file()
        },
        "output_hashes": {
            "runs_history.parquet": sha256_file(runs_history_path),
            "models_history.parquet": sha256_file(models_history_path),
            "latest_report.json": sha256_file(latest_report_json_path),
            "latest_report.md": sha256_file(latest_report_md_path),
        },
    }
    report_manifest_path.write_text(
        json.dumps(_sanitize_json(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "runs_history_parquet": runs_history_path,
        "models_history_parquet": models_history_path,
        "latest_report_json": latest_report_json_path,
        "latest_report_md": latest_report_md_path,
        "report_manifest_json": report_manifest_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect project metrics from saved artifacts.")
    parser.add_argument("--subject-run-ts", default="latest")
    parser.add_argument("--history-source", default="lineage")
    parser.add_argument("--out-dir", default="data/reporting")
    parser.add_argument("--roadmap-path", default="docs/future_metrics_roadmap.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    outputs = generate_project_metrics_report(
        subject_run_ts=str(args.subject_run_ts),
        history_source=str(args.history_source),
        out_dir=str(args.out_dir),
        roadmap_path=str(args.roadmap_path),
    )

    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

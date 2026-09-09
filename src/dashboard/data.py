from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyarrow.parquet import ParquetFile
from yaml import YAMLError

from src.registry.registry import ActiveModelRef, RegistryError, read_active


@dataclass(frozen=True)
class DashboardPaths:
    """Local artifact locations consumed by the dashboard, relative to a project root."""

    root: Path
    account: Path
    heartbeat: Path
    lock: Path
    equity: Path
    trades: Path
    regimes: Path
    predictions: Path
    scorecard: Path
    walkforward: Path
    deployments: Path
    registry_active: Path
    registry_history: Path
    drift: Path
    pipeline_runs: Path
    backtests: Path
    lock_timeout_seconds: int = 900

    @classmethod
    def from_root(cls, root: str | Path) -> DashboardPaths:
        project_root = Path(root).resolve()
        return cls(
            root=project_root,
            account=project_root / "data/live_sim/account_state.json",
            heartbeat=project_root / "data/live_sim/heartbeat.json",
            lock=project_root / "data/live_sim/live_sim.lock",
            equity=project_root / "data/live_sim/equity_curve.parquet",
            trades=project_root / "data/live_sim/trades.parquet",
            regimes=project_root / "data/regimes/latest.parquet",
            predictions=project_root / "data/predictions/latest.parquet",
            scorecard=project_root / "data/scorecards/latest.json",
            walkforward=project_root / "data/walkforward/latest.parquet",
            deployments=project_root / "data/deployments/events.parquet",
            registry_active=project_root / "registry/active_model.yaml",
            registry_history=project_root / "registry/history.parquet",
            drift=project_root / "artifacts/drift/latest.json",
            pipeline_runs=project_root / "artifacts/pipeline_runs",
            backtests=project_root / "artifacts/backtest",
        )


@dataclass(frozen=True)
class StateFreshness:
    label: str
    updated_at: str | None
    detail: str


@dataclass
class DashboardSnapshot:
    """Data prepared for presentation without any dashboard-framework dependency."""

    freshness: StateFreshness
    overview: dict[str, Any]
    account: dict[str, Any] | None
    active_model: ActiveModelRef | None
    scorecard: dict[str, Any] | None
    drift: dict[str, Any] | None
    equity: pd.DataFrame | None
    trades: pd.DataFrame | None
    regimes: pd.DataFrame | None
    predictions: pd.DataFrame | None
    walkforward: pd.DataFrame | None
    deployments: pd.DataFrame | None
    registry_history: pd.DataFrame | None
    backtest: pd.DataFrame | None
    pipeline_summary: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path, label: str, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Could not use {label}: expected a JSON object.")
        return None
    return payload


def _read_parquet(
    path: Path,
    label: str,
    warnings: list[str],
    *,
    columns: list[str] | None = None,
    max_rows: int = 5_000,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        parquet = ParquetFile(path)
        selected_columns = (
            [column for column in columns if column in parquet.schema.names]
            if columns is not None
            else None
        )
        tail: pd.DataFrame | None = None
        for batch in parquet.iter_batches(batch_size=max_rows, columns=selected_columns):
            batch_frame = batch.to_pandas()
            tail = (
                batch_frame.tail(max_rows)
                if tail is None
                else pd.concat([tail, batch_frame], ignore_index=True).tail(max_rows)
            )
        return tail.reset_index(drop=True) if tail is not None else pd.DataFrame()
    except Exception as exc:  # Optional local artifacts can be incomplete or corrupted.
        warnings.append(f"Could not read {label}: {exc}")
        return None


def _as_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return datetime.fromisoformat(str(parsed.isoformat())).astimezone(UTC)


def _format_timestamp(value: Any) -> str | None:
    parsed = _as_utc(value)
    return parsed.isoformat() if parsed is not None else None


def _freshness(
    heartbeat: dict[str, Any] | None,
    *,
    lock_exists: bool,
    timeout_seconds: int,
    now: datetime,
    has_persisted_state: bool,
    has_historical_data: bool,
) -> StateFreshness:
    updated_at = _format_timestamp(heartbeat.get("updated_at_utc")) if heartbeat else None
    if updated_at is not None:
        heartbeat_time = _as_utc(updated_at)
        age = (now - heartbeat_time).total_seconds() if heartbeat_time is not None else None
        if lock_exists and age is not None and 0 <= age <= timeout_seconds:
            return StateFreshness(
                "LIVE", updated_at, "Fresh local heartbeat and simulator lock observed."
            )
        if age is not None and age > timeout_seconds:
            return StateFreshness(
                "STALE", updated_at, "Persisted heartbeat is older than the live timeout."
            )
        return StateFreshness(
            "LATEST KNOWN", updated_at, "Persisted heartbeat is not live-loop evidence."
        )
    if has_persisted_state:
        return StateFreshness(
            "LATEST KNOWN", None, "No current heartbeat; displaying durable local artifacts."
        )
    if has_historical_data:
        return StateFreshness(
            "HISTORICAL",
            None,
            "No latest runtime state; displaying persisted evaluation, registry, or backtest artifacts.",
        )
    return StateFreshness("UNAVAILABLE", None, "No persisted runtime state has been generated yet.")


def _latest_row(
    frame: pd.DataFrame | None, timestamp_column: str = "timestamp"
) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    if timestamp_column not in frame.columns:
        return frame.iloc[-1]
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if timestamps.notna().any():
        return frame.loc[timestamps.idxmax()]
    return frame.iloc[-1]


def _latest_backtest(directory: Path, warnings: list[str]) -> pd.DataFrame | None:
    candidates = sorted(directory.glob("results_*.parquet"))
    if not candidates:
        return None
    return _read_parquet(candidates[-1], "latest historical backtest", warnings)


def _latest_pipeline_summary(directory: Path, warnings: list[str]) -> dict[str, Any] | None:
    candidates = sorted(directory.glob("pipeline_run*.json"))
    if not candidates:
        return None
    return _read_json(candidates[-1], "latest pipeline summary", warnings)


def _read_active_model(path: Path, warnings: list[str]) -> ActiveModelRef | None:
    if not path.exists():
        return None
    try:
        return read_active(path)
    except (RegistryError, OSError, ValueError, YAMLError) as exc:
        warnings.append(f"Could not read active model registry: {exc}")
        return None


def _overview(
    account: dict[str, Any] | None,
    heartbeat: dict[str, Any] | None,
    regimes: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    active_model: ActiveModelRef | None,
) -> dict[str, Any]:
    regime_row = _latest_row(regimes)
    prediction_row = _latest_row(predictions)
    active_prediction = None
    if predictions is not None and "is_active" in predictions.columns:
        active_rows = predictions.loc[predictions["is_active"].fillna(False).astype(bool)]
        active_prediction = _latest_row(active_rows)
    selected = active_prediction if active_prediction is not None else prediction_row
    return {
        "regime": (heartbeat or {}).get("regime")
        or (regime_row.get("regime") if regime_row is not None else None),
        "model": (heartbeat or {}).get("selected_model_id")
        or (selected.get("active_model_id") if selected is not None else None)
        or (selected.get("model_name") if selected is not None else None)
        or (active_model.model_id if active_model is not None else None),
        "signal": (heartbeat or {}).get("signal")
        or (selected.get("signal") if selected is not None else None),
        "prediction": (heartbeat or {}).get("prediction")
        if heartbeat is not None and heartbeat.get("prediction") is not None
        else (selected.get("y_pred") if selected is not None else None),
        "portfolio_value": (account or {}).get("portfolio_value"),
        "cash": (account or {}).get("cash"),
        "position": (account or {}).get("position"),
        "market_timestamp": (heartbeat or {}).get("bar_timestamp_utc")
        or (regime_row.get("timestamp") if regime_row is not None else None),
    }


def load_dashboard_snapshot(
    root: str | Path = ".",
    *,
    now: datetime | None = None,
    paths: DashboardPaths | None = None,
) -> DashboardSnapshot:
    """Load local artifacts only; this function never starts or mutates a system process."""
    dashboard_paths = paths or DashboardPaths.from_root(root)
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    warnings: list[str] = []
    account = _read_json(dashboard_paths.account, "live account state", warnings)
    heartbeat = _read_json(dashboard_paths.heartbeat, "live heartbeat", warnings)
    equity = _read_parquet(
        dashboard_paths.equity,
        "equity history",
        warnings,
        columns=[
            "timestamp",
            "cash",
            "position",
            "last_price",
            "portfolio_value",
            "realized_pnl",
            "unrealized_pnl",
            "total_pnl",
            "regime",
            "active_model_id",
            "prediction",
            "signal",
            "action_taken",
            "reason",
        ],
    )
    trades = _read_parquet(dashboard_paths.trades, "trade history", warnings)
    regimes = _read_parquet(
        dashboard_paths.regimes,
        "regime history",
        warnings,
        columns=["timestamp", "regime", "regime_explanation"],
    )
    predictions = _read_parquet(
        dashboard_paths.predictions,
        "prediction history",
        warnings,
        columns=["row_id", "model_name", "active_model_id", "y_pred", "is_active", "signal"],
    )
    scorecard = _read_json(dashboard_paths.scorecard, "latest scorecard", warnings)
    walkforward = _read_parquet(dashboard_paths.walkforward, "walk-forward metrics", warnings)
    deployments = _read_parquet(dashboard_paths.deployments, "deployment history", warnings)
    registry_history = _read_parquet(dashboard_paths.registry_history, "registry history", warnings)
    drift = _read_json(dashboard_paths.drift, "drift snapshot", warnings)
    active_model = _read_active_model(dashboard_paths.registry_active, warnings)
    backtest = _latest_backtest(dashboard_paths.backtests, warnings)
    pipeline_summary = _latest_pipeline_summary(dashboard_paths.pipeline_runs, warnings)
    has_state = any(item is not None for item in (account, equity, trades, regimes, predictions))
    has_historical_data = any(
        item is not None
        for item in (
            active_model,
            scorecard,
            walkforward,
            deployments,
            registry_history,
            drift,
            backtest,
            pipeline_summary,
        )
    )
    freshness = _freshness(
        heartbeat,
        lock_exists=dashboard_paths.lock.exists(),
        timeout_seconds=dashboard_paths.lock_timeout_seconds,
        now=current_time.astimezone(UTC),
        has_persisted_state=has_state,
        has_historical_data=has_historical_data,
    )
    return DashboardSnapshot(
        freshness=freshness,
        overview=_overview(account, heartbeat, regimes, predictions, active_model),
        account=account,
        active_model=active_model,
        scorecard=scorecard,
        drift=drift,
        equity=equity,
        trades=trades,
        regimes=regimes,
        predictions=predictions,
        walkforward=walkforward,
        deployments=deployments,
        registry_history=registry_history,
        backtest=backtest,
        pipeline_summary=pipeline_summary,
        warnings=warnings,
    )

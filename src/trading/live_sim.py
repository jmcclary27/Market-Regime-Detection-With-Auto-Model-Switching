from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config.load_config import load_config
from src.features.run_features import run as run_features
from src.inference.batch_predict import run_stage as run_predictions
from src.ingestion.fetch_market_data import fetch_market_data
from src.ingestion.quality import audit_raw_bars, default_audit_output
from src.regimes.run_regime_detection import run as run_regimes
from src.trading.execution import ExecutionConfig, execute_signal, log_trade
from src.trading.live_sim_shared import (
    ActivePrediction,
    is_live_eligible_prediction,
    load_loop_state,
    save_loop_state,
)
from src.trading.signals import SignalConfig, prediction_to_signal
from src.trading.state import load_account_state, log_account_snapshot, save_account_state

LOG = logging.getLogger("live_sim")


@dataclass(frozen=True)
class LiveSimPaths:
    raw_dir: Path = Path("data/raw/live_sim")
    runs_dir: Path = Path("data/runs")
    predictions_dir: Path = Path("data/predictions")
    state_path: Path = Path("data/live_sim/account_state.json")
    loop_state_path: Path = Path("data/live_sim/live_loop_state.json")
    lock_path: Path = Path("data/live_sim/live_sim.lock")
    heartbeat_path: Path = Path("data/live_sim/heartbeat.json")
    trades_path: Path = Path("data/live_sim/trades.parquet")
    equity_path: Path = Path("data/live_sim/equity_curve.parquet")


@dataclass(frozen=True)
class LiveSimConfig:
    symbols: list[str]
    interval: str = "5m"
    poll_sleep_seconds: int = 60
    lookback_days: int = 60
    candle_close_buffer_seconds: int = 60
    stale_bar_tolerance_seconds: int = 600
    market_timezone: str = "America/New_York"
    market_open: str = "09:30"
    market_close: str = "16:00"
    regular_hours_only: bool = True
    starting_cash: float = 100_000.0
    dry_run: bool = True
    lock_timeout_seconds: int = 900
    provider_failure_backoff_seconds: int = 300
    paths: LiveSimPaths = field(default_factory=LiveSimPaths)

    @classmethod
    def from_settings(cls, config_path: Path | None = None) -> LiveSimConfig:
        cfg = load_config(config_path)
        market = cfg.get("market", {})
        live = cfg.get("live_sim", {})
        paths_cfg = live.get("paths", {})

        symbols = [str(s) for s in market.get("symbols", [])]
        if len(symbols) < 2:
            raise ValueError("live_sim requires at least two configured market.symbols")

        paths = LiveSimPaths(
            raw_dir=Path(paths_cfg.get("raw_dir", "data/raw/live_sim")),
            runs_dir=Path(paths_cfg.get("runs_dir", "data/runs")),
            predictions_dir=Path(paths_cfg.get("predictions_dir", "data/predictions")),
            state_path=Path(paths_cfg.get("state_path", "data/live_sim/account_state.json")),
            loop_state_path=Path(
                paths_cfg.get("loop_state_path", "data/live_sim/live_loop_state.json")
            ),
            lock_path=Path(paths_cfg.get("lock_path", "data/live_sim/live_sim.lock")),
            heartbeat_path=Path(
                paths_cfg.get("heartbeat_path", "data/live_sim/heartbeat.json")
            ),
            trades_path=Path(paths_cfg.get("trades_path", "data/live_sim/trades.parquet")),
            equity_path=Path(paths_cfg.get("equity_path", "data/live_sim/equity_curve.parquet")),
        )

        return cls(
            symbols=symbols,
            interval=str(live.get("interval", "5m")),
            poll_sleep_seconds=int(live.get("poll_sleep_seconds", 60)),
            lookback_days=int(live.get("lookback_days", 60)),
            candle_close_buffer_seconds=int(live.get("candle_close_buffer_seconds", 60)),
            stale_bar_tolerance_seconds=int(live.get("stale_bar_tolerance_seconds", 600)),
            market_timezone=str(live.get("market_timezone", "America/New_York")),
            market_open=str(live.get("market_open", "09:30")),
            market_close=str(live.get("market_close", "16:00")),
            regular_hours_only=bool(live.get("regular_hours_only", True)),
            starting_cash=float(live.get("starting_cash", 100_000.0)),
            dry_run=bool(live.get("dry_run", True)),
            lock_timeout_seconds=int(live.get("lock_timeout_seconds", 900)),
            provider_failure_backoff_seconds=int(
                live.get("provider_failure_backoff_seconds", 300)
            ),
            paths=paths,
        )


@dataclass(frozen=True)
class MarketBar:
    timestamp_utc: pd.Timestamp
    open_price: float
    close_price: float


@dataclass(frozen=True)
class CycleArtifacts:
    raw_path: Path
    features_path: Path
    regimes_path: Path
    predictions_path: Path


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")


def utc_iso(ts: pd.Timestamp | datetime) -> str:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    return str(out.tz_convert("UTC").isoformat())


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    return out.tz_convert("UTC")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_heartbeat(
    path: Path,
    *,
    status: str,
    message: str,
    run_ts: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "run_ts": run_ts,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def is_market_open(now: datetime, cfg: LiveSimConfig) -> bool:
    local = now.astimezone(ZoneInfo(cfg.market_timezone))
    if cfg.regular_hours_only and local.weekday() >= 5:
        return False
    open_h, open_m = _parse_hhmm(cfg.market_open)
    close_h, close_m = _parse_hhmm(cfg.market_close)
    market_open = local.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = local.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return market_open <= local < market_close


def market_close_for_signal(signal_ts: pd.Timestamp | datetime, cfg: LiveSimConfig) -> pd.Timestamp:
    local_signal = to_utc_timestamp(signal_ts).tz_convert(cfg.market_timezone)
    close_h, close_m = _parse_hhmm(cfg.market_close)
    market_close = local_signal.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return market_close.tz_convert("UTC")


def _should_queue_pending_signal(
    signal: str,
    *,
    signal_ts: pd.Timestamp | datetime,
    expected_fill_ts: pd.Timestamp | datetime,
    cfg: LiveSimConfig,
) -> bool:
    if signal.upper() == "HOLD":
        return False
    return bool(to_utc_timestamp(expected_fill_ts) < market_close_for_signal(signal_ts, cfg))


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported live_sim interval: {interval}")


def _effective_yfinance_lookback_days(interval: str, requested_days: int) -> int:
    if requested_days < 1:
        raise ValueError("live_sim lookback_days must be >= 1")

    unit = interval[-1].lower()
    if unit == "m":
        # Yahoo minute bars are limited to a 60-day request span. Since this
        # poll uses tomorrow as the exclusive end date to include today's bars,
        # a configured 60-day lookback would otherwise become a 61-day request.
        return min(requested_days, 59)
    return requested_days


def _normalize_raw_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        if out.index.name in ("Date", "Datetime", "timestamp"):
            out = out.reset_index().rename(columns={out.index.name: "timestamp"})
        elif "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "timestamp"})
        elif "Date" in out.columns:
            out = out.rename(columns={"Date": "timestamp"})
        else:
            out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "timestamp" not in out.columns:
        raise ValueError(f"Fetched data for {symbol} does not contain a timestamp column")
    if "symbol" not in out.columns:
        out["symbol"] = symbol

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].fillna(symbol).astype(str)
    out = out.dropna(subset=["timestamp"])
    return out


def poll_intraday_bars(cfg: LiveSimConfig, *, run_ts: str, now: datetime) -> Path:
    lookback_days = _effective_yfinance_lookback_days(cfg.interval, cfg.lookback_days)
    start_date = (now.date() - timedelta(days=lookback_days)).isoformat()
    end_date = (now.date() + timedelta(days=1)).isoformat()

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for symbol in cfg.symbols:
        try:
            df = fetch_market_data(
                symbol,
                start_date,
                end_date,
                interval=cfg.interval,
                auto_adjust=False,
            )
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
            continue
        frames.append(_normalize_raw_frame(df, symbol))

    if not frames:
        raise ValueError(f"No market data frames were fetched. failures={failures}")

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)

    symbol_tag = "-".join(cfg.symbols)
    output_path = cfg.paths.raw_dir / f"{symbol_tag}_{cfg.interval}_{run_ts}.parquet"
    latest_path = cfg.paths.raw_dir / "latest.parquet"
    atomic_write_parquet(bars, output_path)
    atomic_write_parquet(bars, latest_path)

    run_record = {
        "run_type": "live_sim_intraday_poll",
        "run_ts": run_ts,
        "symbols": cfg.symbols,
        "interval": cfg.interval,
        "start_date": start_date,
        "end_date": end_date,
        "output_path": str(output_path),
        "latest_path": str(latest_path),
        "rows": int(len(bars)),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "provider_failure_count": int(len(failures)),
        "provider_attempt_count": int(len(cfg.symbols)),
    }
    audit_path = default_audit_output(Path.cwd(), run_ts)
    audit_raw_bars(
        raw_path=output_path,
        run_ts=run_ts,
        finished_at_utc=str(run_record["finished_at_utc"]),
        symbols=list(cfg.symbols),
        interval=cfg.interval,
        provider_failure_count=int(len(failures)),
        provider_attempt_count=int(len(cfg.symbols)),
        run_type="live_sim_intraday_poll",
        output_path=audit_path,
    )
    run_record["data_quality_path"] = str(audit_path)
    atomic_write_json(cfg.paths.runs_dir / f"live_poll_{run_ts}.json", run_record)
    return output_path


def latest_closed_bar(
    raw: pd.DataFrame,
    cfg: LiveSimConfig,
    *,
    now: datetime,
    symbol: str | None = None,
) -> MarketBar | None:
    symbol = symbol or cfg.symbols[0]
    required = {"timestamp", "symbol", "open", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw bars missing required columns: {sorted(missing)}")

    data = raw[raw["symbol"].astype(str) == symbol].copy()
    if data.empty:
        raise ValueError(f"No raw bars found for traded symbol: {symbol}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["timestamp", "open", "close"])
    if data.empty:
        raise ValueError(f"No finite raw open/close bars found for traded symbol: {symbol}")

    interval = interval_to_timedelta(cfg.interval)
    cutoff = to_utc_timestamp(now) - interval - pd.Timedelta(
        seconds=cfg.candle_close_buffer_seconds
    )
    closed = data[data["timestamp"] <= cutoff].sort_values("timestamp", kind="mergesort")
    if closed.empty:
        return None

    row = closed.iloc[-1]
    open_price = float(row["open"])
    close_price = float(row["close"])
    if not np.isfinite(open_price) or not np.isfinite(close_price):
        raise ValueError("Latest closed bar contains non-finite open/close")

    return MarketBar(
        timestamp_utc=pd.Timestamp(row["timestamp"]).tz_convert("UTC"),
        open_price=open_price,
        close_price=close_price,
    )


def validate_bar_freshness(bar: MarketBar, cfg: LiveSimConfig, *, now: datetime) -> None:
    interval = interval_to_timedelta(cfg.interval)
    expected_close = bar.timestamp_utc + interval
    age = to_utc_timestamp(now) - expected_close
    max_age = pd.Timedelta(seconds=cfg.stale_bar_tolerance_seconds)
    if age > max_age:
        raise ValueError(
            f"Latest closed bar is stale: bar={bar.timestamp_utc.isoformat()} age={age}"
        )


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_regime_label(regime: str) -> str | None:
    normalized = regime.lower().strip()
    if normalized in {"bullish", "bearish", "sideways"}:
        return normalized
    return None


def _regime_for_bar(regimes: pd.DataFrame, *, bar_timestamp_utc: str) -> tuple[int, str]:
    if "timestamp" not in regimes.columns:
        raise ValueError("Regimes artifact missing required column: timestamp")

    regimes = regimes.copy()
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True, errors="coerce")
    target_ts = to_utc_timestamp(bar_timestamp_utc)
    matches = regimes[regimes["timestamp"] == target_ts]
    if matches.empty:
        raise ValueError(f"No regime/features row found for bar timestamp {bar_timestamp_utc}")

    regime_col = _first_existing_column(
        regimes,
        ["regime", "regime_label", "detected_regime", "active_regime"],
    )
    if regime_col is None:
        raise ValueError(f"Regimes artifact has no regime column: {list(regimes.columns)}")

    regime_value = matches.iloc[-1][regime_col]
    if pd.isna(regime_value):
        raise ValueError(f"Regime is missing for bar timestamp {bar_timestamp_utc}")

    return int(matches.index[-1]), str(regime_value)


def active_prediction_for_bar(
    predictions_path: Path,
    regimes_path: Path,
    *,
    bar_timestamp_utc: str,
) -> tuple[ActivePrediction, str]:
    preds = pd.read_parquet(predictions_path)
    regimes = pd.read_parquet(regimes_path)

    required_pred = {"row_id", "y_pred", "is_active"}
    missing_pred = required_pred - set(preds.columns)
    if missing_pred:
        raise ValueError(f"Predictions missing required columns: {sorted(missing_pred)}")

    row_id, regime = _regime_for_bar(regimes, bar_timestamp_utc=bar_timestamp_utc)
    active = preds[preds["is_active"].astype(bool)].copy()
    active = active[active["row_id"].astype(int) == row_id]
    if active.empty:
        raise ValueError(f"No active prediction found for row_id={row_id}")

    active = active.sort_values(["row_id"], kind="mergesort")
    row = active.iloc[-1]
    prediction = float(row["y_pred"])
    if not np.isfinite(prediction):
        raise ValueError(f"Active prediction is not finite: {prediction!r}")

    actual_model_id = str(row.get("active_model_id", row.get("model_name", "unknown")))
    actual_model_name = (
        actual_model_id if str(row.get("model_name", "active")) == "active" else str(row["model_name"])
    )

    return (
        ActivePrediction(
            row_id=row_id,
            prediction=prediction,
            active_model_id=actual_model_id,
            model_name=actual_model_name,
            active_model_type=(
                str(row["active_model_type"]) if pd.notna(row.get("active_model_type")) else None
            ),
            model_path=str(row["model_path"]) if pd.notna(row.get("model_path")) else None,
        ),
        regime,
    )


def regime_matched_lightgbm_prediction_for_bar(
    predictions_path: Path,
    regimes_path: Path,
    *,
    bar_timestamp_utc: str,
) -> tuple[ActivePrediction | None, str, str | None]:
    preds = pd.read_parquet(predictions_path)
    regimes = pd.read_parquet(regimes_path)

    required_pred = {"row_id", "model_name", "y_pred"}
    missing_pred = required_pred - set(preds.columns)
    if missing_pred:
        raise ValueError(f"Predictions missing required columns: {sorted(missing_pred)}")

    row_id, regime = _regime_for_bar(regimes, bar_timestamp_utc=bar_timestamp_utc)
    normalized_regime = normalize_regime_label(regime)
    if normalized_regime is None:
        return None, regime, f"unsupported regime for LightGBM expert selection: {regime}"

    model_name = f"expert_lightgbm_{normalized_regime}"
    matched = preds[preds["row_id"].astype(int) == row_id].copy()
    matched = matched[matched["model_name"].astype(str) == model_name]
    if matched.empty:
        return None, regime, f"missing regime-matched expert prediction: {model_name}"

    matched = matched.sort_values(["row_id"], kind="mergesort")
    row = matched.iloc[-1]
    prediction = float(row["y_pred"])
    if not np.isfinite(prediction):
        return None, regime, f"non-finite regime-matched expert prediction: {model_name}"

    return (
        ActivePrediction(
            row_id=row_id,
            prediction=prediction,
            active_model_id=model_name,
            model_name=model_name,
        ),
        regime,
        None,
    )


def _pending_expired(
    pending: dict[str, Any],
    current_bar: MarketBar,
    cfg: LiveSimConfig,
) -> bool:
    if str(pending.get("signal", "")).upper() == "HOLD":
        return True

    expected = pending.get("expected_fill_bar_timestamp_utc")
    if not expected:
        return True
    expected_ts = to_utc_timestamp(str(expected))
    if current_bar.timestamp_utc != expected_ts:
        return True

    signal_ts = to_utc_timestamp(str(pending.get("signal_bar_timestamp_utc")))
    market_close = market_close_for_signal(signal_ts, cfg)
    return bool(expected_ts >= market_close)


def execute_pending_signal(
    pending: dict[str, Any],
    bar: MarketBar,
    cfg: LiveSimConfig,
    *,
    timestamp: str,
) -> dict[str, Any]:
    state = load_account_state(cfg.paths.state_path, starting_cash=cfg.starting_cash)
    signal = str(pending["signal"])

    prediction = float(cast(float | int | str, pending.get("prediction", 0.0)))

    trade = execute_signal(
        state=state,
        signal=signal,
        price=bar.open_price,
        timestamp=timestamp,
        config=ExecutionConfig(),
    )
    trade.update(
        {
            "fill_bar_timestamp": utc_iso(bar.timestamp_utc),
            "signal_bar_timestamp": pending.get("signal_bar_timestamp_utc"),
            "regime": pending.get("regime"),
            "active_model_id": pending.get("active_model_id"),
            "prediction": prediction,
            "fill_policy": "next_open",
        }
    )

    log_trade(cfg.paths.trades_path, trade)
    log_account_snapshot(
        path=cfg.paths.equity_path,
        state=state,
        timestamp=timestamp,
        price=bar.open_price,
        regime=str(pending.get("regime")),
        active_model_id=str(pending.get("active_model_id")),
        prediction=prediction,
        signal=signal,
        action_taken=trade["action"],
        reason=f"filled pending signal at next open: {trade['reason']}",
    )
    save_account_state(state, cfg.paths.state_path)
    return cast(dict[str, Any], trade)


def mark_decision_snapshot(
    cfg: LiveSimConfig,
    *,
    timestamp: str,
    price: float,
    regime: str,
    active_model_id: str,
    prediction: float,
    signal: str,
    reason: str,
) -> None:
    state = load_account_state(cfg.paths.state_path, starting_cash=cfg.starting_cash)
    state.mark_to_market(price)
    log_account_snapshot(
        path=cfg.paths.equity_path,
        state=state,
        timestamp=timestamp,
        price=price,
        regime=regime,
        active_model_id=active_model_id,
        prediction=prediction,
        signal=signal,
        action_taken="NONE",
        reason=reason,
    )
    save_account_state(state, cfg.paths.state_path)


def build_cycle_artifacts(cfg: LiveSimConfig, *, run_ts: str, now: datetime) -> CycleArtifacts:
    raw_path = poll_intraday_bars(cfg, run_ts=run_ts, now=now)
    features_path, _manifest_path = run_features(input_path=raw_path, timestamp=run_ts)
    regimes_path = run_regimes(input_path=features_path, timestamp=run_ts)
    predictions_path = run_predictions(
        features_path=regimes_path,
        output_dir=cfg.paths.predictions_dir,
        runs_dir=cfg.paths.runs_dir,
    )
    return CycleArtifacts(
        raw_path=raw_path,
        features_path=features_path,
        regimes_path=regimes_path,
        predictions_path=predictions_path,
    )


def run_once(cfg: LiveSimConfig, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    run_ts = utc_timestamp()

    if not is_market_open(now, cfg):
        extra: dict[str, Any] = {}
        loop_state = load_loop_state(cfg.paths.loop_state_path)
        pending = loop_state.get("pending_signal")
        if isinstance(pending, dict):
            loop_state["canceled_pending_signal"] = pending
            loop_state["pending_signal"] = None
            save_loop_state(cfg.paths.loop_state_path, loop_state)
            extra["canceled_pending_signal"] = pending

        write_heartbeat(
            cfg.paths.heartbeat_path,
            status="idle",
            message="market is closed",
            run_ts=run_ts,
            extra=extra,
        )
        return {"status": "idle", "reason": "market_closed"}

    artifacts = build_cycle_artifacts(cfg, run_ts=run_ts, now=now)
    raw = pd.read_parquet(artifacts.raw_path)
    bar = latest_closed_bar(raw, cfg, now=now)
    if bar is None:
        write_heartbeat(
            cfg.paths.heartbeat_path,
            status="warning",
            message="no closed candle available",
            run_ts=run_ts,
        )
        return {"status": "skipped", "reason": "no_closed_candle"}

    validate_bar_freshness(bar, cfg, now=now)
    bar_ts = utc_iso(bar.timestamp_utc)
    loop_state = load_loop_state(cfg.paths.loop_state_path)

    if "last_processed_bar_timestamp_utc" not in loop_state:
        loop_state["last_processed_bar_timestamp_utc"] = bar_ts
        loop_state["initialized_at_utc"] = datetime.now(UTC).isoformat()
        save_loop_state(cfg.paths.loop_state_path, loop_state)
        write_heartbeat(
            cfg.paths.heartbeat_path,
            status="initialized",
            message="initialized to latest closed candle without trading",
            run_ts=run_ts,
            extra={"bar_timestamp_utc": bar_ts},
        )
        return {"status": "initialized", "bar_timestamp_utc": bar_ts}

    if loop_state.get("last_processed_bar_timestamp_utc") == bar_ts:
        write_heartbeat(
            cfg.paths.heartbeat_path,
            status="idle",
            message="latest closed candle already processed",
            run_ts=run_ts,
            extra={"bar_timestamp_utc": bar_ts},
        )
        return {"status": "idle", "reason": "already_processed", "bar_timestamp_utc": bar_ts}

    timestamp = datetime.now(UTC).isoformat()
    pending = loop_state.get("pending_signal")
    trade: dict[str, Any] | None = None

    if isinstance(pending, dict):
        if _pending_expired(pending, bar, cfg):
            loop_state["expired_pending_signal"] = pending
            loop_state["pending_signal"] = None
        else:
            trade = execute_pending_signal(pending, bar, cfg, timestamp=timestamp)
            loop_state["last_filled_signal"] = pending
            loop_state["pending_signal"] = None

    _, regime = _regime_for_bar(pd.read_parquet(artifacts.regimes_path), bar_timestamp_utc=bar_ts)
    active: ActivePrediction | None = None
    selection_error: str | None = None
    try:
        active, _active_regime = active_prediction_for_bar(
            artifacts.predictions_path,
            artifacts.regimes_path,
            bar_timestamp_utc=bar_ts,
        )
        if _active_regime != regime:
            regime = _active_regime
    except ValueError as exc:
        selection_error = str(exc)

    expected_fill_ts = bar.timestamp_utc + interval_to_timedelta(cfg.interval)
    pending_signal: dict[str, Any] | None = None
    selected_model_id = active.active_model_id if active is not None else "unavailable"
    selected_model_name = active.model_name if active is not None else "unavailable"
    selected_prediction = active.prediction if active is not None else float("nan")

    if active is None:
        signal = "HOLD"
        decision_reason = selection_error or "missing active prediction"
    elif not is_live_eligible_prediction(active):
        signal = "HOLD"
        decision_reason = f"active model not live-eligible: {active.active_model_id}"
    else:
        signal = prediction_to_signal(
            active.prediction,
            regime=regime,
            config=SignalConfig(),
        )
        decision_reason = "created pending signal for next candle open"

    if signal == "HOLD":
        loop_state["pending_signal"] = None
        if (
            active is not None
            and decision_reason == "created pending signal for next candle open"
        ):
            decision_reason = "hold signal recorded without pending fill"
    elif not _should_queue_pending_signal(
        signal,
        signal_ts=bar.timestamp_utc,
        expected_fill_ts=expected_fill_ts,
        cfg=cfg,
    ):
        loop_state["pending_signal"] = None
        decision_reason = "signal not queued because expected fill is at or after market close"
    else:
        assert active is not None
        pending_signal = {
            "signal_bar_timestamp_utc": bar_ts,
            "expected_fill_bar_timestamp_utc": utc_iso(expected_fill_ts),
            "signal": signal,
            "prediction": selected_prediction,
            "regime": regime,
            "active_model_id": selected_model_id,
            "model_name": selected_model_name,
            "row_id": active.row_id,
            "created_at_utc": timestamp,
        }
        loop_state["pending_signal"] = pending_signal

    loop_state["last_processed_bar_timestamp_utc"] = bar_ts

    mark_decision_snapshot(
        cfg,
        timestamp=timestamp,
        price=bar.close_price,
        regime=regime,
        active_model_id=selected_model_id,
        prediction=selected_prediction,
        signal=signal,
        reason=decision_reason,
    )
    save_loop_state(cfg.paths.loop_state_path, loop_state)

    write_heartbeat(
        cfg.paths.heartbeat_path,
        status="ok",
        message="processed live simulation candle",
        run_ts=run_ts,
        extra={
            "bar_timestamp_utc": bar_ts,
            "regime": regime,
            "selected_model_id": selected_model_id,
            "selected_model_name": selected_model_name,
            "prediction": selected_prediction,
            "signal": signal,
            "decision_reason": decision_reason,
            "pending_signal": pending_signal,
            "trade": trade,
            "artifacts": {k: str(v) for k, v in asdict(artifacts).items()},
        },
    )
    return {
        "status": "ok",
        "bar_timestamp_utc": bar_ts,
        "regime": regime,
        "selected_model_id": selected_model_id,
        "prediction": selected_prediction,
        "signal": signal,
        "decision_reason": decision_reason,
        "pending_signal": pending_signal,
        "trade": trade,
    }


class LiveSimLock:
    def __init__(self, cfg: LiveSimConfig) -> None:
        self.cfg = cfg
        self.acquired = False

    def __enter__(self) -> LiveSimLock:
        path = self.cfg.paths.lock_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            heartbeat = self.cfg.paths.heartbeat_path
            stale = True
            if heartbeat.exists():
                data = json.loads(heartbeat.read_text(encoding="utf-8"))
                updated = to_utc_timestamp(str(data.get("updated_at_utc")))
                age = pd.Timestamp.now(tz="UTC") - updated
                stale = age.total_seconds() > self.cfg.lock_timeout_seconds
            if not stale:
                raise RuntimeError(f"live_sim lock is active: {path}")

        payload = {
            "pid": os.getpid(),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "heartbeat_path": str(self.cfg.paths.heartbeat_path),
        }
        atomic_write_json(path, payload)
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.acquired and self.cfg.paths.lock_path.exists():
            self.cfg.paths.lock_path.unlink()


def run_loop(cfg: LiveSimConfig, *, once: bool = False) -> None:
    provider_failures = 0
    with LiveSimLock(cfg):
        while True:
            try:
                result = run_once(cfg)
                provider_failures = 0
                LOG.info("live_sim cycle result: %s", result)
            except Exception as exc:
                provider_failures += 1
                LOG.exception("live_sim cycle failed")
                write_heartbeat(
                    cfg.paths.heartbeat_path,
                    status="error",
                    message=str(exc),
                    extra={"provider_failures": provider_failures},
                )

            if once:
                return

            sleep_seconds = cfg.poll_sleep_seconds
            if provider_failures >= 3:
                sleep_seconds = max(sleep_seconds, cfg.provider_failure_backoff_seconds)
            time.sleep(sleep_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intraday live paper-trading simulation.")
    parser.add_argument("--config", type=Path, default=None, help="Path to settings.yaml")
    parser.add_argument("--once", action="store_true", help="Run one live-sim cycle and exit")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = LiveSimConfig.from_settings(args.config)
    run_loop(cfg, once=bool(args.once))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

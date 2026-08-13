from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.trading.live_sim import (
    CycleArtifacts,
    LiveSimConfig,
    LiveSimLock,
    LiveSimPaths,
    MarketBar,
    _pending_expired,
    _should_queue_pending_signal,
    active_prediction_for_bar,
    atomic_write_json,
    is_market_open,
    latest_closed_bar,
    load_loop_state,
    poll_intraday_bars,
    run_once,
    save_loop_state,
)
from src.trading.state import AccountState, load_account_state, save_account_state


def _cfg(tmp_path: Path) -> LiveSimConfig:
    return LiveSimConfig(
        symbols=["SPY", "QQQ"],
        paths=LiveSimPaths(
            raw_dir=tmp_path / "raw",
            runs_dir=tmp_path / "runs",
            predictions_dir=tmp_path / "predictions",
            state_path=tmp_path / "live" / "account_state.json",
            loop_state_path=tmp_path / "live" / "loop_state.json",
            lock_path=tmp_path / "live" / "live_sim.lock",
            heartbeat_path=tmp_path / "live" / "heartbeat.json",
            trades_path=tmp_path / "live" / "trades.parquet",
            equity_path=tmp_path / "live" / "equity.parquet",
        ),
    )


def test_market_hours_gate_regular_session(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    assert is_market_open(datetime(2026, 5, 18, 14, 0, tzinfo=UTC), cfg)
    assert not is_market_open(datetime(2026, 5, 18, 12, 0, tzinfo=UTC), cfg)
    assert not is_market_open(datetime(2026, 5, 18, 21, 0, tzinfo=UTC), cfg)
    assert not is_market_open(datetime(2026, 5, 23, 14, 0, tzinfo=UTC), cfg)


def test_latest_closed_bar_uses_buffer(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-18T14:00:00Z",
                "2026-05-18T14:05:00Z",
                "2026-05-18T14:10:00Z",
            ],
            "symbol": ["SPY", "SPY", "SPY"],
            "open": [100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5],
        }
    )

    bar = latest_closed_bar(raw, cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert bar is not None
    assert bar.timestamp_utc.isoformat() == "2026-05-18T14:05:00+00:00"
    assert bar.open_price == 101.0
    assert bar.close_price == 101.5


def test_active_prediction_for_latest_bar_validates_active_row(tmp_path: Path) -> None:
    regimes = pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["sideways", "bullish"],
        }
    )
    preds = pd.DataFrame(
        {
            "row_id": [1, 1],
            "model_name": ["active", "shadow"],
            "y_pred": [0.002, -0.1],
            "is_active": [True, False],
            "active_model_id": ["expert_lightgbm_bullish", "expert_lightgbm_bullish"],
        }
    )
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"
    regimes.to_parquet(regimes_path, index=False)
    preds.to_parquet(preds_path, index=False)

    active, regime = active_prediction_for_bar(
        preds_path,
        regimes_path,
        bar_timestamp_utc="2026-05-18T14:05:00+00:00",
    )

    assert active.row_id == 1
    assert active.prediction == 0.002
    assert active.active_model_id == "expert_lightgbm_bullish"
    assert active.model_name == "expert_lightgbm_bullish"
    assert regime == "bullish"


def test_active_prediction_missing_active_row_fails(tmp_path: Path) -> None:
    regimes = pd.DataFrame({"timestamp": ["2026-05-18T14:05:00Z"], "regime": ["bullish"]})
    preds = pd.DataFrame({"row_id": [0], "y_pred": [0.002], "is_active": [False]})
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"
    regimes.to_parquet(regimes_path, index=False)
    preds.to_parquet(preds_path, index=False)

    with pytest.raises(ValueError, match="No active prediction"):
        active_prediction_for_bar(
            preds_path,
            regimes_path,
            bar_timestamp_utc="2026-05-18T14:05:00+00:00",
        )


def test_poll_intraday_bars_clamps_minute_lookback_for_yahoo_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    calls: list[tuple[str, str, str, str]] = []

    def fake_fetch_market_data(
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        calls.append((symbol, start_date, end_date, interval))
        return pd.DataFrame(
            {
                "timestamp": ["2026-05-28T14:00:00Z"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

    monkeypatch.setattr("src.trading.live_sim.fetch_market_data", fake_fetch_market_data)

    poll_intraday_bars(
        cfg,
        run_ts="20260528_190000Z",
        now=datetime(2026, 5, 28, 19, 0, tzinfo=UTC),
    )

    assert calls == [
        ("SPY", "2026-03-30", "2026-05-29", "5m"),
        ("QQQ", "2026-03-30", "2026-05-29", "5m"),
    ]


def test_first_start_initializes_without_trade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:05:00Z"],
            "symbol": ["SPY"],
            "open": [101.0],
            "close": [101.5],
        }
    )
    raw.to_parquet(raw_path, index=False)

    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(
            raw_path, tmp_path / "f", tmp_path / "r", tmp_path / "p"
        ),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "initialized"
    assert not cfg.paths.trades_path.exists()
    state = load_loop_state(cfg.paths.loop_state_path)
    assert state["last_processed_bar_timestamp_utc"] == "2026-05-18T14:05:00+00:00"


def test_pending_signal_fills_then_creates_next_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"

    raw = pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
        }
    )
    regimes = pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["bullish", "bullish"],
        }
    )
    preds = pd.DataFrame(
        {
            "row_id": [1, 1],
            "model_name": ["active", "expert_lightgbm_bullish"],
            "y_pred": [0.003, -0.1],
            "is_active": [True, False],
            "active_model_id": ["expert_lightgbm_bullish", "expert_lightgbm_bullish"],
        }
    )
    raw.to_parquet(raw_path, index=False)
    regimes.to_parquet(regimes_path, index=False)
    preds.to_parquet(preds_path, index=False)

    save_loop_state(
        cfg.paths.loop_state_path,
        {
            "last_processed_bar_timestamp_utc": "2026-05-18T14:00:00+00:00",
            "pending_signal": {
                "signal_bar_timestamp_utc": "2026-05-18T14:00:00+00:00",
                "expected_fill_bar_timestamp_utc": "2026-05-18T14:05:00+00:00",
                "signal": "BUY",
                "prediction": 0.002,
                "regime": "bullish",
                "active_model_id": "old_model",
            },
        },
    )
    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(raw_path, tmp_path / "f", regimes_path, preds_path),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "ok"
    assert result["trade"]["action"] == "BUY"
    assert result["trade"]["price"] == 101.0
    assert result["selected_model_id"] == "expert_lightgbm_bullish"
    state = load_loop_state(cfg.paths.loop_state_path)
    assert state["pending_signal"]["signal_bar_timestamp_utc"] == "2026-05-18T14:05:00+00:00"
    assert state["pending_signal"]["expected_fill_bar_timestamp_utc"] == "2026-05-18T14:10:00+00:00"
    assert state["pending_signal"]["active_model_id"] == "expert_lightgbm_bullish"


def test_live_sim_uses_active_prediction_instead_of_regime_matched_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"

    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(raw_path, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["sideways", "bearish"],
        }
    ).to_parquet(regimes_path, index=False)
    pd.DataFrame(
        {
            "row_id": [1, 1, 1, 1],
            "model_name": [
                "active",
                "expert_lightgbm_bullish",
                "expert_lightgbm_bearish",
                "expert_lightgbm_sideways",
            ],
            "y_pred": [0.002, -0.004, -0.002, 0.001],
            "is_active": [True, False, False, False],
            "active_model_id": [
                "expert_lightgbm_bullish",
                "expert_lightgbm_bullish",
                "expert_lightgbm_bullish",
                "expert_lightgbm_bullish",
            ],
        }
    ).to_parquet(preds_path, index=False)

    save_loop_state(
        cfg.paths.loop_state_path,
        {"last_processed_bar_timestamp_utc": "2026-05-18T14:00:00+00:00"},
    )
    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(raw_path, tmp_path / "f", regimes_path, preds_path),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "ok"
    assert result["selected_model_id"] == "expert_lightgbm_bullish"
    assert result["signal"] == "HOLD"
    assert "hold signal recorded without pending fill" == result["decision_reason"]
    state = load_loop_state(cfg.paths.loop_state_path)
    assert state["pending_signal"] is None


def test_hold_signal_does_not_create_pending_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"

    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(raw_path, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["bullish", "bullish"],
        }
    ).to_parquet(regimes_path, index=False)
    pd.DataFrame(
        {
            "row_id": [1, 1],
            "model_name": ["active", "expert_lightgbm_bullish"],
            "y_pred": [0.0, 0.003],
            "is_active": [True, False],
            "active_model_id": ["expert_lightgbm_bullish", "expert_lightgbm_bullish"],
        }
    ).to_parquet(preds_path, index=False)

    save_loop_state(
        cfg.paths.loop_state_path,
        {"last_processed_bar_timestamp_utc": "2026-05-18T14:00:00+00:00"},
    )
    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(raw_path, tmp_path / "f", regimes_path, preds_path),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "ok"
    assert result["pending_signal"] is None
    state = load_loop_state(cfg.paths.loop_state_path)
    assert state["pending_signal"] is None


def test_missing_active_prediction_records_hold_without_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"

    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(raw_path, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["sideways", "bullish"],
        }
    ).to_parquet(regimes_path, index=False)
    pd.DataFrame(
        {
            "row_id": [1],
            "model_name": ["expert_lightgbm_sideways"],
            "y_pred": [0.003],
            "is_active": [False],
            "active_model_id": ["expert_lightgbm_sideways"],
        }
    ).to_parquet(preds_path, index=False)

    save_loop_state(
        cfg.paths.loop_state_path,
        {"last_processed_bar_timestamp_utc": "2026-05-18T14:00:00+00:00"},
    )
    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(raw_path, tmp_path / "f", regimes_path, preds_path),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "ok"
    assert result["signal"] == "HOLD"
    assert result["pending_signal"] is None
    assert result["selected_model_id"] == "unavailable"
    assert "No active prediction found" in result["decision_reason"]
    loop_state = load_loop_state(cfg.paths.loop_state_path)
    assert loop_state["pending_signal"] is None
    heartbeat = json.loads(cfg.paths.heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["signal"] == "HOLD"
    assert "No active prediction found" in heartbeat["decision_reason"]


def test_validated_arima_active_model_is_live_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw_path = tmp_path / "raw.parquet"
    regimes_path = tmp_path / "regimes.parquet"
    preds_path = tmp_path / "preds.parquet"

    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(raw_path, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-05-18T14:00:00Z", "2026-05-18T14:05:00Z"],
            "regime": ["bullish", "bullish"],
        }
    ).to_parquet(regimes_path, index=False)
    pd.DataFrame(
        {
            "row_id": [1, 1],
            "model_name": ["active", "expert_lightgbm_bullish"],
            "y_pred": [0.003, 0.003],
            "is_active": [True, False],
            "active_model_id": ["expert_arima_bullish", "expert_arima_bullish"],
            "model_path": [
                "models/experts/bullish/latest.arima.json",
                "models/experts/bullish/latest.arima.json",
            ],
        }
    ).to_parquet(preds_path, index=False)

    save_loop_state(
        cfg.paths.loop_state_path,
        {"last_processed_bar_timestamp_utc": "2026-05-18T14:00:00+00:00"},
    )
    monkeypatch.setattr(
        "src.trading.live_sim.build_cycle_artifacts",
        lambda cfg, run_ts, now: CycleArtifacts(raw_path, tmp_path / "f", regimes_path, preds_path),
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 14, 11, 0, tzinfo=UTC))

    assert result["status"] == "ok"
    assert result["signal"] == "BUY"
    assert result["pending_signal"] is not None
    assert result["selected_model_id"] == "expert_arima_bullish"
    assert result["decision_reason"] == "created pending signal for next candle open"


def test_close_time_signal_is_not_queued(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    signal_ts = pd.Timestamp("2026-05-18T19:55:00Z")
    expected_fill_ts = pd.Timestamp("2026-05-18T20:00:00Z")

    assert not _should_queue_pending_signal(
        "BUY",
        signal_ts=signal_ts,
        expected_fill_ts=expected_fill_ts,
        cfg=cfg,
    )


def test_market_closed_clears_pending_signal_without_changing_account(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    save_loop_state(
        cfg.paths.loop_state_path,
        {
            "last_processed_bar_timestamp_utc": "2026-05-18T19:50:00+00:00",
            "pending_signal": {
                "signal_bar_timestamp_utc": "2026-05-18T19:50:00+00:00",
                "expected_fill_bar_timestamp_utc": "2026-05-18T19:55:00+00:00",
                "signal": "BUY",
            },
        },
    )
    save_account_state(
        AccountState(cash=500.0, position=2.0, cost_basis=200.0, last_price=100.0),
        cfg.paths.state_path,
    )

    result = run_once(cfg, now=datetime(2026, 5, 18, 21, 0, 0, tzinfo=UTC))

    assert result == {"status": "idle", "reason": "market_closed"}
    loop_state = load_loop_state(cfg.paths.loop_state_path)
    assert loop_state["pending_signal"] is None
    assert loop_state["canceled_pending_signal"]["signal"] == "BUY"
    account = load_account_state(cfg.paths.state_path, starting_cash=cfg.starting_cash)
    assert account.cash == pytest.approx(500.0)
    assert account.position == pytest.approx(2.0)
    assert account.cost_basis == pytest.approx(200.0)


def test_pending_hold_expires_without_trade(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    pending = {
        "signal_bar_timestamp_utc": "2026-05-18T14:00:00+00:00",
        "expected_fill_bar_timestamp_utc": "2026-05-18T14:05:00+00:00",
        "signal": "HOLD",
    }
    bar = MarketBar(
        timestamp_utc=pd.Timestamp("2026-05-18T14:05:00Z"),
        open_price=101.0,
        close_price=101.5,
    )

    assert _pending_expired(pending, bar, cfg)


def test_atomic_write_json_replaces_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"b": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"b": 2}


def test_file_lock_rejects_active_lock(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    atomic_write_json(cfg.paths.heartbeat_path, {"updated_at_utc": datetime.now(UTC).isoformat()})
    atomic_write_json(cfg.paths.lock_path, {"pid": 123})

    with pytest.raises(RuntimeError, match="lock is active"):
        with LiveSimLock(cfg):
            pass

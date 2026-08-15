"""Deterministic three-portfolio daily experiment state machine."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.experiment.manifest import FrozenExperimentManifest
from src.experiment.trading import TargetExposureConfig, prediction_to_target, rebalance_to_target
from src.trading.state import AccountState

PORTFOLIOS = ("buy_and_hold", "static_ml", "regime_ml")


def initial_state(manifest: FrozenExperimentManifest) -> dict[str, Any]:
    return {
        "last_bar_timestamp_utc": None,
        "accounts": {
            name: asdict(
                AccountState(cash=manifest.starting_cash, portfolio_value=manifest.starting_cash)
            )
            for name in PORTFOLIOS
        },
        "pending_targets": {},
    }


def _account(raw: dict[str, Any]) -> AccountState:
    return AccountState(**raw)


def _model_prediction(predictions: pd.DataFrame, *, row_id: int, model_id: str) -> float:
    rows = predictions[(predictions["row_id"] == row_id) & (predictions["model_name"] == model_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one prediction for frozen model {model_id!r}")
    value = float(rows.iloc[0]["y_pred"])
    if not pd.notna(value):
        raise ValueError(f"Frozen model {model_id!r} prediction is not finite")
    return value


def process_daily_bar(
    *,
    state: dict[str, Any],
    manifest: FrozenExperimentManifest,
    predictions: pd.DataFrame,
    bar_timestamp_utc: str,
    row_id: int,
    regime: str,
    regime_confidence: float | None,
    open_price: float,
    close_price: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply prior targets at open, then calculate and queue close-derived targets."""
    if (
        state.get("last_bar_timestamp_utc") is not None
        and bar_timestamp_utc <= state["last_bar_timestamp_utc"]
    ):
        raise ValueError("bar timestamp must be strictly newer than experiment state")
    if regime == "neutral":
        regime = "sideways"
    if regime not in manifest.regime_models:
        raise ValueError(f"No frozen model exists for regime {regime!r}")
    accounts = {name: _account(state["accounts"][name]) for name in PORTFOLIOS}
    fills: dict[str, dict[str, Any] | None] = {name: None for name in PORTFOLIOS}
    cfg = TargetExposureConfig(fee_bps=manifest.fee_bps, slippage_bps=manifest.slippage_bps)
    for name, pending in state.get("pending_targets", {}).items():
        fill = rebalance_to_target(
            state=accounts[name],
            target_exposure=float(pending["target_exposure"]),
            open_price=open_price,
            config=cfg,
        )
        if fill is not None:
            fill["signal_bar_timestamp_utc"] = pending["signal_bar_timestamp_utc"]
        fills[name] = fill

    static_prediction = _model_prediction(
        predictions, row_id=row_id, model_id=manifest.static_model.model_id
    )
    regime_ref = manifest.regime_models[regime]
    regime_prediction = _model_prediction(predictions, row_id=row_id, model_id=regime_ref.model_id)
    low, high = manifest.exposure_thresholds
    targets = {
        "buy_and_hold": 1.0,
        "static_ml": prediction_to_target(static_prediction, lower=low, upper=high),
        "regime_ml": prediction_to_target(regime_prediction, lower=low, upper=high),
    }
    for account in accounts.values():
        account.mark_to_market(close_price)
    pending_targets = {
        name: {"target_exposure": target, "signal_bar_timestamp_utc": bar_timestamp_utc}
        for name, target in targets.items()
    }
    new_state = {
        "last_bar_timestamp_utc": bar_timestamp_utc,
        "accounts": {name: asdict(account) for name, account in accounts.items()},
        "pending_targets": pending_targets,
    }
    event = {
        "bar_timestamp_utc": bar_timestamp_utc,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "open_price": open_price,
        "close_price": close_price,
        "static_model_id": manifest.static_model.model_id,
        "static_model_version": manifest.static_model.version,
        "static_prediction": static_prediction,
        "regime_model_id": regime_ref.model_id,
        "regime_model_version": regime_ref.version,
        "regime_prediction": regime_prediction,
        "targets": targets,
        "fills": fills,
        "portfolio_values": {name: account.portfolio_value for name, account in accounts.items()},
    }
    return new_state, event

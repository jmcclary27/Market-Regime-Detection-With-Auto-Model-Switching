from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = 0.001
    sell_threshold: float = -0.001
    use_regime_filter: bool = True


def prediction_to_signal(
    prediction: float,
    *,
    regime: str | None = None,
    config: SignalConfig | None = None,
) -> Signal:
    """
    Convert a model prediction into a trading signal.

    Assumes prediction is expected future return.
    Example:
      0.002  -> BUY
     -0.002  -> SELL
      0.0001 -> HOLD
    """
    config = config or SignalConfig()
    pred = float(prediction)

    signal: Signal

    if pred >= config.buy_threshold:
        signal = "BUY"
    elif pred <= config.sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    if config.use_regime_filter and regime is not None:
        signal = apply_regime_filter(signal, regime)

    return signal


def apply_regime_filter(signal: Signal, regime: str) -> Signal:
    """
    Simple safety filter based on detected market regime.

    You can tune this later.
    """
    regime = regime.lower().strip()

    if regime == "bullish":
        return signal

    if regime == "sideways":
        if signal == "SELL":
            return "HOLD"
        return signal

    if regime == "bearish":
        if signal == "BUY":
            return "HOLD"
        return signal

    return signal

from src.trading.signals import SignalConfig, prediction_to_signal


def test_prediction_to_buy_signal():
    assert prediction_to_signal(0.002, config=SignalConfig(use_regime_filter=False)) == "BUY"


def test_prediction_to_sell_signal():
    assert prediction_to_signal(-0.002, config=SignalConfig(use_regime_filter=False)) == "SELL"


def test_prediction_to_hold_signal():
    assert prediction_to_signal(0.0001, config=SignalConfig(use_regime_filter=False)) == "HOLD"


def test_bearish_regime_blocks_buy():
    assert prediction_to_signal(0.002, regime="bearish") == "HOLD"


def test_sideways_regime_blocks_sell():
    assert prediction_to_signal(-0.002, regime="sideways") == "HOLD"


def test_bullish_regime_allows_buy():
    assert prediction_to_signal(0.002, regime="bullish") == "BUY"
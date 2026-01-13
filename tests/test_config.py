from src.config.load_config import load_config


def test_load_config_has_market_symbols():
    cfg = load_config()
    assert "market" in cfg
    assert "symbols" in cfg["market"]
    assert isinstance(cfg["market"]["symbols"], list)
    assert len(cfg["market"]["symbols"]) > 0

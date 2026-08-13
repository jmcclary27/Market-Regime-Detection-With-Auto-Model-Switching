from __future__ import annotations

import pandas as pd

from src.demo.run import DEMO_SYMBOLS, build_demo_bars


def test_demo_fixture_is_deterministic_and_complete() -> None:
    first = build_demo_bars()
    second = build_demo_bars()

    pd.testing.assert_frame_equal(first, second)
    assert set(first["symbol"]) == set(DEMO_SYMBOLS)
    assert first.groupby("symbol").size().nunique() == 1
    assert (first["close"] > 0).all()

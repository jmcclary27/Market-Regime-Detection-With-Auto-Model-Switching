"""Create a deterministic, offline recruiter demonstration run."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_TIMESTAMP = "20240131_160000Z"
DEMO_SYMBOLS = ("SPY", "QQQ")


def build_demo_bars(rows: int = 420) -> pd.DataFrame:
    """Return deterministic, autocorrelated synthetic daily bars for two symbols."""
    if rows < 250:
        raise ValueError("rows must be at least 250 to train the demo baseline")

    dates = pd.date_range("2022-01-03", periods=rows, freq="B", tz="UTC")
    frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(DEMO_SYMBOLS):
        steps = np.arange(rows, dtype=float)
        innovations = 0.0007 * np.sin(steps / (11.0 + index))
        returns = 0.0012 * np.sin(steps / (5.0 + index)) + innovations
        returns[1:] += 0.45 * returns[:-1]
        close = (100.0 + index * 50.0) * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "close": close,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "volume": 1_000_000 + index * 100_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run() -> int:
    """Run the canonical pipeline against only deterministic synthetic inputs."""
    # Import lazily: the pipeline imports ``build_demo_bars`` for its offline poll step.
    from src.pipeline.run import main as pipeline_main

    return pipeline_main(["--offline", "--run-ts", DEMO_TIMESTAMP])


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

# src/monitoring/metrics.py
from __future__ import annotations

# Run-level regime diagnostics metrics, logged to MLflow
REGIME_ENTROPY = "regime_entropy"
AVG_REGIME_DURATION = "avg_regime_duration"
SWITCHES_PER_1000_STEPS = "switches_per_1000_steps"

# Per-regime breakdown metrics, logged to MLflow as keys with suffixes
# Example, pct_time_regime_0, pct_time_regime_1, ...
PCT_TIME_REGIME_PREFIX = "pct_time_regime_"

# Optional, but useful if you want more visibility later
AVG_CONFIDENCE = "avg_regime_confidence"
MIN_CONFIDENCE = "min_regime_confidence"

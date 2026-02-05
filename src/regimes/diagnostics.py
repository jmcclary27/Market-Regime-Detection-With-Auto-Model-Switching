# src/regimes/diagnostics.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RegimeConfidenceStats:
    mean: float
    p10: float
    p50: float
    p90: float
    min: float
    max: float


@dataclass(frozen=True)
class RegimeDiagnostics:
    run_ts: str
    n_steps: int
    n_regimes: int

    # Counts and rates
    n_switches: int
    switches_per_1000_steps: float

    # Distribution of time spent in each regime
    pct_time_regime: list[float]  # length == n_regimes

    # Durations of consecutive runs of the same regime
    avg_regime_duration: float
    regime_durations: dict[str, list[int]]  # keys are regime ids as strings

    # Transitions
    transition_counts: list[list[int]]  # shape (K, K)
    transition_probs: list[list[float]]  # shape (K, K)

    # Uncertainty
    regime_entropy: float
    confidence: RegimeConfidenceStats | None = None

    # Helpful debug metadata, optional
    model_version: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def compute_switches(labels: np.ndarray) -> int:
    if labels.size <= 1:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))


def compute_pct_time(labels: np.ndarray, n_regimes: int) -> list[float]:
    counts = np.bincount(labels.astype(int), minlength=n_regimes)
    total = max(int(labels.size), 1)
    return [float(c) / float(total) for c in counts.tolist()]


def compute_durations(labels: np.ndarray) -> dict[str, list[int]]:
    if labels.size == 0:
        return {}

    out: dict[str, list[int]] = {}
    current = int(labels[0])
    run_len = 1

    for v in labels[1:]:
        v_int = int(v)
        if v_int == current:
            run_len += 1
        else:
            out.setdefault(str(current), []).append(run_len)
            current = v_int
            run_len = 1

    out.setdefault(str(current), []).append(run_len)
    return out


def compute_transition_counts(labels: NDArray[np.integer[Any]], n_regimes: int) -> NDArray[np.int_]:
    counts: NDArray[np.int_] = np.zeros((n_regimes, n_regimes), dtype=int)
    if labels.size <= 1:
        return counts

    a: NDArray[np.int_] = labels[:-1].astype(int)
    b: NDArray[np.int_] = labels[1:].astype(int)

    for i, j in zip(a, b, strict=False):
        ii = int(i)
        jj = int(j)
        if 0 <= ii < n_regimes and 0 <= jj < n_regimes:
            counts[ii, jj] += 1
    return counts


def normalize_rows(counts: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    probs: NDArray[np.float64] = counts.astype(float)
    row_sums: NDArray[np.float64] = probs.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(
            probs,
            row_sums,
            out=np.zeros_like(probs),
            where=row_sums != 0.0,
        )
    return probs


def compute_entropy(p: NDArray[np.floating[Any]], eps: float = 1e-12) -> float:
    """
    Shannon entropy over regime occupancy distribution.
    p should sum to ~1. Returns entropy in nats.
    """
    p_arr: NDArray[np.float64] = np.asarray(p, dtype=float)
    p_clip: NDArray[np.float64] = np.clip(p_arr, eps, 1.0)
    denom = float(p_clip.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")

    p_norm: NDArray[np.float64] = p_clip / denom
    s = float(np.sum(p_norm * np.log(p_norm)))
    ent = -s
    return _safe_float(ent)


def confidence_stats(conf: np.ndarray) -> RegimeConfidenceStats:
    conf = conf.astype(float)
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        return RegimeConfidenceStats(
            mean=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            min=float("nan"),
            max=float("nan"),
        )

    q10, q50, q90 = np.quantile(conf, [0.10, 0.50, 0.90])
    return RegimeConfidenceStats(
        mean=_safe_float(float(np.mean(conf))),
        p10=_safe_float(float(q10)),
        p50=_safe_float(float(q50)),
        p90=_safe_float(float(q90)),
        min=_safe_float(float(np.min(conf))),
        max=_safe_float(float(np.max(conf))),
    )

# tests/test_regime_diagnostics.py
from __future__ import annotations

import numpy as np

from src.regimes.diagnostics import (
    compute_durations,
    compute_entropy,
    compute_pct_time,
    compute_switches,
    compute_transition_counts,
    normalize_rows,
)


def test_regime_diagnostics_components_stable() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 2], dtype=int)
    k = 3

    n_switches = compute_switches(labels)
    assert n_switches == 3  # 0->1, 1->0, 0->2

    pct = compute_pct_time(labels, n_regimes=k)
    assert len(pct) == k
    assert abs(sum(pct) - 1.0) < 1e-9

    ent = compute_entropy(np.array(pct, dtype=float))
    assert np.isfinite(ent)
    assert ent >= 0.0

    durs = compute_durations(labels)
    assert set(durs.keys()) <= {"0", "1", "2"}
    assert durs["0"] == [2, 1]
    assert durs["1"] == [3]
    assert durs["2"] == [4]

    tc = compute_transition_counts(labels, n_regimes=k)
    assert tc.shape == (k, k)
    # check a couple transitions
    assert tc[0, 0] == 1  # 0->0 once
    assert tc[0, 1] == 1  # 0->1 once
    assert tc[1, 1] == 2  # 1->1 twice

    tp = normalize_rows(tc)
    assert tp.shape == (k, k)
    # rows with transitions should sum to 1
    row_sums = tp.sum(axis=1)
    assert np.isclose(row_sums[0], 1.0)
    assert np.isclose(row_sums[1], 1.0)
    assert np.isclose(row_sums[2], 1.0) or np.isclose(
        row_sums[2], 0.0
    )  # depends on whether last regime has outgoing transitions

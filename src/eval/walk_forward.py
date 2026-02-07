from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """
    One walk-forward split defined by integer position ranges on an index.

    Ranges are half-open: [start, end)
    """

    split_id: str
    train: range
    val: range
    test: range

    def train_index(self, index: pd.Index) -> pd.Index:
        return index[self.train.start : self.train.stop]

    def val_index(self, index: pd.Index) -> pd.Index:
        return index[self.val.start : self.val.stop]

    def test_index(self, index: pd.Index) -> pd.Index:
        return index[self.test.start : self.test.stop]


def _as_index(idx: pd.Index | Sequence[object] | Iterable[object]) -> pd.Index:
    if isinstance(idx, pd.Index):
        return idx
    return pd.Index(list(idx))


def walk_forward_splits(
    index: pd.Index | Sequence[object] | Iterable[object],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    anchored: bool = True,
) -> list[WalkForwardSplit]:
    """
    Deterministic walk-forward splitter using bar counts (integer sizes).

    anchored=True:
      train window expands from 0..train_end
    anchored=False:
      train window rolls with fixed size

    Example:
      train_size=100, val_size=20, test_size=20, step_size=20
    """
    if train_size <= 0 or val_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("All sizes must be positive integers")

    idx = _as_index(index)
    n = len(idx)

    block = train_size + val_size + test_size
    if n < block:
        return []

    splits: list[WalkForwardSplit] = []

    # The first split uses:
    # train: [0, train_size)
    # val:   [train_size, train_size+val_size)
    # test:  [train_size+val_size, train_size+val_size+test_size)
    start_val = train_size
    start_test = train_size + val_size

    split_num = 0
    while start_test + test_size <= n:
        split_num += 1

        if anchored:
            train_start = 0
            train_stop = start_val
        else:
            train_stop = start_val
            train_start = max(0, train_stop - train_size)

        val_start = start_val
        val_stop = val_start + val_size

        test_start = start_test
        test_stop = test_start + test_size

        # Defensive sanity, though loop condition should ensure this.
        if not (
            0 <= train_start < train_stop <= val_start < val_stop <= test_start < test_stop <= n
        ):
            raise RuntimeError("Invalid split construction, please report")

        splits.append(
            WalkForwardSplit(
                split_id=f"wf_{split_num:04d}",
                train=range(train_start, train_stop),
                val=range(val_start, val_stop),
                test=range(test_start, test_stop),
            )
        )

        # Advance by step_size: shift val/test forward
        start_val += step_size
        start_test += step_size

    return splits

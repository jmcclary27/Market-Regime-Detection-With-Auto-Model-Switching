from __future__ import annotations

import pandas as pd

from src.eval.walk_forward import walk_forward_splits


def _assert_ordered_non_overlapping(train: range, val: range, test: range) -> None:
    assert train.stop <= val.start
    assert val.stop <= test.start

    # Within-split, ranges must be non-empty
    assert train.stop > train.start
    assert val.stop > val.start
    assert test.stop > test.start


def test_walk_forward_splits_are_deterministic_and_time_ordered_anchored() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")

    splits1 = walk_forward_splits(
        idx,
        train_size=100,
        val_size=20,
        test_size=20,
        step_size=20,
        anchored=True,
    )
    splits2 = walk_forward_splits(
        idx,
        train_size=100,
        val_size=20,
        test_size=20,
        step_size=20,
        anchored=True,
    )

    # Deterministic
    assert splits1 == splits2
    assert len(splits1) > 0

    for s in splits1:
        _assert_ordered_non_overlapping(s.train, s.val, s.test)

        train_idx = idx[s.train.start : s.train.stop]
        val_idx = idx[s.val.start : s.val.stop]
        test_idx = idx[s.test.start : s.test.stop]

        # Time ordered, no overlap in timestamps
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()


def test_walk_forward_splits_roll_train_when_not_anchored() -> None:
    idx = pd.date_range("2020-01-01", periods=160, freq="D", tz="UTC")

    splits = walk_forward_splits(
        idx,
        train_size=60,
        val_size=20,
        test_size=20,
        step_size=20,
        anchored=False,
    )

    assert len(splits) > 0

    # First split has fixed-size train by construction here
    first = splits[0]
    assert (first.train.stop - first.train.start) == 60

    # Later splits should still have fixed-size train (except potentially very early edge cases)
    for s in splits[1:]:
        _assert_ordered_non_overlapping(s.train, s.val, s.test)
        assert (s.train.stop - s.train.start) == 60


def test_walk_forward_returns_empty_when_not_enough_data() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")

    splits = walk_forward_splits(
        idx,
        train_size=6,
        val_size=3,
        test_size=3,
        step_size=3,
        anchored=True,
    )

    assert splits == []

# tests/test_model_contracts.py
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_COLS = [
    "row_id",
    "model_name",
    "model_source",
    "y_pred",
    "inference_ts",
    "features_path",
    "model_path",
    "is_active",
    "active_model_type",
    "active_model_id",
    "active_model_version",
    "active_regime",
]

ALLOWED_MODEL_SOURCES = {"baseline", "expert", "pretrained"}
ALLOWED_ACTIVE_TYPES = {"baseline", "expert", "pretrained"}


def _load_predictions(path: str = "data/predictions/latest.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def test_predictions_schema_exact() -> None:
    df = _load_predictions()
    assert list(df.columns) == EXPECTED_COLS, (
        "prediction schema drift detected.\n"
        f"Expected columns:\n{EXPECTED_COLS}\n"
        f"Actual columns:\n{list(df.columns)}"
    )


def test_predictions_basic_integrity() -> None:
    df = _load_predictions()
    assert len(df) > 0, "predictions frame is empty"

    # row_id integrity
    assert df["row_id"].notna().all(), "row_id contains nulls"

    # model_name integrity
    assert df["model_name"].notna().all(), "model_name contains nulls"
    assert (df["model_name"].astype(str).str.strip() != "").all(), "model_name contains blanks"

    # model_source values
    bad_sources = sorted(set(df["model_source"].astype(str)) - ALLOWED_MODEL_SOURCES)
    assert not bad_sources, f"unexpected model_source values: {bad_sources}"

    # y_pred numeric + finite
    assert pd.api.types.is_numeric_dtype(df["y_pred"]), "y_pred must be numeric dtype"
    y = df["y_pred"].to_numpy(dtype=float)
    assert np.isfinite(y).all(), "y_pred contains NaN or inf"


def test_one_prediction_per_row_id_per_model() -> None:
    df = _load_predictions()

    dup = df.duplicated(subset=["row_id", "model_name"], keep=False)
    assert not dup.any(), (
        "duplicate predictions found for same (row_id, model_name). "
        f"Examples:\n{df.loc[dup, ['row_id', 'model_name']].head(20)}"
    )


def test_equal_row_id_coverage_across_models() -> None:
    df = _load_predictions()

    models = sorted(df["model_name"].unique().tolist())
    assert len(models) >= 1

    ids_by_model: dict[str, set[int]] = {}
    for m in models:
        ids_by_model[m] = set(df.loc[df["model_name"] == m, "row_id"].tolist())

    base = models[0]
    base_ids = ids_by_model[base]

    for m in models[1:]:
        if ids_by_model[m] != base_ids:
            only_in_m = sorted(list(ids_by_model[m] - base_ids))[:10]
            only_in_base = sorted(list(base_ids - ids_by_model[m]))[:10]
            raise AssertionError(
                "row_id coverage mismatch across models.\n"
                f"baseline_model={base}\n"
                f"other_model={m}\n"
                f"only_in_other(sample)={only_in_m}\n"
                f"only_in_baseline(sample)={only_in_base}"
            )


def test_run_level_metadata_constant_within_file() -> None:
    df = _load_predictions()

    # latest.parquet should represent one inference run
    assert df["inference_ts"].nunique(dropna=False) == 1, "inference_ts must be constant per file"
    assert df["features_path"].nunique(dropna=False) == 1, "features_path must be constant per file"


def test_model_path_constant_per_model() -> None:
    df = _load_predictions()

    # A model_name should map to a single artifact path in the run
    counts = df.groupby("model_name", sort=False)["model_path"].nunique(dropna=False)
    bad = counts[counts != 1]
    assert len(bad) == 0, (
        f"model_path must be constant per model_name.\nViolations:\n{bad.to_string()}"
    )


def test_predictions_not_all_constant_per_model() -> None:
    df = _load_predictions()

    # Optional sanity check: predictions for a model should not be all identical
    # (Allows weird edge cases, but catches obvious bugs like always-0)
    for m, g in df.groupby("model_name", sort=False):
        y = g["y_pred"].to_numpy(dtype=float)
        # If there is only 1 unique value, that is very suspicious.
        if np.unique(y).size == 1:
            raise AssertionError(f"model_name={m} has constant y_pred across all rows")


def test_active_pointer_consistency() -> None:
    df = _load_predictions()

    active = df[df["is_active"] == True]  # noqa: E712

    # Allow no active rows if your pipeline didn't annotate them for this run
    if len(active) == 0:
        return

    # At most 1 active model per row_id
    per_row_counts = active.groupby("row_id", sort=False).size()
    bad_rows = per_row_counts[per_row_counts > 1]
    assert len(bad_rows) == 0, (
        f"multiple active models found for row_id(s): {bad_rows.index.tolist()[:10]}"
    )

    # active_model_type must be filled + valid
    assert active["active_model_type"].notna().all(), (
        "active_model_type must be non-null for active rows"
    )
    bad_types = sorted(
        set(active["active_model_type"].astype(str).str.lower()) - ALLOWED_ACTIVE_TYPES
    )
    assert not bad_types, f"unexpected active_model_type values: {bad_types}"

    # active_model_id/version must be filled
    assert active["active_model_id"].notna().all(), (
        "active_model_id must be non-null for active rows"
    )
    assert active["active_model_version"].notna().all(), (
        "active_model_version must be non-null for active rows"
    )

    # active_regime rules: required only if expert
    is_expert = active["active_model_type"].astype(str).str.lower() == "expert"
    if is_expert.any():
        assert active.loc[is_expert, "active_regime"].notna().all(), (
            "active_regime must be set for expert"
        )
    if (~is_expert).any():
        assert active.loc[~is_expert, "active_regime"].isna().all(), (
            "active_regime must be null if not expert"
        )

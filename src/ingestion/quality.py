from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported raw bars format: {path.suffix}")


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]

    if "timestamp" not in out.columns:
        for candidate in ("datetime", "date", "unnamed:_0", "index"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "timestamp"})
                break

    if "timestamp" not in out.columns and out.index.name in ("Date", "Datetime", "timestamp"):
        out = out.reset_index().rename(columns={out.index.name: "timestamp"})
        out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]

    if "timestamp" not in out.columns:
        raise ValueError("Raw bars are missing a timestamp column")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()

    if "symbol" not in out.columns:
        out["symbol"] = "unknown"
    out["symbol"] = out["symbol"].fillna("unknown").astype(str)
    return out


def _interval_to_timedelta(interval: str | None) -> pd.Timedelta | None:
    if interval in (None, ""):
        return None
    try:
        return pd.to_timedelta(str(interval))
    except ValueError:
        pass

    text = str(interval).strip().lower()
    if text.endswith("m"):
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("h"):
        return pd.Timedelta(hours=int(text[:-1]))
    if text.endswith("d"):
        return pd.Timedelta(days=int(text[:-1]))
    return None


def _business_day_missing_count(timestamps: pd.Series) -> int:
    if timestamps.empty:
        return 0
    dates = pd.to_datetime(timestamps.dt.normalize().drop_duplicates().sort_values(), utc=True)
    if dates.empty:
        return 0
    missing = 0
    for idx in range(1, len(dates)):
        expected = pd.bdate_range(dates.iloc[idx - 1], dates.iloc[idx], inclusive="both")
        missing += max(len(expected) - 2, 0)
    return int(missing)


def _intraday_missing_count(timestamps: pd.Series, base_interval: pd.Timedelta) -> int:
    if timestamps.empty or base_interval <= pd.Timedelta(0):
        return 0

    sorted_ts = timestamps.drop_duplicates().sort_values()
    if sorted_ts.empty:
        return 0

    missing = 0
    session_gap = max(pd.Timedelta(hours=8), base_interval * 24)
    for idx in range(1, len(sorted_ts)):
        prev_ts = pd.Timestamp(sorted_ts.iloc[idx - 1])
        cur_ts = pd.Timestamp(sorted_ts.iloc[idx])
        gap = cur_ts - prev_ts
        if gap <= base_interval:
            continue
        if gap >= session_gap:
            continue
        missing += max(int(round(gap / base_interval)) - 1, 0)
    return int(missing)


def _latest_staleness_seconds(
    bars: pd.DataFrame,
    *,
    finished_at: pd.Timestamp,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, group in bars.groupby("symbol", sort=True):
        latest = group["timestamp"].max()
        out[str(symbol)] = max((finished_at - latest).total_seconds(), 0.0)
    return out


def _status_from_metrics(
    *,
    row_count: int,
    duplicate_rate: float,
    missing_rate: float,
    late_data_rate: float,
    provider_failure_count: int,
) -> str:
    if row_count == 0:
        return "error"
    if (
        provider_failure_count > 0
        or duplicate_rate > 0.0
        or missing_rate > 0.0
        or late_data_rate > 0.0
    ):
        return "warning"
    return "ok"


def audit_raw_bars(
    *,
    raw_path: Path,
    run_ts: str,
    finished_at_utc: str | None = None,
    symbols: list[str] | None = None,
    interval: str | None = None,
    provider_failure_count: int = 0,
    provider_attempt_count: int | None = None,
    run_type: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    frame = _normalize_bars(_read_frame(raw_path))
    finished_at = pd.Timestamp(finished_at_utc or datetime.now(UTC).isoformat())
    if finished_at.tzinfo is None:
        finished_at = finished_at.tz_localize("UTC")
    else:
        finished_at = finished_at.tz_convert("UTC")
    interval_td = _interval_to_timedelta(interval)

    duplicate_count = int(frame.duplicated(subset=["symbol", "timestamp"]).sum())
    unique_rows = frame.drop_duplicates(subset=["symbol", "timestamp"]).copy()

    missing_count = 0
    expected_count = 0
    for _symbol, group in unique_rows.groupby("symbol", sort=True):
        timestamps = group["timestamp"].sort_values()
        observed = int(timestamps.nunique())
        if interval_td is not None and interval_td < pd.Timedelta(days=1):
            missing_for_symbol = _intraday_missing_count(timestamps, interval_td)
        else:
            missing_for_symbol = _business_day_missing_count(timestamps)
        missing_count += missing_for_symbol
        expected_count += observed + missing_for_symbol

    staleness_by_symbol = _latest_staleness_seconds(unique_rows, finished_at=finished_at)
    staleness_values = pd.Series(list(staleness_by_symbol.values()), dtype="float64")
    late_threshold = (
        max(interval_td.total_seconds() * 1.5, 60.0)
        if interval_td is not None
        else 24.0 * 60.0 * 60.0
    )
    late_count = int((staleness_values > late_threshold).sum()) if not staleness_values.empty else 0

    attempts = provider_attempt_count
    if attempts is None:
        attempts = len(symbols) if symbols else int(unique_rows["symbol"].nunique())
        attempts = max(attempts, provider_failure_count)

    duplicate_rate = duplicate_count / float(len(frame)) if len(frame) else 0.0
    missing_rate = missing_count / float(expected_count) if expected_count else 0.0
    late_data_rate = late_count / float(max(len(staleness_by_symbol), 1))
    provider_failure_rate = provider_failure_count / float(max(attempts, 1))

    audit = {
        "run_ts": run_ts,
        "run_type": run_type or "raw_poll",
        "raw_path": str(raw_path),
        "symbols": list(symbols)
        if symbols is not None
        else sorted(unique_rows["symbol"].unique().tolist()),
        "interval": interval,
        "row_count": int(len(frame)),
        "unique_bar_count": int(len(unique_rows)),
        "duplicate_bar_count": duplicate_count,
        "duplicate_bar_rate": duplicate_rate,
        "missing_bar_count": missing_count,
        "missing_bar_rate": missing_rate,
        "expected_bar_count": expected_count,
        "latest_staleness_seconds_by_symbol": staleness_by_symbol,
        "stale_bar_p95_seconds": float(staleness_values.quantile(0.95))
        if not staleness_values.empty
        else None,
        "late_data_count": late_count,
        "late_data_rate": late_data_rate,
        "provider_failure_count": int(provider_failure_count),
        "provider_attempt_count": int(max(attempts, 0)),
        "provider_failure_rate": provider_failure_rate,
        "started_timestamp_utc": (
            str(unique_rows["timestamp"].min().isoformat()) if not unique_rows.empty else None
        ),
        "latest_timestamp_utc": (
            str(unique_rows["timestamp"].max().isoformat()) if not unique_rows.empty else None
        ),
        "finished_at_utc": finished_at.isoformat(),
        "status": _status_from_metrics(
            row_count=int(len(frame)),
            duplicate_rate=duplicate_rate,
            missing_rate=missing_rate,
            late_data_rate=late_data_rate,
            provider_failure_count=int(provider_failure_count),
        ),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return audit


def default_audit_output(project_root: Path, run_ts: str) -> Path:
    return project_root / "artifacts" / "data_quality" / f"data_quality_{run_ts}.json"

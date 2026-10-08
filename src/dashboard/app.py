from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.dashboard.data import DashboardSnapshot, load_dashboard_snapshot


def _display(value: Any, fallback: str = "Not available") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "Not available"


def _number(value: Any) -> str:
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return "Not available"


def _time_index(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        return frame
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="coerce")
    return (
        out.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column)
        .set_index(timestamp_column)
    )


def _drawdown(frame: pd.DataFrame, value_column: str) -> pd.DataFrame | None:
    if value_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if not values.notna().any():
        return None
    return pd.DataFrame({"drawdown": values / values.cummax() - 1.0}, index=frame.index)


def _scorecard_table(scorecard: dict[str, Any] | None) -> pd.DataFrame | None:
    if scorecard is None:
        return None
    rows: list[dict[str, Any]] = []
    for scope, block in [("overall", scorecard.get("overall", {}))]:
        for model, metrics in block.get("by_model", {}).items():
            rows.append({"scope": scope, "model": model, "n": block.get("n"), **metrics})
    for regime, block in scorecard.get("by_regime", {}).items():
        for model, metrics in block.get("by_model", {}).items():
            rows.append(
                {"scope": f"regime: {regime}", "model": model, "n": block.get("n"), **metrics}
            )
    return pd.DataFrame(rows) if rows else None


def _overview(snapshot: DashboardSnapshot) -> None:
    st.subheader("System snapshot")
    st.caption(snapshot.freshness.detail)
    columns = st.columns(4)
    columns[0].metric("Regime", _display(snapshot.overview["regime"]))
    columns[1].metric("Selected model", _display(snapshot.overview["model"]))
    columns[2].metric("Latest signal", _display(snapshot.overview["signal"]))
    columns[3].metric("Portfolio value", _currency(snapshot.overview["portfolio_value"]))
    columns = st.columns(4)
    columns[0].metric("Cash", _currency(snapshot.overview["cash"]))
    columns[1].metric("Position", _number(snapshot.overview["position"]))
    columns[2].metric("Prediction", _number(snapshot.overview["prediction"]))
    columns[3].metric("Latest market bar", _display(snapshot.overview["market_timestamp"]))
    if snapshot.account is not None:
        st.caption(f"Account state updated: {_display(snapshot.account.get('updated_at'))}")


def _portfolio(snapshot: DashboardSnapshot) -> None:
    st.subheader("Portfolio performance")
    if snapshot.equity is None or snapshot.equity.empty:
        st.info("No persisted live-simulation equity history is available.")
    else:
        equity = _time_index(snapshot.equity)
        if "portfolio_value" in equity.columns:
            st.line_chart(equity[["portfolio_value"]])
            drawdown = _drawdown(equity, "portfolio_value")
            if drawdown is not None:
                st.caption("Drawdown from the persisted equity curve")
                st.area_chart(drawdown)
        st.dataframe(equity.tail(100), use_container_width=True)
    st.subheader("Historical backtest")
    if snapshot.backtest is None or snapshot.backtest.empty:
        st.info("No persisted backtest result is available.")
    else:
        backtest = _time_index(snapshot.backtest.reset_index())
        equity_column = "equity" if "equity" in backtest.columns else "portfolio_value"
        if equity_column in backtest.columns:
            st.line_chart(backtest[[equity_column]])
        st.caption("Historical artifact; it is not a live portfolio value.")


def _regimes(snapshot: DashboardSnapshot) -> None:
    st.subheader("Regime history")
    if (
        snapshot.regimes is None
        or snapshot.regimes.empty
        or "regime" not in snapshot.regimes.columns
    ):
        st.info("No persisted regime history is available.")
        return
    regimes = _time_index(snapshot.regimes)
    timeline = regimes.reset_index().tail(1_000)
    st.vega_lite_chart(
        timeline,
        {
            "mark": {"type": "point", "filled": True, "size": 45},
            "encoding": {
                "x": {"field": "timestamp", "type": "temporal", "title": "Time"},
                "y": {"field": "regime", "type": "nominal", "title": "Regime"},
                "color": {"field": "regime", "type": "nominal", "legend": None},
                "tooltip": [
                    {"field": "timestamp", "type": "temporal"},
                    {"field": "regime", "type": "nominal"},
                ],
            },
        },
        use_container_width=True,
    )
    st.dataframe(regimes[["regime"]].tail(200), use_container_width=True)
    counts = regimes["regime"].astype(str).value_counts().rename("observations")
    st.bar_chart(counts)
    if "regime_explanation" in regimes.columns:
        st.caption("Latest regime explanation")
        st.write(_display(regimes.iloc[-1].get("regime_explanation")))


def _models(snapshot: DashboardSnapshot) -> None:
    st.subheader("Active model and evaluation")
    serving = snapshot.serving_status
    if serving["label"] == "SERVING":
        st.success(f"Serving status: {serving['label']} — {serving['detail']}")
    else:
        st.warning(f"Serving status: {serving['label']} — {serving['detail']}")
    if snapshot.active_model is None:
        st.info("No valid active-model registry pointer is available.")
    else:
        st.json(
            {
                "model_id": snapshot.active_model.model_id,
                "model_type": snapshot.active_model.model_type,
                "version": snapshot.active_model.version,
                "regime": snapshot.active_model.regime,
                "artifact_path": str(snapshot.active_model.artifact_path),
                "updated_at": snapshot.active_model.updated_at,
            }
        )
    used = serving.get("latest_prediction_model")
    if isinstance(used, dict):
        st.caption("Model used by the latest active prediction")
        st.json(used)
    scorecard = _scorecard_table(snapshot.scorecard)
    if scorecard is not None:
        st.caption("Historical evaluation scorecard; lower MAE/RMSE is better.")
        st.dataframe(scorecard, use_container_width=True)
    else:
        st.info("No persisted scorecard is available.")
    if snapshot.walkforward is not None and not snapshot.walkforward.empty:
        st.caption("Historical walk-forward portfolio metrics")
        st.dataframe(snapshot.walkforward, use_container_width=True)
    if snapshot.deployments is not None and not snapshot.deployments.empty:
        st.caption("Model switching and deployment decisions")
        st.dataframe(snapshot.deployments.tail(100), use_container_width=True)
    if snapshot.promotion is not None:
        st.caption("Latest promotion policy decision and evidence")
        st.json(snapshot.promotion)


def _trading(snapshot: DashboardSnapshot) -> None:
    st.subheader("Signals and trades")
    if snapshot.trades is None or snapshot.trades.empty:
        st.info("No persisted live-simulation trades are available.")
    else:
        st.dataframe(snapshot.trades.tail(100), use_container_width=True)
    if snapshot.predictions is not None and not snapshot.predictions.empty:
        columns = [
            column
            for column in (
                "timestamp",
                "market_regime",
                "row_id",
                "model_name",
                "model_source",
                "active_model_id",
                "active_model_type",
                "active_model_version",
                "active_regime",
                "inference_ts",
                "y_pred",
                "is_active",
                "signal",
            )
            if column in snapshot.predictions
        ]
        st.caption(
            "Persisted predictions with market timestamp/regime and inference model provenance "
            "(active rows are marked by the producer)."
        )
        st.dataframe(snapshot.predictions.loc[:, columns].tail(100), use_container_width=True)


def _health(snapshot: DashboardSnapshot) -> None:
    st.subheader("System and model health")
    if snapshot.drift is not None:
        st.json(
            {
                "status": snapshot.drift.get("status"),
                "action": snapshot.drift.get("action"),
                "run_ts": snapshot.drift.get("run_ts"),
                "warnings": snapshot.drift.get("warnings", []),
                "errors": snapshot.drift.get("errors", []),
            }
        )
    else:
        st.info("No persisted drift snapshot is available.")
    if snapshot.pipeline_summary is not None:
        st.caption("Latest persisted pipeline run")
        st.json(snapshot.pipeline_summary)
    if snapshot.pipeline_runs:
        rows = [
            {
                "run_ts": run.get("run_ts"),
                "run_kind": "offline/demo" if run.get("offline") else "real",
                "mode": run.get("mode"),
                "replay": run.get("replay"),
                "status": run.get("status"),
                "started_at_utc": run.get("started_at_utc"),
                "finished_at_utc": run.get("finished_at_utc"),
                "duration_seconds": run.get("duration_seconds"),
                "error": run.get("error"),
            }
            for run in snapshot.pipeline_runs
        ]
        st.caption("Pipeline execution history")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if snapshot.registry_history is not None and not snapshot.registry_history.empty:
        st.caption("Registry pointer history")
        st.dataframe(snapshot.registry_history.tail(100), use_container_width=True)


def _render_dashboard(root: Path) -> None:
    snapshot = load_dashboard_snapshot(root)
    st.caption(
        f"State: {snapshot.freshness.label} · Last updated: {_display(snapshot.freshness.updated_at)}"
    )
    if st.button("Refresh now"):
        pass
    if snapshot.warnings:
        with st.expander(f"Artifact warnings ({len(snapshot.warnings)})"):
            for warning in snapshot.warnings:
                st.warning(warning)
    overview, portfolio, regimes, models, trading, health = st.tabs(
        ["Overview", "Portfolio", "Regimes", "Models", "Trading", "Health"]
    )
    with overview:
        _overview(snapshot)
    with portfolio:
        _portfolio(snapshot)
    with regimes:
        _regimes(snapshot)
    with models:
        _models(snapshot)
    with trading:
        _trading(snapshot)
    with health:
        _health(snapshot)


def main() -> None:
    st.set_page_config(page_title="Market Regime Dashboard", layout="wide")
    root = Path(os.environ.get("MARKET_REGIME_DASHBOARD_ROOT", Path.cwd()))
    st.title("Market Regime Detector")
    st.caption("Local, read-only view of real and offline/demo pipeline artifacts.")
    st.info(
        "This dashboard never places trades or connects to a brokerage. Offline/demo runs are labeled "
        "explicitly; real runs use the same durable artifact schema."
    )
    st.sidebar.caption(f"Project root: {root.resolve()}")
    st.sidebar.caption("Refreshes every 30 seconds while this page is open.")

    fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if fragment is None:  # pragma: no cover - supported Streamlit releases provide a fragment API.
        _render_dashboard(root)
        return

    @fragment(run_every=30)
    def refreshable_dashboard() -> None:
        _render_dashboard(root)

    refreshable_dashboard()


if __name__ == "__main__":
    main()

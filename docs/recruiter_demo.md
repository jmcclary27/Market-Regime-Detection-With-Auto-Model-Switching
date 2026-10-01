# Recruiter demo

## What this demonstrates

This is a local-first MLOps demonstration, not a trading product. One offline,
deterministic run produces synthetic market bars; features; rule-based market
regimes; active and shadow model predictions; chronological scorecards and
walk-forward metrics; a guarded model-switching decision; registry history;
drift/quality telemetry; and artifact lineage.

The expected deployment result is a safe `hold` with reason
`no_promotable_challenger`. That is intentional: a model is never promoted just
to make a demo look active.

## Run it

From Git Bash, WSL, or another Bash shell with Docker Desktop running:

```bash
bash scripts/run_recruiter_demo.sh
```

Open [http://localhost:8501](http://localhost:8501). The script builds the
image, resets only the `market-regime-recruiter-demo` Compose volume, runs the
canonical offline pipeline, starts Streamlit, and verifies its health endpoint.

To stop the dashboard and remove the isolated demo state:

```bash
docker compose --project-name market-regime-recruiter-demo down --volumes --remove-orphans
```

## Dashboard route

- **Overview:** regime, selected registry model, latest synthetic prediction,
  and the conservative historical-state badge.
- **Regimes:** deterministic rule-based regime history and explanation.
- **Models:** active registry pointer, scorecard, walk-forward metrics, and the
  recorded safe deployment hold.
- **Trading:** persisted prediction rows; this is not a brokerage or execution
  screen.
- **Health:** drift status, pipeline telemetry, and registry pointer history.

## 3–5 minute talk track

1. Start with the architecture: the system separates ingestion, deterministic
   feature/regime construction, inference, chronological evaluation, guarded
   deployment, and auditable runtime artifacts.
2. Show **Regimes** and explain that the demo uses fixed synthetic inputs and
   rules-based labels so it is repeatable without a network or pre-existing
   model state.
3. Show **Models**: the active baseline remains registry-selected; active and
   shadow inference feed scorecards and walk-forward evaluation. Point out that
   the promotion result is an explicit hold, demonstrating safety over
   automation theater.
4. Show **Health**: persisted telemetry, drift snapshot, lineage, and registry
   history make the run inspectable and replayable.
5. Close with the boundary: the dashboard is read-only, all data is synthetic,
   and there is no live trading or brokerage connection.

## Limitations and safety boundaries

- The demo does not establish trading performance or use real market data.
- The named Docker volume is local-only and disposable; it is intentionally not
  a production artifact store.
- The dashboard shows durable historical artifacts and must not be interpreted
  as a live portfolio or trading signal.
- MLflow uses a fresh Linux-local file store under the demo volume. It never
  mounts the host `mlruns/` directory, preventing Windows artifact paths from
  being interpreted inside the container.

# Recruiter demo

## What this demonstrates

This is a local-first MLOps demonstration, not a trading product. One offline,
deterministic journey produces synthetic market bars; features; HMM market
regimes; active and shadow model predictions; chronological scorecards and
walk-forward metrics; a guarded promotion; a post-switch inference; and a
separate rejected-challenger decision. Registry/deployment history, telemetry,
lineage, and immutable scenario manifests make each step inspectable.

## Run it

From Git Bash, WSL, or another Bash shell with Docker Desktop running:

```bash
bash scripts/run_recruiter_demo.sh
```

Open [http://localhost:8501](http://localhost:8501). The script builds the
image, resets only the `market-regime-recruiter-demo` Compose volume, mounts the
host `models/` directory read-only as the demo artifact source, runs canonical
pipeline stages, starts Streamlit, and verifies its health endpoint.

The source must contain a valid HMM bundle and at least three published,
promotion-eligible experts. The demo snapshots them to its disposable runtime
volume and exits safely before processing if they are missing or incompatible.

To stop the dashboard and remove the isolated demo state:

```bash
docker compose --project-name market-regime-recruiter-demo down --volumes --remove-orphans
```

## Dashboard route

- **Overview:** regime, selected registry model, latest synthetic prediction,
  and the conservative historical-state badge.
- **Regimes:** deterministic HMM regime history and explanation.
- **Models:** active registry pointer, active/shadow scorecards, walk-forward
  metrics, policy evidence, and immutable promotion/rejection scenario rows.
- **Trading:** persisted numerical prediction rows and provenance; this is not a
  brokerage or execution screen.
- **Health:** drift status, pipeline telemetry, and registry pointer history.

## 3–5 minute talk track

1. Start with the architecture: the system separates ingestion, deterministic
   feature/regime construction, inference, chronological evaluation, guarded
   deployment, and auditable runtime artifacts.
2. Show **Regimes** and explain that the demo uses fixed synthetic inputs with
   the real, read-only HMM artifact; no market-data feed is contacted.
3. Show **Models**: active and shadow inference feed the real scorecard and
   walk-forward evaluator. The scenario evidence shows a guarded promotion,
   post-switch active prediction, and separately a genuine policy rejection.
4. Show **Health**: persisted telemetry, drift snapshot, lineage, and registry
   history make the run inspectable and replayable.
5. Close with the boundary: the dashboard is read-only, all data is synthetic,
   and there is no live trading or brokerage connection.

## Limitations and safety boundaries

- The demo does not establish trading performance or use real market data.
- The named Docker volume is local-only and disposable; it is intentionally not
  a production artifact store.
- The source artifacts remain local, read-only inputs. They are ignored by Git,
  so CI verifies fail-closed preflight/contracts rather than this successful
  artifact-dependent journey.
- The dashboard shows durable historical artifacts and must not be interpreted
  as a live portfolio or trading signal.
- MLflow uses a fresh Linux-local file store under the demo volume. It never
  mounts the host `mlruns/` directory, preventing Windows artifact paths from
  being interpreted inside the container.

# Market Regime Detection & Auto-Model Switching

A cost-conscious, local-first MLOps project focused on building an end-to-end
system for market regime detection and automatic model switching.

This project emphasizes **MLOps architecture and system design** over pure
trading alpha.

---

## Project Goals

- Periodically poll market data on a schedule
- Detect market regimes using rule-based methods first, HMM later
- Maintain multiple expert models, one per regime
- Run shadow predictions across all models
- Automatically switch the active model using canary and rollback logic
- Operate primarily on a VM with near-zero cloud cost

---

## Quickstart: offline recruiter demo

The default container command is a deterministic, offline demonstration. It
creates synthetic two-symbol bars, builds features and regimes, trains a local
baseline, writes the active registry pointer, and runs inference. No AWS,
market-data, DVC, or pre-existing model state is required.

```bash
docker compose build
docker compose run --rm market
```

The command prints the generated raw data, features, regime labels, registry,
and predictions. These outputs live under `data/`, `models/`, `registry/`, and
`mlruns/`; they are local runtime state and intentionally excluded from Git.

To run it without Docker:

```bash
pip install -r requirements.txt
python -m src.demo.run
```

## Run individual stages

```bash
python -m src.jobs.poll_market_data
python -m src.features.run_features --input <bars.csv|bars.parquet> --timestamp <timestamp>
python -m src.regimes.run_regime_detection --input <features.parquet> --timestamp <timestamp>
```

The live poll and full pipeline use external market data; the offline demo is
the recommended presentation path.

## Run Machine Learning Parts

- Generate fixture data

```bash
python tools/make_training_fixture.py
```

- Create the pretrained expert artifact

```bash
python tools/make_pretrained_expert.py
```

- Run training

```bash
python -m src.models.train
```

### Safe retraining candidates

Baseline, Ridge-expert, and ARIMA retraining now write versioned artifacts under
`models/candidates/` by default. Candidate directories are not discovered by
inference and do not update the registry or a `latest` pointer. Review their
metadata and evaluation results before an explicit publish step.

```bash
# Global baseline candidate (next-period decimal log-return target)
python -m src.models.train \
  --features-path data/features/latest.parquet \
  --output-dir models/candidates/baseline

# Regime-specific Ridge candidate
python tools/make_pretrained_expert.py \
  --features-path data/features/latest.parquet \
  --regimes-path data/regimes/latest.parquet \
  --regime bullish \
  --output-dir models/candidates/pretrained

# All available LightGBM regime candidates (skips regimes without enough labels)
python tools/retrain_lightgbm_experts.py \
  --features-path data/regimes/latest.parquet \
  --output-dir models/candidates/lightgbm

# Regime-specific ARIMA candidate
python tools/train_arima_expert.py \
  --features-path data/features/latest.parquet \
  --regimes-path data/regimes/latest.parquet \
  --target-col log_return_1_x \
  --target-shift -1 \
  --regime bullish \
  --model-name arima_p1d0q1 \
  --output-dir models/candidates/arima
```

The ARIMA trainer filters the full, time-ordered target frame by the requested
regime only after creating the next-period target. This preserves the one-period
forecast horizon and records `regime_filter_applied: true` in its metadata.
It assigns a stable ID such as `expert_arima_bullish_arima_p1d0q1`, so multiple
ARIMA experts can coexist for one regime. An explicit publish writes canonical
metadata to `models/experts/<regime>/arima/<model_id>.json`; it does not alter
the legacy single-expert `latest.arima.json` unless `--update-legacy-pointer` is
also supplied.
All trainers reject undersized data and write runtime dependency versions to
their artifacts. Constant, low-diversity, or weak candidates are retained only
for diagnosis and cannot publish a live pointer. Ridge, ARIMA, and LightGBM
candidates also record a zero-return test baseline; candidates whose test RMSE
exceeds it cannot publish. LightGBM additionally requires sufficient prediction
diversity and validation RMSE improvement over a train-mean baseline before an
explicit publish is allowed.

## Batch Inference & Shadow Predictions

This project supports **batch inference across all available models** (baseline, regime experts, and pretrained models) to produce **shadow predictions** for comparison and monitoring.

### Purpose
- Run inference for *every* model on the same feature set
- Enable side-by-side comparison between active and shadow models
- Provide the data needed for future **model selection and auto-switching**

### How to Run

```bash
python -m src.inference.batch_predict
```

## Evaluator + Scorecards

This project includes an evaluator that compares all models, overall and per market regime, and writes a scorecard artifact that will be consumed by the auto-switching logic.

### Inputs

The evaluator expects the latest pipeline artifacts:

- `data/features/latest.parquet`
- `data/regimes/latest.parquet`
- `data/predictions/latest.parquet`

Notes:
- `data/predictions/latest.parquet` is long-form and contains `row_id`, `model_name`, and `y_pred`.
- `data/features/latest.parquet` and `data/regimes/latest.parquet` do not contain `row_id`, so the evaluator reconstructs it deterministically by sorting by `timestamp` then `symbol`, then setting `row_id = index`.

### Run evaluator

```bash
python -m src.eval.run_evaluator
```

## Model Registry (Active Model Pointer)

This project uses a lightweight **local model registry** to track which model is
currently active for inference.

The active model is defined by a single local pointer file: `registry/active_model.yaml`.


This file specifies the exact model artifact to load (type, version, path), making
model selection **deterministic and reproducible**.

### Why this exists
- Avoids relying on filesystem heuristics like `latest.joblib`
- Enables safe model switching and rollback
- Keeps inference decoupled from training details

### Usage
- Batch inference loads the active model via the registry when configured
- Shadow predictions are still run for all discovered models
- Switching the active model is as simple as updating `active_model.yaml`

This registry is intentionally minimal and local-first, it does not depend on any
external services.

## Regime-Aware Switching & Canary Windows

PR 9 extends the canary switcher introduced in PR 8 by making model switching **regime-aware** and enforcing **explicit canary windows** before decisions are allowed.

### Goals

- Ensure model promotions are **context-sensitive** to the current market regime
- Prevent premature switching by requiring sufficient evaluation evidence
- Increase robustness and realism of the deployment logic

### Implemented behavior

**1. Regime-Aware Promotion Logic**
- Read the current regime from `data/regimes/latest.parquet`
- Evaluate candidate vs active performance **within the active regime**
- Promotion requires the candidate to outperform the active model in:
  - the current regime, or
  - a weighted combination of recent regimes

**2. Enforced Canary Windows**
- Require a minimum number of evaluations (`N`) before allowing:
  - promote
  - rollback
- Canary state persists across runs until the window is satisfied
- Window progress inferred from deployment events and/or scorecards

**3. Canary State Tracking**
- Track whether a candidate is:
  - newly introduced
  - mid-canary
  - completed (decision eligible)
- State derived from deployment event history (no mutable state)

**4. Safety Guards**
- Automatic rollback on:
  - missing metrics
  - NaNs or infinite values
  - sudden large error spikes
- Explicit `blocked` or `invalid` decision states logged to events

### Outputs

- Extended deployment events with:
  - regime context
  - canary progress
  - decision eligibility flags
- More reliable and explainable model switching behavior

## Pipeline Orchestration

The local pipeline entrypoint stitches together the end-to-end workflow using
programmatic `run(...)` APIs.

### What it does

Running the pipeline will execute, in order:

1. **poll** , fetch raw market bars
2. **features** , build deterministic features + manifest
3. **regimes** , label regimes using rule-based logic
4. **predict** , run batch inference (active + shadow predictions)
5. **eval** , generate scorecards from predictions
6. **switch** , select / switch the active model from evaluation results

### Run it

```bash
python -m src.pipeline.run -v
```

To opt into an S3 bootstrap, set `ARTIFACT_BUCKET` and use the cloud override:

```bash
ARTIFACT_BUCKET=<bucket> docker compose -f docker-compose.yml -f docker-compose.cloud.yml run --rm market pipeline
```

Without that override, S3 synchronization is disabled even if a bucket name is
present in the environment.

## Frozen daily paper-trading experiment

The optional experiment path runs three comparable SPY portfolios with frozen
artifacts, daily Alpaca bars, 3 bps common friction, and next-open target fills.
It is intentionally separate from the active-model switcher. See
[`docs/frozen_daily_experiment.md`](docs/frozen_daily_experiment.md) for the
readiness gate and AWS deployment contract.

Freeze its candidate package only through the explicit, cutoff-bound command:

```bash
python -m tools.freeze_experiment --experiment-id <id> --official-start-date <YYYY-MM-DD> \
  --data-cutoff <YYYY-MM-DD> --features-path <features.parquet> \
  --regimes-path <regimes.parquet> --feature-manifest-path <features.manifest.json> \
  --hmm-artifacts-dir <hmm-dir> --output-dir <artifact-dir>
```

## CI + Hardening

Local quality gates (same as CI):

```bash
ruff check .
ruff format --check .
mypy src tests
pytest -q -m 'not integration'
pytest -q -m integration
```

The integration command runs the deterministic offline demo from empty working
directories; it does not require local `data/`, `models/`, `registry/`, or
`mlruns/` state. See [`docs/test-audit.md`](docs/test-audit.md) for the test
coverage map and fixture policy.

### Data and model versioning (DVC)

This repo can use DVC to version datasets and trained model artifacts through an
S3 remote. It is optional and is not needed for the local demo.

- Pull tracked data/model artifacts:
  `dvc pull`

- Reproduce the configured pipeline:
  `dvc repro`

- After generating new artifacts:
  `dvc push`

## Agentic development

Repository-wide agent instructions live in [`AGENTS.md`](AGENTS.md). The
issue-to-PR, reviewer, verifier, MCP, runtime-aware review, and
GitHub-workflow target architecture is documented in
[`docs/agentic-development.md`](docs/agentic-development.md). Current CI and
local quality-gate commands remain the source of truth for implementation
changes; future agentic phases must not be treated as enabled until their
permissions, tests, and runtime evidence are in place.

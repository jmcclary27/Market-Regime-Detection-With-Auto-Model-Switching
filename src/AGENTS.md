# Agent guidance for `src/`

`src/` contains the production-shaped Python packages. Read the repository
root `AGENTS.md` first; this file focuses on implementation boundaries and ML
invariants.

## Package boundaries

- `ingestion` owns provider access, normalization, and input quality audits.
- `features` owns deterministic feature construction and feature manifests.
- `regimes` owns rule/HMM regime labels and diagnostics.
- `inference` discovers the active registry model and produces long-form
  active/shadow predictions.
- `eval` owns chronological splits, leakage checks, scorecards, and walk-forward
  evaluation.
- `models` owns training and publication guards. Training should produce
  candidates; publication is an explicit operation.
- `registry` owns the active pointer and append-only history.
- `deploy` owns promotion/hold/rollback decisions and canary evidence.
- `pipeline` orchestrates stages and writes run telemetry/lineage.
- `monitoring` and `reporting` turn saved artifacts into auditable health and
  project metrics.
- `aws_lambda` implements immutable, versioned S3 request/completion contracts.
- `experiment` and `trading` are paper-only paths with separate state models.

Do not make one package reach around another package's contract just to find a
file. Use typed function boundaries and explicit paths/configuration.

## Required implementation behavior

- Keep transformations deterministic for fixed inputs, configuration, random
  seeds, and dependency versions.
- Sort time-series data explicitly and preserve symbol/timestamp/row identity.
- Use existing leakage assertions and walk-forward helpers for time-series
  logic. A high score from a random split is not valid evidence here.
- Return or persist stable status/error information for stages that are
  consumed by telemetry or agents.
- When adding an artifact, define its producer, consumer, schema, path,
  freshness, and lineage/hash behavior in code and tests.
- Keep model candidate discovery separate from active-model discovery. Never
  turn a candidate directory into an implicit deployment mechanism.
- For deployment decisions, prefer explicit `blocked`/`hold` states when
  evidence is missing or unsafe. Preserve the reason in deployment history.
- For Lambda/S3 code, pin object versions and hashes, validate paths, preserve
  idempotency, and test duplicate/invalid events.

## Where to test a change

Use the existing test family as a map:

- features/regimes: `test_features_*`, `test_regimes_*`,
  `test_no_leakage_guards.py`, `test_regime_diagnostics.py`;
- inference/models: `test_inference_*`, `test_models_*`,
  `test_model_contracts.py`, `test_registry_active_model.py`;
- evaluation/backtest: `test_eval_metrics.py`, `test_walk_forward_*`,
  `test_backtest_*`, `test_trading_accounting.py`;
- pipeline/data/telemetry: `test_pipeline_replay.py`, `test_replay_audit.py`,
  `test_data_quality.py`, `test_project_metrics.py`,
  `test_drift_monitoring.py`;
- deployment: `test_deploy_switcher_*`, `test_run_promotion.py`,
  `test_retraining_candidates.py`;
- Lambda contracts: `test_lambda_inference_contract.py`,
  `test_lambda_live_sim.py`, `test_lambda_inference_*`.

Add a focused test near the existing family before relying on a broad smoke
test. Use `tmp_path`; do not write to the repository's actual runtime folders.

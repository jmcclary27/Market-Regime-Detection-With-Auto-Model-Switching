# Test and CI audit

This document records the clean-checkout audit completed for CI hardening. It
maps test families to the behavior they protect and defines the fixture policy
for future changes. Runtime directories such as `data/`, `models/`, `registry/`,
`artifacts/`, and `mlruns/` are intentionally ignored and must never be test
inputs unless a test creates them under `tmp_path`.

## Coverage map

| Test family | Production area | Protected behavior |
| --- | --- | --- |
| `test_ingestion`, `test_alpaca`, `test_poll_job`, `test_data_quality` | Ingestion | Provider validation/normalization, credential rejection, deterministic polling outputs, duplicate/missing/stale-data audit status. |
| `test_features_*`, `test_manifest`, `test_no_leakage_guards` | Features | Stable schemas and hashes, row-order invariance, and future-data leakage rejection. |
| `test_regimes_*`, `test_regime_diagnostics` | Regimes | Rule/HMM label contracts, explicit missing-artifact errors, and stable diagnostics. |
| `test_inference_*`, `test_model_contracts`, `test_registry_active_model` | Inference and registry | Active-pointer-only inference, prediction schemas, feature/model safety guards, and registry history. |
| `test_models_*`, `test_retraining_candidates`, `test_train_lightgbm_expert`, `test_run_promotion` | Models and promotion | Candidate/live separation, artifact quality gates, and promotion blocking. |
| `test_eval_*`, `test_walk_forward_*`, `test_backtest_*` | Evaluation and backtest | Chronological splits, metric definitions, deterministic results, and cost/guardrail behavior. |
| `test_deploy_switcher_*` | Deployment | Promote, hold, rollback, scorecard metrics, and decision history. |
| `test_drift_monitoring`, `test_replay_audit`, `test_project_metrics` | Monitoring and reporting | Deterministic telemetry, drift/reference filtering, replay evidence, and report outputs. |
| `test_experiment`, `test_freeze_experiment`, `test_live_sim`, `test_trading_accounting`, `test_lambda_live_sim` | Paper trading and frozen experiment | Frozen artifacts, next-bar fills, accounting, idempotency, locks, and paper-only behavior. |
| `test_lambda_inference_contract` | Lambda inference | Version/hash-scoped requests, bundle layout, safe extraction, and duplicate-event contracts. |
| `test_pipeline_replay` | End-to-end offline demo | A deterministic demo from empty working directories with no ignored local inputs. |

## Findings and controls

- Replaced two live-yfinance tests with mocked provider contracts. The ingestion
  wrapper now verifies delegation rather than downloading market data.
- Replaced the HMM test's local-artifact skip with temporary serialized
  artifacts, including the missing-artifact failure path. A missing fixture is
  now a test failure, not a skip.
- Current MLflow versions require explicit file-store opt-in. All three local
  trainers now preserve the repository's documented file-backed tracking
  contract, preventing clean CI runs from depending on an older MLflow release.
- Isolated the replay test from the repository checkout so local generated
  artifacts cannot make it pass. The demo now uses its packaged settings and a
  generated rule-based demo config rather than unpublished HMM artifacts; the
  test runs the CLI with the current Python executable and explicit source path
  in two empty directories.
- Removed `test_features_package_exist.py`; it contained only `pass` and did
  not protect a behavior. The two batch-inference smoke tests remain because
  one verifies the returned artifact while the other verifies the `latest`
  pointer and public prediction fields.
- CI runs unit and integration selections separately on pull requests and
  `main`. Pytest strict configuration means an invalid marker/configuration or
  an empty required selection fails clearly.

## Required validation

Run the same test selections as CI, then the aggregate suite when practical:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest -q -m 'not integration'
pytest -q -m integration
pytest -q
```

The canonical clean environment is the Docker image used by
[`CI.yml`](../.github/workflows/CI.yml). Tests must use committed fixtures,
deterministic in-test data, mocks, or `tmp_path` setup rather than developer
machine state.

# Agent instructions for Market Regime Detection & Auto-Model Switching

This file is the repository-wide operating contract for Codex, CI agents, PR
reviewers, verification agents, and human collaborators. Read it before
changing code, tests, workflows, infrastructure, data contracts, or model
artifacts. More specific `AGENTS.md` files add local rules for their subtree;
they do not replace this file.

## Mission and system boundary

This repository is a local-first MLOps system for market-regime detection and
automatic model switching. Its primary engineering goals are reproducibility,
safe model deployment, traceable artifacts, and deterministic evaluation. It
is not a brokerage integration and it must not be treated as permission to
trade real capital.

The normal data path is:

```text
market bars -> ingestion -> features -> regimes -> predictions
           -> evaluation/scorecards -> guarded switcher -> active registry
           -> telemetry, lineage, replay audit, and project metrics
```

The repository also contains an optional frozen daily paper-trading experiment
and an event-driven AWS Lambda inference/live-simulation path. These are
separate from the default offline demo and from one another. Preserve those
boundaries when making changes.

## Current state versus target agentic architecture

The repository already has the deterministic foundation for the proposed
architecture:

- `AGENTS.md` guidance is present at the repository root.
- `.github/workflows/CI.yml` runs secret scanning, Dockerized Ruff, formatting,
  mypy, unit tests, and an integration/replay job for pull requests and `main`.
- Pipeline runs write telemetry summaries, lineage, data-quality audits, drift
  snapshots, replay audits, deployment history, and project-metrics reports.
- The local registry and switcher provide explicit active-model pointers,
  guarded promotion/rollback decisions, and history.
- DVC describes the reproducible data pipeline, but DVC and cloud storage are
  optional for the offline demo.

The following phases are architectural targets, not capabilities an agent may
pretend are already installed:

1. Root instructions and Codex CLI context — implemented by this file and the
   scoped guidance files.
2. Issue -> agent implementation -> branch -> PR — templates and contracts are
   documented; GitHub-triggered implementation is not yet enabled.
3. Strong deterministic CI — the current CI is the required baseline; expand
   gates deliberately and keep them reproducible.
4. Codex PR reviewer — not yet enabled.
5. Verification agent — not yet enabled.
6. Market-Regime MCP — not yet enabled. Current agents use checked-in code and
   local artifacts instead.
7. Runtime-aware review — telemetry is collected locally/from saved artifacts,
   but no production telemetry connector is configured.
8. GitHub-triggered workflows — not yet enabled.

When implementing a future phase, update `docs/agentic-development.md` and
this status list in the same change. Never describe a planned connector,
workflow, or reviewer as active until its code, permissions, and CI behavior
are present and tested.

## Repository map

| Path | Responsibility | Typical agent action |
| --- | --- | --- |
| `src/ingestion/` | Market-data adapters, normalization, quality audits | Preserve provider boundaries and input validation. |
| `src/features/` | Deterministic feature construction and manifests | Add leakage and shuffle-invariance tests for feature changes. |
| `src/regimes/` | Rule-based and HMM regime detection and diagnostics | Preserve label semantics, time order, and diagnostic artifacts. |
| `src/inference/` | Active-model inference and shadow predictions | Keep active selection registry-driven and predictions traceable. |
| `src/eval/` | Time-aware evaluation, walk-forward splits, and leakage checks | Never use random or future-looking splits for time-series claims. |
| `src/models/` | Training, promotion, and model safety guards | Candidates are not live models until explicitly published. |
| `src/registry/` | Active pointer, validation, and registry history | Treat pointer changes as deployment events. |
| `src/deploy/` | Canary, promotion, rollback, and decision history | Prefer hold/blocked to an unsafe promotion. |
| `src/pipeline/` | End-to-end orchestration and pipeline telemetry | Keep stages observable and replayable. |
| `src/monitoring/` | Drift, replay audits, and runtime/quality summaries | Emit stable, JSON-serializable metrics with clear status. |
| `src/reporting/` | Project-level metrics and roadmap reporting | Keep generated reports out of source control unless explicitly requested. |
| `src/aws_lambda/` | Pinned S3 inference and paper-live execution contracts | Preserve version IDs, checksums, idempotency, and safe extraction. |
| `src/experiment/` | Frozen daily paper-trading experiment | Do not couple it to mutable registry or auto-promotion behavior. |
| `src/trading/` | Local/demo paper-trading state and execution | Paper-only; do not add brokerage side effects implicitly. |
| `tools/` | Training, packaging, replay backfill, and metrics utilities | Use explicit paths and write versioned candidates/artifacts. |
| `tests/` | Unit, smoke, contract, determinism, and integration tests | Add the narrowest regression test for every behavior change. |
| `docs/` | Human/agent architecture and operational documentation | Update docs when behavior or a contract changes. |
| `.github/` | CI and future issue/PR automation | Keep workflows permission-minimal and branch-aware. |
| `data/`, `models/`, `registry/`, `mlruns/`, `artifacts/`, `runs/` | Local runtime state and generated artifacts | Inspect as needed; do not commit generated state by default. |
| `infra/`, `docker/`, `docker-compose*.yml` | Deployment and container definitions | Make infrastructure changes explicit and contract-tested. |

## First actions for every task

1. Start at the repository root and read this file plus the nearest scoped
   `AGENTS.md`.
2. Inspect `git status --short --branch`. Existing changes belong to the user;
   preserve them and avoid unrelated cleanup.
3. Read the issue, acceptance criteria, and the relevant source/tests/docs
   before editing. Identify the data contract and the artifact(s) that prove
   completion.
4. Search with `rg` or `rg --files`. Trace callers and tests before changing a
   public function, CLI, file layout, or serialized schema.
5. Make the smallest coherent change. Do not silently broaden an issue from a
   code fix into infrastructure, cloud deployment, credentials, or workflow
   automation.
6. Add or update deterministic tests and documentation with the change.
7. Run the smallest relevant checks while iterating, then run the full local
   quality gate before handing off.
8. Report changed files, validation commands/results, known limitations, and
   any follow-up issue needed for an unimplemented target phase.

## Development environment and canonical commands

The project targets Python 3.11. Dependencies are in `requirements.txt`;
format/lint/type/test settings are in `pyproject.toml` and `pytest.ini`.
Docker is the canonical reproducible environment because CI builds the image
from `Dockerfile` and runs checks inside it.

Offline demo, recommended smoke path:

```bash
docker compose build
docker compose run --rm market
```

Equivalent local command when dependencies are installed:

```bash
python -m src.demo.run
```

Full local quality gate, matching CI intent:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest -q
```

Useful focused commands:

```bash
pytest -q tests/test_<area>.py
pytest -q -m 'not integration'
pytest -q -m integration
python -m src.pipeline.run -v
python tools/collect_project_metrics.py --subject-run-ts latest --history-source lineage --out-dir data/reporting --roadmap-path docs/future_metrics_roadmap.md
```

Container equivalents include `docker run --rm market-regime:ci ruff`,
`docker run --rm market-regime:ci sh -lc "ruff format --check ."`,
`docker run --rm market-regime:ci sh -lc "mypy src tests"`, and
`docker run --rm market-regime:ci sh -lc "pytest -m 'not integration'"`.

The default demo is offline and synthetic. Do not make ordinary tests depend
on network access, Alpaca/yfinance credentials, AWS credentials, DVC remotes,
MLflow servers, or pre-existing local model/data state.

## Change classification and required evidence

Before editing, classify the change. The classification determines the test
and review evidence expected in the PR.

| Change | Minimum evidence |
| --- | --- |
| Pure documentation | Markdown review; run link/path checks if links or commands changed. |
| Feature or bug fix in one package | Focused regression test plus relevant lint/type checks. |
| Feature or bug fix crossing pipeline stages | Contract test, deterministic fixture, and end-to-end smoke coverage. |
| Time-series feature, regime, evaluation, or trading logic | No-leakage/time-order test, determinism test, and metric interpretation in the PR. |
| Model trainer, artifact, registry, or promotion logic | Candidate/live separation test, artifact contract test, and promotion/rollback tests. |
| Serialized schema or S3/Lambda contract | Backward/invalid-input tests, checksum/version/idempotency coverage, and docs update. |
| CI, workflow, permissions, Docker, or Terraform | YAML/config validation where available, least-privilege review, and a local equivalent check. |
| Runtime-aware/MCP integration | Read-only contract first, redaction/timeout behavior, stale-data handling, and fixture-based tests without production access. |

If a requested behavior cannot be tested deterministically, document why and
define a bounded verification strategy. Do not weaken or skip an existing gate
to make a change pass.

## Non-negotiable ML and data invariants

- Time-series evaluation must be chronological. Use the existing walk-forward
  helpers and leakage checks; do not introduce random splits for model claims.
- Features, targets, regimes, and predictions must retain their documented
  timestamp/symbol/row identity. Long-form predictions use `row_id` and model
  identity; evaluation reconstructs row identity deterministically when needed.
- Feature and regime computation must be invariant to input row order where the
  contract says it is. Sort explicitly and test shuffled inputs.
- Never use future information when constructing a target, feature, regime, or
  fill. A next-period target is created before a regime filter in the ARIMA
  trainer so the forecast horizon remains correct.
- Reject or record missing, non-finite, duplicate, stale, or insufficient data;
  do not silently repair a quality failure into a promotion.
- Candidate artifacts under `models/candidates/` are diagnostic/unpublished.
  Inference must not discover them, and training must not mutate the active
  registry or a live `latest` pointer as a side effect.
- Live promotion requires the existing model safety/quality guards, evaluation
  evidence, canary-window rules, finite metrics, and explicit deployment
  history. A hold or blocked result is valid and safer than guessing.
- Registry pointers must identify exact artifacts. Preserve registry history
  and make rollback possible.
- Lineage hashes and run metadata are evidence, not decoration. When an output
  changes, update the associated lineage/manifest/contract tests.
- Never commit credentials, tokens, private keys, `.env` files, generated
  datasets, model binaries, MLflow databases, or runtime state unless a task
  explicitly changes the repository policy and CI proves the reason safe.

## Artifact and runtime-state rules

Most runtime outputs are intentionally local and ignored by Git. Common areas
include:

```text
data/raw/                 downloaded or generated bars
data/features/            feature parquet and manifests
data/regimes/             regime parquet
data/predictions/         long-form active/shadow predictions
data/scorecards/          evaluation outputs
data/reporting/           generated project metrics reports
artifacts/lineage/        run lineage and hashes
artifacts/pipeline_runs/  step-level pipeline summaries
artifacts/replay/         replay audit results
artifacts/regimes/        regime diagnostics
data/deployments/         deployment decision event history
models/candidates/        unpublished model candidates
models/experts/           published expert artifacts
registry/                 active pointer and history
mlruns/                   local MLflow tracking state
```

Use temporary directories or pytest `tmp_path` for tests. Do not use real
runtime directories as test fixtures. When investigating runtime state, prefer
read-only inspection and never overwrite `latest` artifacts just to explore.

For AWS/Lambda paths, the immutable contract is especially strict: requests
reference versioned S3 objects and SHA-256 digests; request objects are written
last; completion markers are the canonical handoff; duplicate events must be
idempotent; and model bundle extraction must reject path traversal and invalid
layouts. Read `docs/aws_lambda_inference.md` before touching that path.

## Issue -> branch -> PR contract

The target delivery unit is one issue, one focused branch, and one reviewable
PR. The repository currently documents this contract but does not yet run an
automatic GitHub issue-to-implementation workflow.

For each implementation task:

- Link the branch and PR to one issue. Use a descriptive branch such as
  `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`, or
  `chore/<short-name>`.
- Preserve the issue's acceptance criteria verbatim in the PR checklist and
  map each criterion to code, test, artifact, or explicit verification output.
- Keep generated artifacts, credentials, and local runtime state out of the
  branch unless the issue specifically changes a checked-in fixture or schema.
- Do not push, open, merge, approve, or comment on a PR unless the user/task
  explicitly authorizes that external action.
- A PR is not complete because tests pass: it must also explain architectural
  impact, data/model contract impact, runtime/telemetry impact, rollback, and
  any unimplemented future-phase dependency.

Use `.github/ISSUE_TEMPLATE/feature.md`,
`.github/ISSUE_TEMPLATE/bug.md`, and `.github/PULL_REQUEST_TEMPLATE.md` as the
structured handoff format.

## Review and verification expectations

A future PR reviewer agent should be read-only and should evaluate:

1. Does the implementation satisfy every issue acceptance criterion?
2. Is the change in the correct package and consistent with the pipeline map?
3. Could it introduce future-data leakage, nondeterminism, unsafe promotion,
   artifact ambiguity, or an unbounded runtime side effect?
4. Are tests testing behavior rather than merely execution, and are negative
   paths covered?
5. Are telemetry, lineage, schema, and documentation updated where required?
6. Does CI exercise the same path a developer claims to have validated?

A separate verification agent should independently trace acceptance criteria to
tests and observable outputs. It should not trust the implementation's own
summary. For runtime claims, it should distinguish:

- static evidence: source, tests, manifests, and CI results;
- local evidence: saved pipeline telemetry, replay audits, drift, deployment,
  and project-metrics artifacts;
- production evidence: an authenticated, time-bounded telemetry source.

No production evidence is available to agents until a reviewed connector/MCP is
implemented. Never invent runtime state or report a local fixture as
production telemetry.

## Runtime-aware and MCP design constraints

The intended Market-Regime MCP should begin as a read-only, bounded adapter
over approved metrics and runtime state. It should expose stable, narrow
operations such as run summaries, data-quality status, drift snapshots,
deployment decisions, active-model identity, and replay results. It should not
expose arbitrary shell execution, arbitrary filesystem writes, credentials,
raw secrets, or unrestricted trading actions.

Any future connector must define:

- an explicit schema and timestamp/freshness field for every response;
- source, run ID, Git commit, model ID/version, and artifact hash where known;
- timeout, retry, pagination, and stale-data behavior;
- redaction and authorization boundaries;
- deterministic fixtures and contract tests;
- behavior when telemetry is missing, partial, contradictory, or delayed.

Reviewers must treat runtime context as evidence with provenance, not as an
override of repository safety gates. Production telemetry may inform a review
or block a promotion, but it must not silently bypass tests, lineage, canary
windows, or explicit approval policy.

## Documentation and code-style rules

- Use Markdown headings and concise tables when they make an agent decision
  unambiguous. Keep current behavior separate from planned behavior.
- Update the nearest documentation when a command, path, schema, safety rule,
  or operational contract changes.
- Follow the existing Ruff configuration: Python 3.11, line length 100,
  import sorting, and the enabled `E`, `F`, `I`, `B`, and `UP` rules.
- Prefer small typed functions, explicit paths, stable serialized fields, and
  clear exceptions over hidden global state or heuristic discovery.
- Do not add broad dependencies for a narrow task. Keep Lambda dependencies
  smaller than the development environment where possible.

## Handoff format

End an agent task with:

```text
Summary:
- <what changed and why>

Validation:
- <exact command>: <result>

Acceptance criteria:
- [x]/[ ] <criterion> -> <evidence or blocker>

Risk and rollback:
- <operational/data/model risk and how to reverse it>

Follow-up:
- <next issue or "none">
```

If blocked, state the concrete missing authority, dependency, or user choice.
Do not hide an unverified claim behind a green unit-test result.

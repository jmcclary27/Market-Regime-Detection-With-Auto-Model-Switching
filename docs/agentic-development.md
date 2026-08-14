# Agentic development architecture

This document describes the intended issue-to-runtime feedback loop for the
repository. `AGENTS.md` is the day-to-day operating contract; this is the
architecture and rollout reference for the eight phases.

## Design principle

Automation should increase delivery speed without weakening reproducibility,
model safety, or human control over external side effects. Every automated
decision needs a bounded input, a durable output, and an independent check.

```text
Issue with acceptance criteria
        |
        v
Implementation agent -> focused branch -> deterministic CI
        |                                      |
        +---------------- PR ------------------+
                         |
              architecture/ML review agent
                         |
                  independent verifier
                         |
                    human approval
                         |
             merge/deploy/pipeline execution
                         |
       lineage + telemetry + runtime-aware review context
```

The arrows after human approval must remain explicit. Runtime context can
inform a review and can safely block a promotion, but it must not silently
execute trading, merge code, or bypass a quality gate.

## Phase status and deliverables

| Phase | Deliverable | Repository status | Completion signal |
| --- | --- | --- | --- |
| 1 | Repository instructions and Codex CLI context | In place | Root and scoped agent guidance are current. |
| 2 | Issue -> branch -> PR implementation loop | Contract in place | Structured issue/PR templates; authenticated automation still pending. |
| 3 | Strong deterministic CI | Baseline in place | `.github/workflows/CI.yml` passes secret scan, Docker checks, tests, and replay/integration gates. |
| 4 | Codex PR reviewer | Planned | A read-only reviewer posts findings tied to changed files and architecture rules. |
| 5 | Verification agent | Planned | An independent verifier maps every acceptance criterion to evidence. |
| 6 | Market-Regime MCP | Planned | A bounded, read-only connector passes schema, auth, redaction, timeout, and fixture tests. |
| 7 | Runtime-aware review | Partial foundation | Local metrics exist; approved production telemetry source is still pending. |
| 8 | GitHub-triggered workflows | Planned | Permissions, approvals, retries, idempotency, and audit trail are tested end to end. |

## Current evidence surfaces

Agents should use the narrowest evidence surface that answers a question:

| Question | Current source |
| --- | --- |
| Did the pipeline run and which step failed? | `artifacts/pipeline_runs/pipeline_run_<run_ts>.json` |
| Can the run be reproduced exactly? | `artifacts/lineage/` and `artifacts/replay/` |
| Is input data usable? | `artifacts/data_quality/` and lineage-linked audits |
| Has the feature/model/regime distribution shifted? | `artifacts/drift/` and reporting outputs |
| What model is active? | `registry/active_model.yaml` plus registry history |
| What deployment decision occurred? | `data/deployments/events.parquet` and deployment history consumed by reporting |
| What are project-level trends? | `data/reporting/latest_report.json` and `latest_report.md` |
| What code/config produced an artifact? | lineage Git commit, config hash, and artifact SHA-256 fields |

These are local artifacts and may be absent, stale, partial, or ignored by
Git. Agents must report freshness and provenance. The absence of a file is not
evidence that a production system is healthy.

## Agent responsibilities

### Implementation agent

The implementation agent owns a focused change on a branch. It reads the issue
and applicable agent instructions, identifies impacted contracts, adds tests,
runs checks, and prepares a PR-ready handoff. It does not approve its own PR,
claim production validation without a connector, or mutate live state as a
side-effect of testing.

### PR reviewer agent

The reviewer is read-only. It checks architecture boundaries, leakage,
determinism, model safety, artifact provenance, error handling, permissions,
and documentation. Findings should include severity, file/line, why it
matters, and a concrete remediation. It should not rewrite code or dismiss a
failure because a broad test suite passes.

### Verification agent

The verifier starts from the issue, not the implementation summary. For every
acceptance criterion it records `verified`, `not verified`, or `blocked`, with
independent evidence. It should check negative paths, generated outputs,
schemas, and the actual CI job that protects the behavior.

### Runtime-aware reviewer

This reviewer consumes only an approved, time-bounded telemetry snapshot. It
must record source, freshness, run ID, model/version, and commit when
available. It should distinguish observation from inference and should fail
closed when telemetry is stale or unavailable for a decision that requires it.

## Proposed MCP surface

The first Market-Regime MCP should be intentionally small and read-only. A
possible future contract is:

| Operation | Required response context |
| --- | --- |
| `get_run_summary(run_ts)` | status, steps, duration, source, freshness |
| `get_data_quality(run_ts)` | audit status, quality metrics, input IDs/hashes |
| `get_drift(run_ts)` | reference run, windows, metrics, thresholds, freshness |
| `get_active_model()` | model ID/type/version/path or artifact hash, changed-at |
| `get_deployment_history(...)` | decisions, canary state, evidence, timestamps |
| `get_replay_audit(run_ts)` | exact/semantic pass, drift, failure breakdown |
| `get_project_metrics(...)` | report version, source runs, generated-at, freshness |

The exact names are a design proposal, not an existing API. Before enabling
the MCP, define schemas, authentication, allowed data, redaction, rate and
time limits, stale-data policy, and fixture-backed tests. Never expose
arbitrary SQL, shell, filesystem writes, secret values, or brokerage actions.

## GitHub workflow shape

The eventual GitHub-triggered flow should be idempotent and permission-minimal:

1. An issue is labeled or explicitly dispatched for agent implementation.
2. The workflow validates that the issue contains acceptance criteria and no
   unsupported external side effect.
3. The agent creates or reuses a dedicated branch and records the issue ID.
4. The implementation agent edits only the repository and opens a PR.
5. CI runs on the PR. Reviewer and verifier agents receive the diff, issue,
   test results, and allowed runtime snapshot.
6. A human approval gate is required before merge or deployment.
7. Every retry uses an idempotency key and updates the same run/PR record rather
   than creating duplicate branches, comments, or deployments.

Required controls include least-privilege GitHub tokens, fork/untrusted-code
boundaries, secret isolation, concurrency cancellation, retry limits, audit
logs, artifact retention, and explicit handling for stale issue or PR state.

## Rollout gates

Do not enable the next phase until the prior phase is observable and reversible.

- Before Phase 4: CI is required and branch protection prevents bypassing it.
- Before Phase 5: reviewer findings and verifier results are stored as distinct
  statuses; neither agent can merge.
- Before Phase 6: MCP responses are read-only, redacted, provenance-bearing,
  timeout-bounded, and tested with stale/missing data.
- Before Phase 7: telemetry freshness and production-vs-local provenance are
  visible in every review decision.
- Before Phase 8: workflows are idempotent, permission-minimal, approval-gated,
  and tested against duplicate events and partial failures.

## Open implementation work

The following should become separate issues rather than being hidden in a
large automation change:

- Define protected-branch requirements and required CI status names.
- Add a read-only PR reviewer runner with a stable finding schema.
- Add an independent acceptance-criteria verifier.
- Define and implement the Market-Regime MCP contract and fixtures.
- Choose an approved production telemetry source and retention/freshness SLO.
- Add GitHub workflow idempotency, permissions, approvals, and audit storage.

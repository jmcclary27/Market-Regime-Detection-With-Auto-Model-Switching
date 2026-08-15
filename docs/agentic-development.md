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
| 4 | Codex PR reviewer | Implemented | `.github/workflows/codex-pr-review.yml` posts advisory, read-only findings for eligible PRs. |
| 5 | Verification agent | Implemented | `.github/workflows/requirement-verification.yml` posts one advisory, post-CI requirement-verification report for eligible same-repository PRs. |
| 6 | Market-Regime MCP | Planned | A bounded, read-only connector passes schema, auth, redaction, timeout, and fixture tests. |
| 7 | Runtime-aware review | Partial foundation | Local metrics exist; approved production telemetry source is still pending. |
| 8 | GitHub-triggered implementation workflows | Planned | Issue-to-PR automation, approvals, retries, and audit trail are not enabled. The Phase 4 reviewer and Phase 5 verifier are advisory-only GitHub-triggered workflows; neither implements issues. |

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

The Phase 4 reviewer runs on non-draft pull requests from branches in this
repository when they are opened, reopened, marked ready for review, or updated.
Fork pull requests are skipped: Codex is not invoked and the OpenAI API key is
never exposed to fork-controlled code. The workflow uses `pull_request_target`
so its workflow definition comes from the default branch, then checks
out GitHub's eligible pull request merge commit only for read-only inspection.

Codex receives `contents: read`, its `:read-only` permission profile, and a
privilege-drop safety strategy. It does not run project code or tests, install
dependencies, access an MCP service, modify files, commit, push, merge, deploy,
or access production telemetry. A separate fresh job has only `issues: write`
to create or update one advisory PR comment; it has no repository-content write
permission. The comment is replaced on subsequent eligible runs so retries and
new commits do not create comment spam.

The reviewer checks architecture boundaries, leakage, determinism, model safety,
artifact provenance, error handling, permissions, documentation, and the
software, ML/data, trading/simulation, and configuration risks described in its
checked-in policy. It treats PR-provided files and metadata as untrusted input.
Its findings use `BLOCKING`, `IMPORTANT`, and `SUGGESTION`, each with a path
when applicable, why it matters, and a recommended fix. It explicitly reports
`No material issues found` when no material issue is identified.

The workflow fails without publishing a success-like comment if Codex fails or
returns no review. It is advisory: human review and deterministic CI remain
authoritative, and a lack of findings is not approval to merge or deploy. To
enable it, add the `OPENAI_API_KEY` GitHub Actions secret and allow the
workflow's scoped `issues: write` token permission in repository Actions
settings. Disable it by removing the secret or disabling
`.github/workflows/codex-pr-review.yml`.

### Verification agent

The Phase 5 verifier runs after a completed `CI` workflow for a non-draft PR
whose head branch belongs to this repository. It starts from the linked issue,
not the implementation summary, and assesses every explicit checklist item
under `## Acceptance criteria`. Each item retains the issue's wording and has
separate implementation, test/CI, and assumptions evidence. Status is `PASS`,
`FAIL`, or `UNVERIFIED`; the overall result is `PASS` only when every material
criterion is supported. Failed CI is still evidence, so the verifier runs after
both successful and failed CI runs.

The PR must close exactly one same-repository issue. The workflow first uses
GitHub's closing-issue relationship, then deterministically parses `Closes`,
`Fixes`, or `Resolves` references in `#123`, `owner/repo#123`, or same-repo
issue-URL form. No link, a cross-repository-only reference, multiple plausible
issues, stale CI evidence, or a missing explicit checklist produces an advisory
`UNVERIFIED` comment; it never guesses an issue. Fork and draft PRs are skipped
before the job that can receive `OPENAI_API_KEY`.

The verifier checks out the PR merge commit only for inspection and does not run
tests, install dependencies, modify state, commit, push, merge, deploy, or
access production telemetry. It receives only `contents: read`,
`pull-requests: read`, `issues: read`, and `actions: read`. A fresh publishing
job has only `issues: write` and updates one marker-tagged PR comment, making
retries idempotent. The report is advisory: deterministic CI and human review
remain authoritative, and the LLM can only assess available evidence. Missing,
ambiguous, or unavailable evidence is explicitly `UNVERIFIED`.

This is distinct from the Phase 4 reviewer, which looks for technical,
architecture, safety, and code-quality risks. The verifier does not add generic
review findings; it answers whether stated issue requirements have evidence.
To enable it, configure the `OPENAI_API_KEY` Actions secret and allow the
workflow's scoped token permissions. Disable it by removing the secret or
disabling `.github/workflows/requirement-verification.yml`. It does not create
branches or PRs, so Phase 8 issue-to-PR automation remains out of scope.

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
5. CI runs on the PR. The reviewer receives the diff and repository context;
   the separate verifier receives one linked issue plus deterministic CI
   evidence after CI completes; a future runtime-aware reviewer may receive an
   approved runtime snapshot.
6. A human approval gate is required before merge or deployment.
7. Every retry uses an idempotency key and updates the same run/PR record rather
   than creating duplicate branches, comments, or deployments.

Required controls include least-privilege GitHub tokens, fork/untrusted-code
boundaries, secret isolation, concurrency cancellation, retry limits, audit
logs, artifact retention, and explicit handling for stale issue or PR state.

## Rollout gates

Do not enable the next phase until the prior phase is observable and reversible.

- Before Phase 4: CI is required and branch protection prevents bypassing it.
- Phase 5 is enabled: reviewer findings and verifier results are distinct
  advisory PR comments; neither agent can merge.
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
- Define and implement the Market-Regime MCP contract and fixtures.
- Choose an approved production telemetry source and retention/freshness SLO.
- Add GitHub workflow idempotency, permissions, approvals, and audit storage.

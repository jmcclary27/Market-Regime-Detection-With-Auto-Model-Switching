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
| 2 | Issue -> branch -> PR implementation loop | Implemented | A maintainer-dispatched Codex workflow validates one issue, produces one dedicated branch, and opens one PR after validation. |
| 3 | Strong deterministic CI | Baseline in place | `.github/workflows/CI.yml` passes secret scan, Docker checks, tests, and replay/integration gates. |
| 4 | Codex PR reviewer | Implemented | `.github/workflows/codex-pr-review.yml` posts advisory, read-only findings for eligible PRs. |
| 5 | Verification agent | Implemented | `.github/workflows/requirement-verification.yml` posts one advisory, post-CI requirement-verification report for eligible same-repository PRs. |
| 6 | Market-Regime MCP | Planned | A bounded, read-only connector passes schema, auth, redaction, timeout, and fixture tests. |
| 7 | Runtime-aware review | Partial foundation | Local metrics exist; approved production telemetry source is still pending. |
| 8 | GitHub-triggered implementation workflows | Implemented, controlled scope | `.github/workflows/codex-issue-implementation.yml` is manually dispatched from the default branch. It is branch-collision safe, validation-gated, and stops at a PR; approvals, merges, deployments, and retries after a collision remain human-controlled. |

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

#### Maintainer-dispatched issue implementation

Run **Codex issue implementation** from the repository default branch and set
the required `issue_number` input, for example `123`. The workflow accepts only
an open GitHub issue—not a pull request—with one or more checklist entries
under `## Acceptance criteria`. It retrieves the title, description, and
criteria directly from GitHub; maintainers must not paste issue content into
workflow inputs.

The preflight job confirms that the dispatcher is a repository maintainer or
administrator, records the default-branch SHA, and derives a branch named
`agent/issue-123-short-description`. Any existing `agent/issue-123-*` branch
causes a safe failure: the workflow never overwrites, force-pushes, or creates
a replacement branch automatically. Closed issues, malformed inputs, missing
criteria, or non-default-branch dispatches also fail before Codex runs.

Codex receives the issue content as explicitly untrusted data and works in a
read-only-token job with a writable checkout. It must inspect, plan, implement,
test, self-check the criteria, and produce a PR description based on the
repository PR template. The workflow records the plan, final handoff, patch,
and validation output as retained Actions artifacts. It independently checks
workflow YAML syntax and runs the Docker-aligned full quality gate before any
publication.

Only a separate publish job receives `CODEX_AGENT_GITHUB_TOKEN`. It applies
the validated patch to the recorded base SHA, commits exactly one
`agent/issue-*` branch without force push, and opens a non-draft PR with
`Closes #<issue-number>`, the implementation plan, criterion evidence, and
validation results. It has no merge, approval, deployment, repository-settings,
secret-management, issue-closing, or branch-deletion action. The resulting PR
uses the normal `pull_request` path, so CI, the read-only reviewer, and the
requirement verifier run before human review and merge.

Configure these Actions secrets before dispatching:

| Secret | Where it is exposed | Required permission/setup |
| --- | --- | --- |
| `OPENAI_API_KEY` | Codex implementation job only | Existing OpenAI API key secret used by `openai/codex-action@v1`. |
| `CODEX_AGENT_GITHUB_TOKEN` | Publish job only | Dedicated fine-grained bot PAT for this repository: Contents read/write, Pull requests read/write, and Metadata read. Do not grant administration, Actions/secrets, workflow, issue-write, or bypass privileges. |

`CODEX_AGENT_GITHUB_TOKEN` is deliberately distinct from `GITHUB_TOKEN` so
the bot-created PR can trigger the repository's normal PR workflows. If Codex,
validation, patch application, commit, push, or PR creation fails, the workflow
surfaces the failure and preserves available artifacts; it does not open a
success-like or partial PR. A human must investigate, decide whether to delete
or continue an existing agent branch, approve the resulting PR, merge it, and
authorize any deployment separately.

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

The controlled GitHub-triggered implementation flow is permission-minimal and
stops at pull-request creation:

1. A maintainer explicitly dispatches the default-branch workflow for one open
   issue.
2. The workflow validates the maintainer, issue, criteria, base SHA, and the
   absence of an existing agent branch for that issue.
3. The implementation agent edits only an isolated repository checkout and
   records its plan and handoff.
4. A fresh publish job applies the validated patch, creates a dedicated branch,
   and opens one PR.
5. CI runs on the PR. The reviewer receives the diff and repository context;
   the separate verifier receives one linked issue plus deterministic CI
   evidence after CI completes; a future runtime-aware reviewer may receive an
   approved runtime snapshot.
6. A human approval gate is required before merge or deployment.
7. A duplicate run waits for the per-issue concurrency lock, then fails if an
   agent branch already exists rather than overwriting it or creating a
   duplicate PR.

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
- Phase 8 is enabled for explicit maintainer dispatch only: the workflow is
  collision-safe, permission-minimal, validation-gated, and tested against
  malformed inputs and partial failures. Automatic issue commands, agent
  retries that reuse a branch, merge automation, deployment automation, and
  runtime-aware implementation remain out of scope.

## Open implementation work

The following should become separate issues rather than being hidden in a
large automation change:

- Define protected-branch requirements and required CI status names.
- Define and implement the Market-Regime MCP contract and fixtures.
- Choose an approved production telemetry source and retention/freshness SLO.
- Consider a future trusted issue-command trigger only after its maintainer
  authorization, retry, and prompt-injection boundaries are separately tested.

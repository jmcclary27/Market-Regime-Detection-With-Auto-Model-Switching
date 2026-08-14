# Agent guidance for `.github/`

This directory controls repository automation. Read the root `AGENTS.md` and
`docs/agentic-development.md` before changing workflows, templates, triggers,
permissions, or required checks.

## Current CI contract

`.github/workflows/CI.yml` currently:

- runs on pull requests and pushes to `main`;
- scans for secrets;
- builds one Docker image and runs Ruff lint, Ruff format check, mypy, and
  non-integration pytest in that image;
- runs the integration/replay suite on the `main` branch after Docker quality
  checks pass.

Keep the Dockerized commands and local commands aligned. If a gate changes,
update `AGENTS.md`, the PR template, and the relevant docs in the same PR.

## Workflow safety

- Use least-privilege `permissions`; start with read-only permissions unless a
  step has a documented need for more.
- Treat pull-request code and fork events as untrusted. Never expose secrets to
  arbitrary PR code.
- Pin or otherwise review third-party actions and record why a permission or
  action is needed.
- Make retries and duplicate webhook events idempotent. Never create duplicate
  branches, comments, deployments, or model promotions on a retry.
- Keep merge, deployment, and external messages behind explicit human approval
  until the corresponding future phase is implemented and tested.
- Do not use GitHub Actions as an arbitrary shell or filesystem access layer for
  an agent. Scope paths, inputs, tokens, and artifact retention.

## Templates

Issue templates must request a problem statement, acceptance criteria,
constraints, affected components, deterministic evidence, and runtime/rollback
considerations. PR templates must map acceptance criteria to evidence and
declare test results, generated artifacts, telemetry, risk, and follow-up work.

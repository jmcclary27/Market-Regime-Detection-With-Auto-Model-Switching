---
name: Bug report
about: Capture a reproducible defect and its guard condition
title: ""
labels: "bug"
assignees: ""
---

## Observed behavior

What happened, and what should have happened?

## Reproduction

- Commit/environment:
- Exact command or trigger:
- Minimal input/artifact/run ID:
- Logs or error:

## Acceptance criteria

- [ ] A regression test fails before the fix and passes after it.
- [ ] The fix preserves relevant data/model/telemetry contracts.
- [ ] Invalid, missing, stale, or duplicate input behavior is covered when relevant.

## Scope and risk

- Affected package/files:
- Could this involve leakage, nondeterminism, model promotion, registry state,
  paper-trading state, or external side effects?
- Rollback or containment plan:

## Validation plan

- Focused tests:
- Full quality gate:
- Replay/integration/runtime evidence:

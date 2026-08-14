---
name: Feature or architecture change
about: Define a reviewable, testable implementation task
title: ""
labels: "enhancement"
assignees: ""
---

## Problem and outcome

What problem exists today, and what user or system outcome should change?

## Acceptance criteria

Write observable criteria. Each criterion must be verifiable by a test, a
checked-in artifact/schema, a CI result, or an explicitly approved manual
check.

- [ ]
- [ ]

## Scope

- Likely source packages/files:
- Data/model/artifact contracts affected:
- Documentation or CI changes required:
- Explicitly out of scope:

## Safety and determinism

- Does this touch time-series ordering, target construction, regimes, model
  promotion, registry state, or trading simulation?
- What prevents future-data leakage, nondeterminism, unsafe publication, or
  duplicate execution?
- What is the failure behavior and rollback/disable path?

## Runtime and agentic context

- Is current evidence static, local runtime state, or production telemetry?
- Does this require a future reviewer, verifier, MCP, or GitHub workflow?
- If external access is needed, what is the minimum permission and data scope?

## Validation plan

- Focused tests:
- Full quality gate:
- Integration/replay or manual evidence:

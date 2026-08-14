## Summary

<!-- What changed, why, and which issue does this close? -->

Closes #

## Acceptance criteria

<!-- Copy the issue criteria and map each item to evidence. -->

- [ ] Criterion 1 — evidence:
- [ ] Criterion 2 — evidence:

## Scope and architecture

- [ ] The change is limited to the issue's intended scope.
- [ ] Impacted source packages, data contracts, and artifact paths are listed.
- [ ] Current behavior is not described as a future agentic phase.
- [ ] Documentation/agent guidance was updated where commands or contracts changed.

## ML/data safety

- [ ] Time ordering and leakage behavior were considered.
- [ ] Determinism and row identity were preserved or tested.
- [ ] Candidate artifacts remain separate from active/published artifacts.
- [ ] Promotion, rollback, canary, or invalid-input behavior is covered when relevant.

## Validation

<!-- Include exact commands and results. -->

- `ruff check .`:
- `ruff format --check .`:
- `mypy src tests`:
- `pytest -q`:
- Focused/integration checks:

## Runtime, telemetry, and artifacts

- [ ] No credentials or generated runtime state are included.
- [ ] Lineage, hashes, manifests, telemetry, or schema outputs were updated when required.
- [ ] Runtime claims are labeled local, static, or production-backed.
- [ ] Any new external side effect is explicitly authorized and documented.

## Risk and rollback

- Risk:
- Rollback/recovery:
- Follow-up issue, if any:

## Agent handoff

- Implementation summary:
- Reviewer focus:
- Independent verification needed:

# Agent guidance for `tests/`

Tests are the executable quality contract. Read the root `AGENTS.md` and the
nearest source guidance before changing them.

## Test priorities

1. Test the changed behavior and its failure/guard path.
2. Keep tests deterministic, offline, and independent of prior local artifacts.
3. Prefer small fixtures built in the test or checked-in deterministic fixtures.
4. Use `tmp_path` for files and isolate environment variables with pytest
   monkeypatching.
5. Assert meaningful values, schemas, statuses, and invariants—not only that
   a function returned without raising.

## ML and trading test rules

- Time-series tests must assert ordering, horizon, and no future leakage.
- Determinism tests should repeat the same input under shuffled row order or a
  fixed seed where that is the contract.
- Promotion tests must cover promote, hold/blocked, rollback, missing metrics,
  non-finite metrics, and canary-window behavior when relevant.
- Artifact tests must check exact paths/metadata and ensure candidates cannot be
  consumed as live models.
- Paper-trading tests must verify next-bar fill timing, costs, state
  idempotency, and that no brokerage/network side effect occurs.
- Lambda tests must cover malformed events, wrong prefixes, version/hash
  mismatches, duplicate delivery, and path traversal.

## Running tests

```bash
pytest -q tests/test_<area>.py
pytest -q -m 'not integration'
pytest -q -m integration
pytest -q
```

The CI unit job uses `pytest -m 'not integration'`; the integration job runs on
pull requests and `main` pushes. Do not mark a test as integration merely to
avoid a deterministic unit gate.

When a test needs a runtime artifact, create it in `tmp_path` and pass its path
explicitly. Never depend on `data/**/latest.*`, `registry/active_model.yaml`,
`mlruns/`, or an existing model in the developer's workspace.

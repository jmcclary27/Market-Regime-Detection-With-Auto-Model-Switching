# Agent guidance for `docs/`

Documentation is part of the system contract. Keep current behavior, planned
architecture, and operational assumptions clearly separated.

- Update docs when a command, path, serialized schema, model safety rule,
  deployment contract, or CI gate changes.
- Link to the source file or test that enforces an important invariant when
  practical.
- Do not claim AWS resources, production telemetry, an MCP server, or automated
  GitHub implementation exists unless the repository and deployment evidence
  support that claim.
- Keep generated reports under runtime output directories; checked-in docs
  should explain how to produce them rather than embedding mutable snapshots.
- For architecture proposals, include status, acceptance criteria, ownership
  boundary, failure behavior, and a rollback/disable path.

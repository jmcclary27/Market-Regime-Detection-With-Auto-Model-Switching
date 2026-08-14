# Agent guidance for `tools/`

The scripts in `tools/` are operational utilities for training, packaging,
replay backfills, fixtures, and project metrics. They can write substantial
artifacts, so agents must inspect arguments and output paths before running
them.

## Safe usage

- Prefer explicit input and output paths. Do not point a test run at the active
  registry or a production/cloud bucket.
- Training utilities should write under `models/candidates/` by default. A
  candidate must not be inferred as active or update a live pointer.
- Keep target construction time-ordered. In particular, build a next-period
  target before filtering rows by regime when that is required to preserve the
  forecast horizon.
- Preserve model IDs, runtime dependency versions, metrics, target summaries,
  and quality-gate results in artifact metadata.
- Packaging must include only validated/published artifacts required by its
  contract and must preserve the active registry pointer and hashes.
- Replay backfills and metrics collection are read-heavy but may create many
  files. Use a disposable copy or explicit output directory during development.

## Verification

For a tool change, run its focused tests plus `ruff check`, `ruff format --check`,
and `mypy` as applicable. Add a dry-run or fixture test when a new command can
otherwise overwrite state or publish an artifact accidentally.

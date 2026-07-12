# Event-driven AWS Lambda inference and paper live simulation

This scaffold moves model inference and paper live simulation, but not model
training, into dedicated event-driven Lambdas in `us-east-1`. Inference runs
the registry-selected global active model plus every distinct discoverable
shadow model from the same version-pinned model bundle. It never follows a
local `latest.parquet` pointer.

No AWS resource has been created or deployed by this change.

## Immutable S3 contract

The Terraform stack creates one private, versioned bucket. Upload objects in
this order:

1. `inference/runs/<run_id>/inputs/<name>.parquet` - the enriched feature /
   regime input current inference consumes. Use `regimes.parquet` for the
   existing pipeline and live-simulation producer.
2. `inference/model-bundles/<bundle_id>.tar.gz` - a bundle made with
   `python -m tools.package_lambda_model_bundle --output <path>`. It contains
   baseline, LightGBM and ARIMA expert, and pretrained shadow artifacts plus
   `registry/active_model.yaml`.
3. `inference/requests/<run_id>/request.json` - uploaded last. This is the
   only prefix that invokes the inference Lambda.

The bucket has versioning enabled. Record the S3 `VersionId` and SHA-256 for
the input and model bundle before creating the request. A live-simulation
request is:

```json
{
  "schema_version": 1,
  "run_id": "20260712_153000Z",
  "mode": "live_sim",
  "inference_ts": 1783860000,
  "target_col": "log_return_1_x",
  "inputs": {
    "inference_input": {
      "key": "inference/runs/20260712_153000Z/inputs/regimes.parquet",
      "version_id": "<s3-version-id>",
      "sha256": "<64-lowercase-hex-digest>"
    },
    "model_bundle": {
      "key": "inference/model-bundles/models-20260712.tar.gz",
      "version_id": "<s3-version-id>",
      "sha256": "<64-lowercase-hex-digest>"
    }
  },
  "live_sim": {
    "strategy_id": "default",
    "bar_timestamp_utc": "2026-07-12T15:30:00Z",
    "row_id": 123,
    "regime": "bullish",
    "open_price": 600.10,
    "close_price": 600.35,
    "queue_signal": true
  }
}
```

Replace `1783860000` with the non-negative Unix timestamp chosen for the run.
`mode` is either `batch` or `live_sim`; both use the same active-plus-shadow
inference path. A `live_sim` request must include the market-bar context shown
above. `row_id` must identify the same enriched-regime row in the inference
input. Set `queue_signal` to `false` for a final session bar when a next-bar
paper fill must not be created.

The Lambda validates request key, S3 versions, checksums, allowed prefixes,
the tarball layout, and the active registry pointer. A batch request writes:

```text
inference/runs/<run_id>/outputs/<request-version-token>/predictions.parquet
inference/runs/<run_id>/outputs/<request-version-token>/inference_run.json
inference/runs/<run_id>/outputs/<request-version-token>/completed.json
```

A `live_sim` request writes under its distinct event prefix:

```text
inference/live-sim/runs/<run_id>/outputs/<request-version-token>/predictions.parquet
inference/live-sim/runs/<run_id>/outputs/<request-version-token>/inference_run.json
inference/live-sim/runs/<run_id>/outputs/<request-version-token>/completed.json
inference/live-sim/runs/<run_id>/outputs/<request-version-token>/live_sim_result.json
```

`completed.json` is the canonical inference handoff. It records exact input
and output object versions and checksums. S3 can deliver an event more than
once; a completion marker for the same request version is returned without
rerunning inference.

The inference runtime enforces the project's published-artifact contract. A
bundle containing only candidate or unpromoted models fails safely instead of
silently producing a live result.

## Paper live-simulation flow

```text
Market-preparation producer
  -> immutable live_sim request.json in S3
  -> inference Lambda (global active plus shadows)
  -> immutable completed.json in S3
  -> serial live-sim Lambda
  -> versioned S3 paper-account state and immutable live_sim_result.json
```

The live-sim executor is a real second Lambda, not a placeholder. It stores
`account_state.json`, `loop_state.json`, `trades.parquet`, and
`equity_curve.parquet` at `live-sim/state/<strategy_id>/`. These are mutable
state projections, but S3 versioning preserves prior state. The canonical
`state_transaction.json` is written first and records the last applied
completion version, so a retry after a partial projection write cannot fill the
same pending signal twice. Its reserved concurrency of one intentionally
serializes one paper account without adding a database bill.

The first closed bar initializes state without a trade. On later bars, it
fills the prior pending paper signal at the supplied current `open_price`,
derives the current signal from the global active prediction and supplied
regime, and queues it for the next received bar. Every input completion gets a
single `live_sim_result.json` idempotency marker, so duplicate S3 events do not
apply another trade. It is paper-only and has no brokerage integration.

The paper executor accepts any globally active model that has passed batch
inference safety checks, including a validated ARIMA JSON artifact. It does
not select a different model by the current regime.

The producer may initially be the existing scheduled feature/regime process,
but it must stop after uploading the immutable request: it must not call local
`run_predictions` or mutate local live-sim state. Replacing market preparation
with a scheduler/market-data Lambda is a later reliability decision; yfinance
rate limits and market-session behavior need validation first.

## Cost controls

- One reserved inference execution uses 2 GB memory, a 5-minute timeout, and
  the included 512 MB temporary storage. The serial paper executor uses 1 GB
  and a 2-minute timeout. Tune after measuring actual ARIMA runtime and bundle
  size.
- ARM64 is the default lower-cost architecture. Use `x86_64` only if a pinned
  native ML wheel cannot run on ARM64.
- The Lambda image installs only inference and paper-execution dependencies;
  MLflow, DVC, market-data, and development tooling stay out of the image.
- There is no VPC, NAT gateway, database, or always-on VM in this stack.
- ECR keeps three images; CloudWatch logs retain 14 days; run/request objects
  retain 90 days; model bundles retain 180 days. Current live-sim state is
  retained and its noncurrent versions expire after 30 days.
- One SQS failure queue stores malformed or failed asynchronous events for 14
  days. Lambda retries once before sending an event there.

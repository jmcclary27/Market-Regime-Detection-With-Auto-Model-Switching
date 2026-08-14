# Frozen Daily Paper-Trading Experiment

This path is separate from the mutable local registry and legacy live simulator.
It compares Buy & Hold, one frozen global model, and frozen HMM-routed experts
using SPY trades and QQQ contextual features.

## Readiness gate

## Freezing a candidate package

Create the immutable local package before configuring any scheduler, Lambda, or
paper-account state. The command requires explicit dates and never derives the
official start from the wall clock. Its HMM input is pre-existing and immutable:
the supplied regime parquet must exactly match labels reproduced by that HMM up
to the inclusive data cutoff.

```bash
python -m tools.freeze_experiment \
  --experiment-id frozen-daily-spy-v1 \
  --official-start-date 2026-09-01 \
  --data-cutoff 2026-08-31 \
  --features-path data/features/latest.parquet \
  --regimes-path data/regimes/latest.parquet \
  --feature-manifest-path data/features/latest.manifest.json \
  --hmm-artifacts-dir models/regimes/hmm \
  --output-dir artifacts/experiment/frozen-daily-spy-v1 \
  --seed 42
```

The output holds a v2 `manifest.json`, the selection scorecard, and a
deterministic `model_bundle.tar.gz`. The archive contains only the selected
global model, three regime specialists, HMM files, and the cutoff-scoped
feature manifest. It has no active-registry pointer, live `latest` path, or
candidate directory. Repeating the command for the same output identity
validates the existing hashes and leaves it byte-identical; different inputs or
dates fail closed.

To publish, add `--publish-s3 --s3-bucket <bucket> --s3-bundle-key
<key> --s3-manifest-key <key>`. The bundle is uploaded first and both objects
must receive non-null S3 VersionIds. The command prints the manifest object
reference for external deployment configuration; an S3 object cannot include
its own returned VersionId and SHA-256 without creating a different version.

Before deployment, create an immutable `experiment/manifest.json` in the
versioned inference bucket. It must contain exact SHA-256 hashes, model IDs,
versions, the HMM artifact, the feature manifest, a data cutoff, and the Git
commit. Use `src.experiment.manifest.freeze_manifest`; a different rewrite is
rejected. Build a matching model bundle with `tools/package_lambda_model_bundle.py`;
the bundle now includes `models/regimes` as well as inference models.

Set `enable_frozen_experiment=true` only with the immutable model bundle key,
VersionId, SHA-256, and an `alpaca_secret_arn`. The secret JSON must be:

```json
{"api_key":"…","api_secret":"…"}
```

The EventBridge Scheduler invokes the producer at 4:20 PM America/New_York on
weekdays. The producer safely skips missing final bars or unknown regimes. It
writes raw data and the enriched input before uploading the inference request;
the request is uploaded last.

The completion triggers the serial experiment executor, which fills yesterday's
targets at today's open, queues new close-derived targets, and writes an
idempotent transaction. It publishes `latest.json` to the private dashboard
bucket. Upload the repository `dashboard/` assets to that bucket before making
the CloudFront URL public.

No automated promotion, rollback, or retraining is part of this experiment.
Treat the 30/60/90-day reports as read-only checkpoints.

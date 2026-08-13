# Frozen Daily Paper-Trading Experiment

This path is separate from the mutable local registry and legacy live simulator.
It compares Buy & Hold, one frozen global model, and frozen HMM-routed experts
using SPY trades and QQQ contextual features.

## Readiness gate

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

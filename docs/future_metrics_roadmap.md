# Future Metrics Roadmap

Phase 1 is now focused on capabilities we can measure from real saved artifacts
in this repo. The table below is the source of truth for what is implemented,
what is partially unlocked, and what still needs more platform work.

| Capability | Status | Unlocked Stats Now | Remaining Work |
| --- | --- | --- | --- |
| Replay audits | Implemented | Exact replay pass rate, semantic replay pass rate, max prediction drift, replay failure breakdown, replay audit coverage by `run_ts` | Backfill more historical runs when legacy lineage points at mutable raw inputs; add scheduled replay cadence if we want continuous checks |
| Deployment event history | Implemented | Promotion counts, hold or blocked counts, promotion precision inputs, rollback-rate inputs, canary-completion inputs, per-source decision history | Add richer rollback markers from live or paper execution so rollback latency can be measured from production-like reversions instead of model-pointer changes alone |
| Data quality audit | Implemented | Missing-bar rate, duplicate-bar rate, stale-bar p95, late-data rate, provider failure count and rate, audit status counts | Extend the same contract to more feeds or providers if we broaden ingestion beyond the current poll paths |
| Infrastructure telemetry | Implemented | Pipeline success rate, pipeline runtime p50 or p95, per-step failure rate, step duration history, mean recovery time between failed and recovered runs | Extend beyond the local pipeline runner into any external scheduler or service orchestration layer we add later |
| Registry change log | Implemented | Active-model tenure, switch frequency, pointer churn, registry change count, deployment-to-pointer-write linkage | Add manual override and approval events into the same stream when a human review workflow exists |
| Richer live or paper execution | Partial | Current live-sim history now reports equity growth and drawdown; decision, model, regime, prediction, and action coverage; signal-to-fill outcomes; position exposure; realized or unrealized PnL; traded notional; fees; and implied fill slippage when the source fields are present | Add benchmark-relative return, external broker-fill reconciliation, multi-asset normalized exposure, and a durable fill-cost decomposition only when those data sources enter the workflow |
| Drift monitoring | Implemented | Per-run regression drift snapshots compare the newest 252 rows with the previous successful batch-inference output: numeric feature mean-shift z-scores, per-model prediction mean or dispersion shifts, and regime-distribution total variation distance | Add alert delivery and acknowledgement history only once an operational alert channel exists; consider longer-term or market-seasonality references after enough immutable inference history accumulates |
| Resource usage collection | Partial | Per-report local managed-storage and model-artifact file or byte snapshots, plus lineage-referenced artifact-size coverage and bytes | Capture CPU and memory per run, remote object-storage usage, and durable time-series resource snapshots when those telemetry sources are available |
| Explicit cost accounting | Planned | None yet | Join resource usage with pricing assumptions to estimate retrain, evaluation, live-loop, and storage cost trends |
| Alerting history | Planned | None yet | Persist alert events, rule ids, timestamps, and acknowledgement data so detection and noise metrics can be measured |
| Calibration or classification outputs | Planned | None yet | Extend prediction artifacts with calibrated probabilities or class labels so accuracy, precision or recall, and calibration metrics can be reported |
| Manual review workflow | Planned | None yet | Add human approval, override, and reason-code logs keyed to promotion candidates and deployment decisions |

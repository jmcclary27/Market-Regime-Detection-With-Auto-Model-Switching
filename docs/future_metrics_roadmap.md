# Future Metrics Roadmap

This document tracks what the project can already measure and what additional
platform or modeling capabilities would unlock stronger MLOps and DevOps
statistics later.

## Available Now

The historical metrics collector already summarizes:

- Lineage-backed run history and subject-run drilldowns
- Git commit and config hash presence rates
- Scorecard MAE and RMSE, including active-versus-baseline deltas
- Per-regime evaluation winners and regime coverage counts
- Walk-forward portfolio metrics such as Sharpe, CAGR, Sortino, max drawdown,
  turnover, and profit factor
- Promotion decisions, challenger/incumbent comparisons, and promotion-guard
  outcomes
- Regime diagnostics such as entropy, duration, switch frequency, occupancy, and
  confidence quantiles
- Live-sim portfolio growth, drawdown, trade counts, and model or regime
  coverage
- Engineering inventory signals such as saved scorecards, saved lineage files,
  test file count, and test case count

## Future Capability -> Unlocked Stats

| Missing Capability | Unlocked Stats | Required Artifact or Source |
| --- | --- | --- |
| Data quality audit stage | Missing-bar rate, duplicate-bar rate, stale-bar p95, late-data rate, provider failure rate | Timestamped data quality report per ingestion run |
| Replay or reproducibility audit stage | Exact replay pass rate, semantic replay pass rate, max prediction drift on replay, replay failure breakdown | Replay summary JSON or parquet keyed by `run_ts` |
| Feature and prediction drift monitor | PSI or KL drift by feature, prediction drift by model, regime-distribution drift, alert lead time | Drift snapshots per run or per day |
| Deployment event history with explicit rollback records | Promotion precision, rollback rate, rollback latency, challenger survival time, canary completion rate | Deployment events table with decision timestamps and rollback markers |
| Richer live or paper execution engine | Average holding time, exposure %, realized versus unrealized PnL, slippage cost, fee impact, benchmark-relative return | Trade ledger with fills, positions, costs, and benchmark series |
| Infrastructure and orchestration telemetry | Pipeline success rate, p50 or p95 runtime, per-step failure rate, mean recovery time, on-call style SLO compliance | Structured pipeline run log or metrics export |
| Resource usage collection | CPU peak, memory peak, storage growth, model artifact size growth, runtime cost proxies | Per-run resource metrics snapshot |
| Explicit cost accounting | Cost per retrain, cost per evaluation sweep, cost per 100 live-sim loops, monthly storage cost trend | Cost report or pricing snapshot joined to resource usage |
| Alerting and monitoring history | Alert volume, false-positive rate, time-to-detect degradation, time-to-acknowledge, noisy-rule count | Alert event log with rule ids and timestamps |
| Model calibration or classification outputs | Directional accuracy, calibration error, precision or recall, confusion by regime | Extended predictions artifact with class labels or calibrated probabilities |
| Registry change log | Active model tenure, switch frequency over time, registry churn rate, rollback-after-promotion rate | Versioned registry history or pointer event stream |
| Manual review or approval workflow | Human override rate, approval lead time, reason-code distribution, blocked-promotion rate | Review audit log keyed by candidate and run |

## Recommended Order

The highest-value additions for recruiter-facing evidence are:

1. Replay or reproducibility audits
2. Deployment event history with rollback metrics
3. Data quality SLAs
4. Infrastructure runtime and failure SLOs
5. Richer execution realism and cost accounting

# Market Regime Detection & Auto-Model Switching

A cost-conscious, local-first MLOps project focused on building an end-to-end
system for market regime detection and automatic model switching.

This project emphasizes **MLOps architecture and system design** over pure
trading alpha.

---

## Project Goals

- Periodically poll market data on a schedule
- Detect market regimes using rule-based methods first, HMM later
- Maintain multiple expert models, one per regime
- Run shadow predictions across all models
- Automatically switch the active model using canary and rollback logic
- Operate primarily on a VM with near-zero cloud cost

---

## Running the Project (Current Stub)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run a Markey Poll

This command fetches a small slice of market data, writes a timestamped CSV
to disk, updates a latest pointer, and logs a JSON run record.

```bash
python -m src.jobs.poll_market_data
```
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

## Run a Market Poll

This command fetches a small slice of market data, writes a timestamped CSV
to disk, updates a latest pointer, and logs a JSON run record.

```bash
python -m src.jobs.poll_market_data
```

## Run Feature Pipeline (v0)

This command builds deterministic features from a provided bars file and writes
a parquet artifact and manifest to disk.

```bash
python -m src.features.run_features --input <bars.csv|bars.parquet> --timestamp <timestamp>
```

Outputs:

- data/features/<timestamp>.parquet
- data/features/<timestamp>.manifest.json (schema + content hash)

## Run Regime Detection (v0, rule-based)

This command labels each row with a simple rule-based market regime and writes
a timestamped parquet artifact to disk. The output includes a
`regime_explanation` column for debugging.

```bash
python -m src.regimes.run_regime_detection
```

Outputs:

- data/regimes/regimes_<timestamp>.parquet

Output columns (minimum):

- timestamp
- symbol
- regime
- regime_explanation

## Run Machine Learning Parts

- Generate fixture data

```bash
python tools/make_training_fixture.py
```

- Create the pretrained expert artifact

```bash
python tools/make_pretrained_expert.py
```

- Run training

```bash
python -m src.models.train
```

## Batch Inference & Shadow Predictions

This project supports **batch inference across all available models** (baseline, regime experts, and pretrained models) to produce **shadow predictions** for comparison and monitoring.

### Purpose
- Run inference for *every* model on the same feature set
- Enable side-by-side comparison between active and shadow models
- Provide the data needed for future **model selection and auto-switching**

### How to Run

```bash
python -m src.inference.batch_predict
```

## Evaluator + Scorecards

This project includes an evaluator that compares all models, overall and per market regime, and writes a scorecard artifact that will be consumed by the auto-switching logic.

### Inputs

The evaluator expects the latest pipeline artifacts:

- `data/features/latest.parquet`
- `data/regimes/latest.parquet`
- `data/predictions/latest.parquet`

Notes:
- `data/predictions/latest.parquet` is long-form and contains `row_id`, `model_name`, and `y_pred`.
- `data/features/latest.parquet` and `data/regimes/latest.parquet` do not contain `row_id`, so the evaluator reconstructs it deterministically by sorting by `timestamp` then `symbol`, then setting `row_id = index`.

### Run evaluator

```bash
python -m src.eval.run_evaluator
```
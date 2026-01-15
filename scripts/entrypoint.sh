#!/bin/sh
set -eu

cmd="${1:-pipeline}"
shift || true

# Match pipeline timestamp behavior (RUN_TS or generated inside pipeline)
# You can optionally pass RUN_TS env var when calling docker compose.

case "$cmd" in
  pipeline)
    exec python -m src.pipeline.run "$@"
    ;;

  poll)
    # Equivalent to pipeline step("poll", ingest_run)
    exec python -c "from src.ingestion.run_ingestion import run; run()"
    ;;

  features)
    # Equivalent to pipeline feature step:
    # 1) find latest raw CSV in DATA_DIR/raw
    # 2) call features_run(input_path=raw_latest, timestamp=RUN_TS or now)
    exec python -c "
from pathlib import Path
import os
from datetime import datetime, UTC
from src.features.run_features import run

project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd())).resolve()
data_dir = Path(os.getenv('DATA_DIR', project_root / 'data')).resolve()
raw_dir = data_dir / 'raw'

candidates = sorted(raw_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f'No raw CSV files found in {raw_dir}')
raw_latest = candidates[0]

run_ts = os.getenv('RUN_TS') or datetime.now(UTC).strftime('%Y%m%d_%H%M%SZ')
run(input_path=raw_latest, timestamp=run_ts)
"
    ;;

  regimes)
    # Uses the latest features parquet under DATA_DIR/features
    exec python -c "
from pathlib import Path
import os
from datetime import datetime, UTC
from src.regimes.run_regime_detection import run

project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd())).resolve()
data_dir = Path(os.getenv('DATA_DIR', project_root / 'data')).resolve()
feat_dir = data_dir / 'features'

candidates = sorted(feat_dir.glob('*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f'No features parquet files found in {feat_dir}')
features_latest = candidates[0]

run_ts = os.getenv('RUN_TS') or datetime.now(UTC).strftime('%Y%m%d_%H%M%SZ')
run(input_path=features_latest, timestamp=run_ts)
"
    ;;

  predict)
    # Equivalent to pipeline call: run_stage(features_path=features_parquet)
    # Uses latest features parquet under DATA_DIR/features
    exec python -c "
from pathlib import Path
import os
from src.inference.batch_predict import run_stage

project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd())).resolve()
data_dir = Path(os.getenv('DATA_DIR', project_root / 'data')).resolve()
feat_dir = data_dir / 'features'

candidates = sorted(feat_dir.glob('*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f'No features parquet files found in {feat_dir}')
features_latest = candidates[0]

run_stage(features_path=features_latest)
"
    ;;

  evaluate|eval)
    # Equivalent to pipeline step("eval", eval_run)
    exec python -c "from src.eval.run_evaluator import run; run()"
    ;;

  switch)
    # Equivalent to pipeline step("switch", switch_run)
    exec python -c "from src.deploy.switcher import run; run()"
    ;;

  train)
    # Your training script is standalone
    exec python -m src.models.train "$@"
    ;;

  test)
    exec pytest -q "$@"
    ;;

  ruff)
    exec ruff check . "$@"
    ;;

  mypy)
    exec mypy "$@"
    ;;

  *)
    exec "$cmd" "$@"
    ;;
esac

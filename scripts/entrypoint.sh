#!/bin/bash
set -euo pipefail

cmd="${1:-demo}"
shift || true

# S3 is deliberately opt-in so the default demo never needs cloud credentials.
if [ "${S3_SYNC_ENABLED:-false}" = "true" ]; then
  if [ -z "${ARTIFACT_BUCKET:-}" ]; then
    echo "S3_SYNC_ENABLED=true requires ARTIFACT_BUCKET to be set." >&2
    exit 2
  fi

  echo "Syncing artifacts from S3 bucket: $ARTIFACT_BUCKET"
  mkdir -p /app/data/raw /app/models
  aws s3 sync "s3://${ARTIFACT_BUCKET}/data/raw" /app/data/raw
  aws s3 sync "s3://${ARTIFACT_BUCKET}/models" /app/models
fi

case "$cmd" in
  demo)
    exec python -m src.demo.run "$@"
    ;;
  pipeline)
    exec python -m src.pipeline.run "$@"
    ;;
  poll)
    exec python -c "from src.ingestion.run_ingestion import run; run()"
    ;;
  features)
    exec python - <<'PYCODE'
from datetime import UTC, datetime
import os
from pathlib import Path

from src.features.run_features import run

project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
data_dir = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()
candidates = sorted((data_dir / "raw").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f"No raw CSV files found in {data_dir / 'raw'}")
run(input_path=candidates[0], timestamp=os.getenv("RUN_TS") or datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ"))
PYCODE
    ;;
  regimes)
    exec python - <<'PYCODE'
from datetime import UTC, datetime
import os
from pathlib import Path

from src.regimes.run_regime_detection import run

project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
data_dir = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()
candidates = sorted((data_dir / "features").glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f"No features parquet files found in {data_dir / 'features'}")
run(input_path=candidates[0], timestamp=os.getenv("RUN_TS") or datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ"))
PYCODE
    ;;
  predict)
    exec python - <<'PYCODE'
import os
from pathlib import Path

from src.inference.batch_predict import run_stage

project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
data_dir = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()
candidates = sorted((data_dir / "regimes").glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f"No regimes parquet files found in {data_dir / 'regimes'}")
run_stage(features_path=candidates[0])
PYCODE
    ;;
  evaluate|eval)
    exec python -c "from src.eval.run_evaluator import run; run()"
    ;;
  switch)
    exec python -c "from src.deploy.switcher import run; run()"
    ;;
  train)
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
  live-sim)
    exec python -m src.trading.live_sim "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-pipeline}"
shift || true

case "$cmd" in
  pipeline)  exec python -m src.pipeline.run "$@" ;;
  poll)      exec python -m src.data.poll "$@" ;;
  features)  exec python -m src.features.build "$@" ;;
  regimes)   exec python -m src.regimes.detect "$@" ;;
  train)     exec python -m src.models.train "$@" ;;
  predict)   exec python -m src.inference.batch_predict "$@" ;;
  evaluate)  exec python -m src.eval.run_evaluator "$@" ;;
  switch)    exec python -m src.switching.run_switcher "$@" ;;
  test)      exec pytest -q "$@" ;;
  ruff)      exec ruff check . "$@" ;;
  mypy)      exec mypy "$@" ;;
  *)         exec "$cmd" "$@" ;;
esac

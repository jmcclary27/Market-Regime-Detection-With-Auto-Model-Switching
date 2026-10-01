#!/bin/bash
set -euo pipefail

# This project name scopes cleanup to the demo's named volume only.
project_name="${RECRUITER_DEMO_PROJECT:-market-regime-recruiter-demo}"
compose=(docker compose --project-name "$project_name")

"${compose[@]}" down --volumes --remove-orphans || true
"${compose[@]}" build
"${compose[@]}" run --rm market demo
"${compose[@]}" up -d dashboard

for _ in $(seq 1 30); do
  if curl --fail --silent --show-error http://localhost:8501/_stcore/health >/dev/null; then
    echo "Recruiter dashboard is ready at http://localhost:8501"
    echo "Stop and remove the isolated demo state with:"
    echo "  docker compose --project-name $project_name down --volumes --remove-orphans"
    exit 0
  fi
  sleep 1
done

"${compose[@]}" logs dashboard
echo "Dashboard did not become healthy at http://localhost:8501" >&2
exit 1

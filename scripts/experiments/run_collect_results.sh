#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-${RESULTS_ROOT:-}}
if [[ -z "${EXPERIMENTS_ROOT}" ]]; then
  echo "[FATAL] Set EXPERIMENTS_ROOT (or RESULTS_ROOT) to your experiment results directory." >&2
  echo "Example: export EXPERIMENTS_ROOT=\$PROJECT_ROOT/runs/experiments" >&2
  exit 2
fi
OUT_DIR_DEFAULT="$EXPERIMENTS_ROOT/collection"
if [[ -n "$RUN_ROOT" ]]; then
  OUT_DIR_DEFAULT="$RUN_ROOT/collection"
fi
OUT_DIR=${OUT_DIR:-"$OUT_DIR_DEFAULT"}
OUT_CSV=${OUT_CSV:-"$OUT_DIR/results_summary.csv"}

if [[ ! -d "$EXPERIMENTS_ROOT" ]]; then
  echo "[FATAL] EXPERIMENTS_ROOT not found: $EXPERIMENTS_ROOT" >&2
  echo "Set EXPERIMENTS_ROOT (or RESULTS_ROOT) correctly and rerun." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

python "$PROJECT_ROOT/scripts/experiments/collect_results.py" \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out-csv "$OUT_CSV"

echo "Collector output: $OUT_CSV"

#!/bin/bash
#SBATCH --job-name=vitaltrace-collect
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=out/%x-%j.out
set -euo pipefail

# Required:
#   PROJECT_ROOT
#   JOBS_TSV
#   OUT_CSV

PROJECT_ROOT=${PROJECT_ROOT:?set PROJECT_ROOT}
JOBS_TSV=${JOBS_TSV:?set JOBS_TSV}
OUT_CSV=${OUT_CSV:?set OUT_CSV}

cd "$PROJECT_ROOT"
python scripts/experiments/collect_results.py \
  --jobs-tsv "$JOBS_TSV" \
  --out-csv "$OUT_CSV"

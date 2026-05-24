#!/bin/bash
#SBATCH --job-name=vitaltrace-s3-full1000
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --output=out/%x-%j.out
set -euo pipefail

# User-provided environment:
#   PROJECT_ROOT
#   DATASET=[mimic|eicu]
#   STAGE2_DIR
#   OUT_DIR
# Optional:
#   HORIZONS, TOP_K, WINDOW_STEPS

PROJECT_ROOT=${PROJECT_ROOT:?set PROJECT_ROOT}
DATASET=${DATASET:?set DATASET}
STAGE2_DIR=${STAGE2_DIR:?set STAGE2_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}

HORIZONS=${HORIZONS:-6,12}
TOP_K=${TOP_K:-1000}
WINDOW_STEPS=${WINDOW_STEPS:-20}

cd "$PROJECT_ROOT"
python -m src.preprocess_pipeline.signal_pipeline_stage3 \
  --dataset "$DATASET" \
  --stage2-cohort "$STAGE2_DIR/stage2_cohort.parquet" \
  --stage2-events "$STAGE2_DIR/stage2_events.parquet" \
  --out-dir "$OUT_DIR" \
  --horizons "$HORIZONS" \
  --simplified-top-k "$TOP_K" \
  --meaningful-steps-per-encounter "$WINDOW_STEPS" \
  --qc-strict

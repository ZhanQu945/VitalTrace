#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/preprocess}

HORIZONS=${HORIZONS:-6,12}
TOP_K=${TOP_K:-1000}
WINDOW_STEPS=${WINDOW_STEPS:-20}

mkdir -p "$RUN_ROOT/mimic_stage3" "$RUN_ROOT/eicu_stage3"

python -m src.preprocess_pipeline.signal_pipeline_stage3 \
  --dataset mimic \
  --stage2-cohort "$RUN_ROOT/mimic_stage2/stage2_cohort.parquet" \
  --stage2-events "$RUN_ROOT/mimic_stage2/stage2_events.parquet" \
  --out-dir "$RUN_ROOT/mimic_stage3" \
  --horizons "$HORIZONS" \
  --simplified-top-k "$TOP_K" \
  --meaningful-steps-per-encounter "$WINDOW_STEPS" \
  --qc-strict

python -m src.preprocess_pipeline.signal_pipeline_stage3 \
  --dataset eicu \
  --stage2-cohort "$RUN_ROOT/eicu_stage2/stage2_cohort.parquet" \
  --stage2-events "$RUN_ROOT/eicu_stage2/stage2_events.parquet" \
  --out-dir "$RUN_ROOT/eicu_stage3" \
  --horizons "$HORIZONS" \
  --simplified-top-k "$TOP_K" \
  --meaningful-steps-per-encounter "$WINDOW_STEPS" \
  --qc-strict

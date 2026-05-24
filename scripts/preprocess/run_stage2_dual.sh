#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/preprocess}
MIMIC_ROOT=${MIMIC_ROOT:-}
EICU_ROOT=${EICU_ROOT:-}

if [[ -z "$MIMIC_ROOT" || -z "$EICU_ROOT" ]]; then
  echo "[FATAL] MIMIC_ROOT and EICU_ROOT must be set." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/mimic_stage2" "$RUN_ROOT/eicu_stage2"

python -m src.preprocess_pipeline.signal_pipeline_stage2 \
  --dataset mimic \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --stage1-cohort "$RUN_ROOT/mimic_stage1/stage1_cohort.parquet" \
  --stage1-interventions "$RUN_ROOT/mimic_stage1/stage1_events_interventions.parquet" \
  --out-dir "$RUN_ROOT/mimic_stage2"

python -m src.preprocess_pipeline.signal_pipeline_stage2 \
  --dataset eicu \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --stage1-cohort "$RUN_ROOT/eicu_stage1/stage1_cohort.parquet" \
  --stage1-interventions "$RUN_ROOT/eicu_stage1/stage1_events_interventions.parquet" \
  --out-dir "$RUN_ROOT/eicu_stage2"

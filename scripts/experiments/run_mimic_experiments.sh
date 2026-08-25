#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/preprocess}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-${EXP_ROOT:-$PROJECT_ROOT/runs/experiments/mimic_$(date +%Y%m%d_%H%M%S)}}
PROTOCOL_JSON=${PROTOCOL_JSON:-$PROJECT_ROOT/data/global_protocol_manual.json}
INPUT_JSONL=${INPUT_JSONL:-$RUN_ROOT/mimic_stage3/stage3_labeled_h6.jsonl}
MODEL_KEY=${MODEL_KEY:-llama8b}

case "$MODEL_KEY" in
  llama8b)
    LLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-256}
    LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-2048}
    ;;
  llama70b)
    LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
    LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-320}
    LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-1536}
    ;;
  mixtral8x7b)
    LLM_MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-256}
    LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-1536}
    ;;
  clinicalcamel70b)
    LLM_MODEL_ID="wanglab/ClinicalCamel-70B"
    LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-256}
    LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-1536}
    ;;
  meditron70b)
    LLM_MODEL_ID="epfl-llm/meditron-70b"
    LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-256}
    LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-1536}
    ;;
  *)
    echo "[FATAL] Unsupported MODEL_KEY=$MODEL_KEY"
    echo "Allowed MODEL_KEY: llama8b | llama70b | mixtral8x7b | clinicalcamel70b | meditron70b"
    exit 2
    ;;
esac

mkdir -p "$EXPERIMENTS_ROOT/configs" "$EXPERIMENTS_ROOT"

CFG_PATH="$EXPERIMENTS_ROOT/configs/vitaltrace_${MODEL_KEY}.yaml"
OUT_DIR="$EXPERIMENTS_ROOT/vitaltrace_${MODEL_KEY}"

cat > "$CFG_PATH" <<YAML
io:
  input_jsonl: $INPUT_JSONL
  out_dir: $OUT_DIR
  protocol_json: $PROTOCOL_JSON
runtime:
  agent_backend: llm
  max_rules: 3
  runner_mode: temporal_loop
  reuse_stage_outputs: false
  max_audit_retries: 1
  fail_policy: conservative_continue
  ablation: none
model:
  llm_model_id: $LLM_MODEL_ID
  llm_max_new_tokens: $LLM_MAX_NEW_TOKENS
  llm_temperature: 0.1
  llm_max_input_tokens: $LLM_MAX_INPUT_TOKENS
YAML

python -m src.latent_pipeline.run_staged_from_config --config "$CFG_PATH"
python -m src.latent_pipeline.evaluate_staged --out-dir "$OUT_DIR" --protocol-json "$PROTOCOL_JSON"
python -m src.latent_pipeline.counterfactual_runner \
  --out-dir "$OUT_DIR" \
  --protocol-json "$PROTOCOL_JSON" \
  --agent-backend llm \
  --llm-model-id "$LLM_MODEL_ID" \
  --llm-max-new-tokens "$LLM_MAX_NEW_TOKENS" \
  --llm-temperature 0.1 \
  --llm-max-input-tokens "$LLM_MAX_INPUT_TOKENS" \
  --max-rules 3

echo "saved experiment outputs in: $OUT_DIR"

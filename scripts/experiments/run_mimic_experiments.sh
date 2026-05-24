#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/preprocess}
EXP_ROOT=${EXP_ROOT:-$PROJECT_ROOT/runs/experiments/mimic_$(date +%Y%m%d_%H%M%S)}
PROTOCOL_JSON=${PROTOCOL_JSON:-$PROJECT_ROOT/data/global_protocol_manual.json}
INPUT_JSONL=${INPUT_JSONL:-$RUN_ROOT/mimic_stage3_full1000/stage3_labeled_h6.jsonl}

mkdir -p "$EXP_ROOT/configs" "$EXP_ROOT"

cat > "$EXP_ROOT/configs/vitaltrace_llama8b.yaml" <<YAML
io:
  input_jsonl: $INPUT_JSONL
  out_dir: $EXP_ROOT/vitaltrace_llama8b
  protocol_json: $PROTOCOL_JSON
runtime:
  agent_backend: llm
  max_rules: 3
  runner_mode: temporal_loop
  max_audit_retries: 1
  fail_policy: conservative_continue
  ablation: none
model:
  llm_model_id: meta-llama/Llama-3.1-8B-Instruct
  llm_max_new_tokens: 256
  llm_temperature: 0.1
  llm_max_input_tokens: 2048
YAML

python -m src.latent_pipeline.run_staged_from_config --config "$EXP_ROOT/configs/vitaltrace_llama8b.yaml"
python -m src.latent_pipeline.evaluate_staged --out-dir "$EXP_ROOT/vitaltrace_llama8b" --protocol-json "$PROTOCOL_JSON"
python -m src.latent_pipeline.counterfactual_runner --out-dir "$EXP_ROOT/vitaltrace_llama8b" --protocol-json "$PROTOCOL_JSON"

echo "saved experiment outputs in: $EXP_ROOT/vitaltrace_llama8b"

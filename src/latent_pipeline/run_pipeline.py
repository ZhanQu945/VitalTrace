import argparse
import os

from src.latent_pipeline.common import log
from src.latent_pipeline.stage1_router import run as run_router
from src.latent_pipeline.stage2_reasoner import run as run_reasoner
from src.latent_pipeline.stage3_auditor import run as run_auditor
from src.latent_pipeline.stage4_steward import run as run_steward


def main(input_jsonl: str, protocol_path: str, output_dir: str, max_rules: int, agent_backend: str, llm_model_id: str, llm_max_new_tokens: int, llm_temperature: float):
    os.makedirs(output_dir, exist_ok=True)

    s1 = os.path.join(output_dir, "stage1_router.jsonl")
    s2 = os.path.join(output_dir, "stage2_reasoner.jsonl")
    s3 = os.path.join(output_dir, "stage3_auditor.jsonl")
    s4 = os.path.join(output_dir, "stage4_steward.jsonl")

    run_router(input_jsonl, protocol_path, s1, os.path.join(output_dir, "metrics_router.json"), max_rules=max_rules, agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature)
    run_reasoner(s1, s2, os.path.join(output_dir, "metrics_reasoner.json"), agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature)
    run_auditor(s2, s3, os.path.join(output_dir, "metrics_auditor.json"), agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature)
    run_steward(s3, s4, os.path.join(output_dir, "metrics_steward.json"), agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature)
    log(f"Pipeline done. Final: {s4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--protocol-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-rules", type=int, default=3)
    ap.add_argument("--agent-backend", choices=["deterministic", "llm"], default=os.environ.get("AGENT_BACKEND", "deterministic"))
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    ap.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    args = ap.parse_args()
    main(args.input_jsonl, args.protocol_path, args.output_dir, args.max_rules, args.agent_backend, args.llm_model_id, args.llm_max_new_tokens, args.llm_temperature)

from __future__ import annotations

import argparse
import json
import os

from src.config.loader import load_config
from src.latent_pipeline.common import iter_jsonl, write_jsonl
from src.latent_pipeline.inference_context import assert_no_future_fields
from src.latent_pipeline.temporal_loop_runner import run as run_temporal_loop
from src.latent_pipeline.stage1_router import run as run_stage1
from src.latent_pipeline.stage2_reasoner import run as run_stage2
from src.latent_pipeline.stage3_auditor import run as run_stage3
from src.latent_pipeline.stage4_steward import run as run_stage4


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    io = cfg["io"]
    model = cfg.get("model", {})
    runtime = cfg.get("runtime", {})

    agent_backend = runtime.get("agent_backend", "llm")
    llm_model_id = model.get("llm_model_id", "meta-llama/Llama-3.1-8B-Instruct")
    llm_max_new_tokens = int(model.get("llm_max_new_tokens", 256))
    llm_temperature = float(model.get("llm_temperature", 0.1))
    llm_max_input_tokens = int(model.get("llm_max_input_tokens", 3072))
    max_rules = int(runtime.get("max_rules", 3))
    runner_mode = str(runtime.get("runner_mode", "flat_staged")).lower()
    max_audit_retries = int(runtime.get("max_audit_retries", 1))
    fail_policy = str(runtime.get("fail_policy", "conservative_continue"))
    ablation = str(runtime.get("ablation", "none")).lower()
    reuse_stage_outputs = bool(runtime.get("reuse_stage_outputs", False))

    s1 = f"{io['out_dir']}/stage1_router.jsonl"
    s2 = f"{io['out_dir']}/stage2_reasoner.jsonl"
    s3 = f"{io['out_dir']}/stage3_auditor.jsonl"
    s4 = f"{io['out_dir']}/stage4_steward.jsonl"

    protocol_json = io["protocol_json"]
    if ablation == "no_global_protocol":
        os.makedirs(io["out_dir"], exist_ok=True)
        protocol_json = f"{io['out_dir']}/ablation_empty_protocol.json"
        with open(protocol_json, "w") as f:
            json.dump({}, f)
    if ablation == "no_router":
        max_rules = 0

    def _validate_reused_stage(path: str) -> None:
        for row_number, ex in enumerate(iter_jsonl(path), start=1):
            assert_no_future_fields(
                ex.get("packet", {}),
                f"reused stage packet {path}:{row_number}",
            )

    def _make_pass_through_auditor(in_jsonl: str, out_jsonl: str):
        rows = []
        for ex in iter_jsonl(in_jsonl):
            ex["audit"] = {"status": "PASS", "issues": [], "suggested_fixes": [], "ablation": "no_auditor"}
            ex["stage3_prediction"] = {"audit_status": "PASS", "n_issues": 0}
            rows.append(ex)
        write_jsonl(out_jsonl, rows)

    def _make_no_memory_steward(in_jsonl: str, out_jsonl: str):
        rows = []
        for ex in iter_jsonl(in_jsonl):
            zero = {
                "hemodynamic_state": 0,
                "respiratory_state": 0,
                "renal_state": 0,
                "metabolic_state": 0,
                "systemic_inflammation_state": 0,
                "active_protocol_prediction": [],
            }
            ex["individual_protocol_state_prev"] = dict(zero)
            ex["individual_protocol_state_next"] = dict(zero)
            ex["individual_protocol_state_delta"] = {
                "hemodynamic_state": 0,
                "respiratory_state": 0,
                "renal_state": 0,
                "metabolic_state": 0,
                "systemic_inflammation_state": 0,
            }
            ex["stage4_prediction"] = {"state_next": dict(zero), "state_delta": ex["individual_protocol_state_delta"], "ablation": "no_memory"}
            rows.append(ex)
        write_jsonl(out_jsonl, rows)

    if runner_mode == "temporal_loop":
        run_temporal_loop(
            input_jsonl=io["input_jsonl"],
            protocol_path=protocol_json,
            out_dir=io["out_dir"],
            max_rules=max_rules,
            max_audit_retries=max_audit_retries,
            fail_policy=fail_policy,
            agent_backend=agent_backend,
            llm_model_id=llm_model_id,
            llm_max_new_tokens=llm_max_new_tokens,
            llm_temperature=llm_temperature,
            llm_max_input_tokens=llm_max_input_tokens,
        )
    else:
        m1 = f"{io['out_dir']}/metrics_router.json"
        m2 = f"{io['out_dir']}/metrics_reasoner.json"
        m3 = f"{io['out_dir']}/metrics_auditor.json"
        m4 = f"{io['out_dir']}/metrics_steward.json"

        if reuse_stage_outputs and os.path.exists(s1):
            _validate_reused_stage(s1)
        else:
            run_stage1(io["input_jsonl"], protocol_json, s1, m1, max_rules=max_rules, agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature, llm_max_input_tokens=llm_max_input_tokens)

        if reuse_stage_outputs and os.path.exists(s2):
            _validate_reused_stage(s2)
        else:
            run_stage2(s1, s2, m2, agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature, llm_max_input_tokens=llm_max_input_tokens)

        if ablation == "no_auditor":
            if not (reuse_stage_outputs and os.path.exists(s3)):
                _make_pass_through_auditor(s2, s3)
            else:
                _validate_reused_stage(s3)
        else:
            if not (reuse_stage_outputs and os.path.exists(s3)):
                run_stage3(s2, s3, m3, agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature, llm_max_input_tokens=llm_max_input_tokens)
            else:
                _validate_reused_stage(s3)

        if ablation == "no_memory":
            if not (reuse_stage_outputs and os.path.exists(s4)):
                _make_no_memory_steward(s3, s4)
            else:
                _validate_reused_stage(s4)
        else:
            if not (reuse_stage_outputs and os.path.exists(s4)):
                run_stage4(s3, s4, m4, agent_backend=agent_backend, llm_model_id=llm_model_id, llm_max_new_tokens=llm_max_new_tokens, llm_temperature=llm_temperature, llm_max_input_tokens=llm_max_input_tokens)
            else:
                _validate_reused_stage(s4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ns = ap.parse_args()
    main(ns.config)

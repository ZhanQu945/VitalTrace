import argparse
import os
import time
from typing import Dict, List

from src.latent_pipeline.common import iter_jsonl, write_jsonl, write_json, log
from src.latent_pipeline.protocol_utils import load_protocol, feature_map_from_facts, rule_score
from src.latent_pipeline.prompts import router_system_prompt, router_user_prompt


def _det_router(rules: Dict, facts: List[Dict], max_rules: int):
    fmap = feature_map_from_facts(facts)
    scored = []
    for rid, rule in rules.items():
        ok, sc = rule_score(rule, fmap)
        if ok:
            scored.append((rid, sc, rule))
    scored.sort(key=lambda x: x[1], reverse=True)
    sel = [x[0] for x in scored[:max_rules]]
    active = {rid: rules[rid] for rid in sel if rid in rules}
    return sel, active


def run(input_jsonl: str, protocol_path: str, output_jsonl: str, metrics_json: str, max_rules: int = 3, agent_backend: str = "deterministic", llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct", llm_max_new_tokens: int = 256, llm_temperature: float = 0.1, llm_max_input_tokens: int = 3072):
    rules = load_protocol(protocol_path)
    rows = []
    with_selection = 0
    severity_high = 0

    llm = None
    if agent_backend == "llm":
        from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig
        llm = LLMBackend(LLMConfig(model_id=llm_model_id, max_new_tokens=llm_max_new_tokens, temperature=llm_temperature, max_input_tokens=llm_max_input_tokens))

    rules_brief = [{"rule_id": rid, "name": r.get("name", ""), "risk": r.get("risk", ""), "trigger": r.get("trigger", [])} for rid, r in rules.items()]
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "50"))
    t0 = time.time()
    n_seen = 0
    seen_example_ids = {}

    for ex in iter_jsonl(input_jsonl):
        n_seen += 1
        facts = ex.get("protocol_observations", [])
        packet = {
            "facts": facts,
            "latent_state": ex.get("latent_state_current", {}),
            "risk_targets": ex.get("targets", {}),
            "counterfactual_candidates": ex.get("counterfactual_candidates", []),
        }

        if llm is None:
            sel, active = _det_router(rules, facts, max_rules)
        else:
            out = llm.generate_json(router_system_prompt(), router_user_prompt(packet, rules_brief, max_rules=max_rules))
            sel = [rid for rid in out.get("selected_rule_ids", []) if rid in rules][:max_rules]
            if not sel:
                sel, _ = _det_router(rules, facts, max_rules)
            active = {rid: rules[rid] for rid in sel if rid in rules}

        if sel:
            with_selection += 1
        severity_high += sum(1 for rid in sel if str(rules[rid].get("severity", "")).lower() == "high")

        step_part = ex.get("step_id", ex.get("step_index", 0))
        try:
            step_part = int(step_part)
        except Exception:
            step_part = 0
        base_example_id = ex.get(
            "example_id",
            f"{ex.get('source_dataset')}_{ex.get('patient_id')}_{ex.get('encounter_id')}_{ex.get('anchor_time')}_{step_part}",
        )
        k = str(base_example_id)
        if k in seen_example_ids:
            seen_example_ids[k] += 1
            example_id = f"{k}__dup{seen_example_ids[k]}"
        else:
            seen_example_ids[k] = 0
            example_id = k

        rows.append({
            "record_id": int(n_seen),
            "example_id": example_id,
            "source_dataset": ex.get("source_dataset"),
            "patient_id": ex.get("patient_id"),
            "encounter_id": ex.get("encounter_id"),
            "anchor_time": ex.get("anchor_time"),
            "counterfactual_candidates": ex.get("counterfactual_candidates", []),
            "packet": packet,
            "selected_rule_ids": sel,
            "active_rules": active,
            "ground_truth_targets": ex.get("targets", {}),
            "stage1_prediction": {
                "selected_rule_ids": sel,
                "n_selected_rules": len(sel),
                "active_rule_ids": list(active.keys()),
            },
            "stage1_ground_truth": ex.get("targets", {}),
        })
        if n_seen % progress_every == 0:
            dt = max(1e-6, time.time() - t0)
            log(f"Router progress: {n_seen} rows ({n_seen/dt:.2f} rows/s), llm_failures={int(getattr(llm, '_failures', 0)) if llm else 0}")

    write_jsonl(output_jsonl, rows)
    pred_gt_path = os.path.join(os.path.dirname(output_jsonl), "stage1_predictions_ground_truth.jsonl")
    pred_gt_rows = [{"example_id": r.get("example_id"), "source_dataset": r.get("source_dataset"), "patient_id": r.get("patient_id"), "encounter_id": r.get("encounter_id"), "anchor_time": r.get("anchor_time"), "stage1_prediction": r.get("stage1_prediction", {}), "stage1_ground_truth": r.get("stage1_ground_truth", {})} for r in rows]
    write_jsonl(pred_gt_path, pred_gt_rows)
    n = len(rows)
    write_json(metrics_json, {"n_examples": n, "selection_rate": (with_selection / n) if n else 0.0, "avg_high_severity_selected": (severity_high / n) if n else 0.0, "agent_backend": agent_backend, "llm_model_id": llm_model_id if llm else None, "llm_calls": int(getattr(llm, "_calls", 0)) if llm else 0, "llm_failures": int(getattr(llm, "_failures", 0)) if llm else 0})
    log(f"Router stage done: n={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--protocol-path", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--max-rules", type=int, default=3)
    ap.add_argument("--agent-backend", choices=["deterministic", "llm"], default=os.environ.get("AGENT_BACKEND", "deterministic"))
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    ap.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    ap.add_argument("--llm-max-input-tokens", type=int, default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "3072")))
    args = ap.parse_args()
    run(args.input_jsonl, args.protocol_path, args.output_jsonl, args.metrics_json, args.max_rules, args.agent_backend, args.llm_model_id, args.llm_max_new_tokens, args.llm_temperature, args.llm_max_input_tokens)

import argparse
import os
import time

from src.latent_pipeline.common import iter_jsonl, write_jsonl, write_json, log
from src.latent_pipeline.inference_context import (
    INFERENCE_CONTEXT_SCHEMA,
    assert_no_future_fields,
)
from src.latent_pipeline.prompts import steward_system_prompt, steward_user_prompt


STATE_KEYS = [
    "hemodynamic_state",
    "respiratory_state",
    "renal_state",
    "metabolic_state",
    "systemic_inflammation_state",
]


def _update_state(prev: dict, prediction: dict, audit: dict, active_rule_ids=None) -> dict:
    st = dict(prev) if prev else {
        "hemodynamic_state": 0,
        "respiratory_state": 0,
        "renal_state": 0,
        "metabolic_state": 0,
        "systemic_inflammation_state": 0,
        "active_protocol_prediction": [],
    }
    for key in STATE_KEYS:
        st.setdefault(key, 0)
    acts = " ".join(prediction.get("predicted_actions", [])).lower()
    if "vasopressor" in acts:
        st["hemodynamic_state"] += 1
    if "respiratory" in acts or "oxygen" in acts:
        st["respiratory_state"] += 1
    if "renal" in acts or "aki" in acts or "dialysis" in acts:
        st["renal_state"] += 1
    if "metabolic" in acts:
        st["metabolic_state"] += 1
    inflammation_evidence = any(
        token in acts for token in ["infection", "sepsis", "inflamm", "antibiotic", "antimicrobial"]
    ) or any(str(rule_id).upper().startswith("INF") for rule_id in (active_rule_ids or []))
    if inflammation_evidence:
        st["systemic_inflammation_state"] += 1
    if audit.get("status") == "FAIL":
        st["hemodynamic_state"] = max(0, st["hemodynamic_state"] - 1)
    for k in STATE_KEYS:
        st[k] = int(max(0, min(5, st.get(k, 0))))
    return st


def _state_delta(prev: dict, nxt: dict) -> dict:
    keys = STATE_KEYS
    if prev is None:
        return {k: nxt.get(k, 0) for k in keys}
    return {k: int(nxt.get(k, 0)) - int(prev.get(k, 0)) for k in keys}


def run(input_jsonl: str, output_jsonl: str, metrics_json: str, agent_backend: str = "deterministic", llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct", llm_max_new_tokens: int = 256, llm_temperature: float = 0.1, llm_max_input_tokens: int = 3072):
    rows = []
    state = {}
    version = {}

    llm = None
    if agent_backend == "llm":
        from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig
        llm = LLMBackend(LLMConfig(model_id=llm_model_id, max_new_tokens=llm_max_new_tokens, temperature=llm_temperature, max_input_tokens=llm_max_input_tokens))
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "50"))
    t0 = time.time()
    n_seen = 0

    for ex in iter_jsonl(input_jsonl):
        n_seen += 1
        assert_no_future_fields(ex.get("packet", {}), "Steward input packet")
        key = f"{ex.get('source_dataset')}_{ex.get('patient_id')}_{ex.get('encounter_id')}"
        prev = state.get(key)

        if llm is None:
            nxt = _update_state(prev, ex.get("reasoner_prediction", {}), ex.get("audit", {}), ex.get("selected_rule_ids", []))
            delta = _state_delta(prev, nxt)
        else:
            out = llm.generate_json(steward_system_prompt(), steward_user_prompt(prev or {}, ex.get("reasoner_prediction", {}), ex.get("audit", {}), ex.get("selected_rule_ids", [])))
            nxt = out.get("state_next", {})
            delta = out.get("state_delta", {})
            keys = STATE_KEYS
            if not all(k in nxt for k in keys):
                nxt = _update_state(prev, ex.get("reasoner_prediction", {}), ex.get("audit", {}), ex.get("selected_rule_ids", []))
                delta = _state_delta(prev, nxt)
            for k in keys:
                nxt[k] = int(max(0, min(5, int(nxt.get(k, 0)))))
            nxt["active_protocol_prediction"] = list(dict.fromkeys([r for r in nxt.get("active_protocol_prediction", []) if isinstance(r, str)]))
            delta = _state_delta(prev, nxt)

        state[key] = nxt
        version[key] = version.get(key, 0) + 1

        ex["individual_protocol_state_prev"] = prev
        ex["inference_context_schema"] = INFERENCE_CONTEXT_SCHEMA
        ex["target_isolation_verified"] = True
        ex["individual_protocol_state_next"] = nxt
        ex["individual_protocol_state_delta"] = delta
        ex["state_version"] = version[key]
        ex["state_update_source"] = "steward"
        ex["stage4_prediction"] = {"state_next": nxt, "state_delta": delta, "state_version": version[key]}
        ex["stage4_ground_truth"] = ex.get("ground_truth_targets", {})
        rows.append(ex)
        if n_seen % progress_every == 0:
            dt = max(1e-6, time.time() - t0)
            log(f"Steward progress: {n_seen} rows ({n_seen/dt:.2f} rows/s), llm_failures={int(getattr(llm, '_failures', 0)) if llm else 0}")

    write_jsonl(output_jsonl, rows)
    pred_gt_path = os.path.join(os.path.dirname(output_jsonl), "stage4_predictions_ground_truth.jsonl")
    pred_gt_rows = [{"example_id": r.get("example_id"), "source_dataset": r.get("source_dataset"), "patient_id": r.get("patient_id"), "encounter_id": r.get("encounter_id"), "anchor_time": r.get("anchor_time"), "stage4_prediction": r.get("stage4_prediction", {}), "stage4_ground_truth": r.get("stage4_ground_truth", {})} for r in rows]
    write_jsonl(pred_gt_path, pred_gt_rows)
    n = len(rows)
    write_json(metrics_json, {"n_examples": n, "n_patient_states": len(state), "agent_backend": agent_backend, "llm_model_id": llm_model_id if llm else None, "llm_calls": int(getattr(llm, "_calls", 0)) if llm else 0, "llm_failures": int(getattr(llm, "_failures", 0)) if llm else 0, "inference_context_schema": INFERENCE_CONTEXT_SCHEMA, "target_isolation_verified": True})
    log(f"Steward stage done: n={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--agent-backend", choices=["deterministic", "llm"], default=os.environ.get("AGENT_BACKEND", "deterministic"))
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    ap.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    ap.add_argument("--llm-max-input-tokens", type=int, default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "3072")))
    args = ap.parse_args()
    run(args.input_jsonl, args.output_jsonl, args.metrics_json, args.agent_backend, args.llm_model_id, args.llm_max_new_tokens, args.llm_temperature, args.llm_max_input_tokens)

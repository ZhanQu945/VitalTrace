import argparse
import os
import time
from typing import Dict, List

from src.latent_pipeline.common import iter_jsonl, write_jsonl, write_json, log
from src.latent_pipeline.inference_context import (
    INFERENCE_CONTEXT_SCHEMA,
    assert_no_future_fields,
)
from src.latent_pipeline.prediction_targets import (
    COMPOSITE_DEFINITION,
    add_composite_probability,
)
from src.latent_pipeline.prompts import reasoner_system_prompt, reasoner_user_prompt


def _deterministic_plan(packet: Dict, selected_rule_ids: List[str]) -> Dict:
    facts = packet.get("facts", [])
    fmap = {f.get("feature"): f for f in facts}
    actions = []

    map_f = fmap.get("map")
    lac_f = fmap.get("lactate")
    if map_f and map_f.get("value_last") is not None and map_f.get("value_last") < 65:
        actions.append("consider vasopressor initiation/titration")
    if lac_f and lac_f.get("trend") == "rising":
        actions.append("consider fluid resuscitation and perfusion reassessment")
    spo2_f = fmap.get("spo2")
    rr_f = fmap.get("rr")
    if (spo2_f and spo2_f.get("value_last") is not None and spo2_f.get("value_last") < 90) or (rr_f and rr_f.get("value_last") is not None and rr_f.get("value_last") > 28):
        actions.append("consider oxygen escalation / respiratory support")
    cr_f = fmap.get("creatinine")
    if cr_f and (cr_f.get("trend") == "rising" or str(cr_f.get("abnormal_flag_last", "")).lower() == "high"):
        actions.append("consider AKI-protective management and renal dosing review")

    if not actions:
        actions.append("continue close monitoring and reassessment")

    risk_probs = add_composite_probability({"vasopressor_signal": float(any("vasopressor" in a.lower() for a in actions)), "resp_support_signal": float(any(("respiratory" in a.lower()) or ("oxygen" in a.lower()) for a in actions)), "renal_support_signal": float(any(("renal" in a.lower()) or ("aki" in a.lower()) for a in actions))})
    return {"next_bundle_type": "Mixed", "predicted_actions": actions[:5], "risk_probs": risk_probs, "any_deterioration_definition": COMPOSITE_DEFINITION, "citations": selected_rule_ids, "counterfactual_notes": []}


def run(input_jsonl: str, output_jsonl: str, metrics_json: str, agent_backend: str = "deterministic", llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct", llm_max_new_tokens: int = 256, llm_temperature: float = 0.1, llm_max_input_tokens: int = 3072):
    rows = []
    any_action = 0

    llm = None
    if agent_backend == "llm":
        from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig
        llm = LLMBackend(LLMConfig(model_id=llm_model_id, max_new_tokens=llm_max_new_tokens, temperature=llm_temperature, max_input_tokens=llm_max_input_tokens))
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "50"))
    t0 = time.time()
    n_seen = 0

    for ex in iter_jsonl(input_jsonl):
        n_seen += 1
        packet = ex.get("packet", {})
        assert_no_future_fields(packet, "Reasoner input packet")
        if llm is None:
            pred = _deterministic_plan(packet, ex.get("selected_rule_ids", []))
        else:
            out = llm.generate_json(reasoner_system_prompt(), reasoner_user_prompt(packet, ex.get("selected_rule_ids", []), ex.get("active_rules", {})))
            pred = {
                "next_bundle_type": out.get("next_bundle_type", "Mixed"),
                "predicted_actions": out.get("predicted_actions", [])[:5],
                "risk_probs": add_composite_probability(out.get("risk_probs", {})),
                "any_deterioration_definition": COMPOSITE_DEFINITION,
                "citations": [r for r in out.get("citations", []) if r in set(ex.get("selected_rule_ids", []))],
                "counterfactual_notes": out.get("counterfactual_notes", []),
            }
            if not pred["predicted_actions"]:
                pred = _deterministic_plan(packet, ex.get("selected_rule_ids", []))

        gt = ex.get("ground_truth_targets", {})
        acts_text = " ".join(pred.get("predicted_actions", [])).lower()

        vaso_pred = int((pred.get("risk_probs", {}).get("vasopressor_signal", 0) >= 0.5) or ("vasopressor" in acts_text))
        resp_pred = int((pred.get("risk_probs", {}).get("resp_support_signal", 0) >= 0.5) or ("respiratory" in acts_text or "oxygen" in acts_text))
        renal_pred = int((pred.get("risk_probs", {}).get("renal_support_signal", 0) >= 0.5) or ("renal" in acts_text or "aki" in acts_text))

        if pred.get("predicted_actions"):
            any_action += 1

        pred_proxy = {"vasopressor_signal": vaso_pred, "resp_support_signal": resp_pred, "renal_support_signal": renal_pred, "any_deterioration": int(vaso_pred or resp_pred or renal_pred)}
        rows.append({**ex, "inference_context_schema": INFERENCE_CONTEXT_SCHEMA, "target_isolation_verified": True, "reasoner_prediction": pred, "stage2_prediction": pred_proxy, "stage2_ground_truth": gt})
        if n_seen % progress_every == 0:
            dt = max(1e-6, time.time() - t0)
            log(f"Reasoner progress: {n_seen} rows ({n_seen/dt:.2f} rows/s), llm_failures={int(getattr(llm, '_failures', 0)) if llm else 0}")

    write_jsonl(output_jsonl, rows)
    pred_gt_path = os.path.join(os.path.dirname(output_jsonl), "stage2_predictions_ground_truth.jsonl")
    pred_gt_rows = [{
        "record_id": r.get("record_id"),
        "example_id": r.get("example_id"),
        "source_dataset": r.get("source_dataset"),
        "patient_id": r.get("patient_id"),
        "encounter_id": r.get("encounter_id"),
        "anchor_time": r.get("anchor_time"),
        "stage2_prediction": r.get("stage2_prediction", {}),
        "stage2_risk_probs": r.get("reasoner_prediction", {}).get("risk_probs", {}),
        "stage2_ground_truth": r.get("stage2_ground_truth", {}),
    } for r in rows]
    write_jsonl(pred_gt_path, pred_gt_rows)
    n = len(rows)
    write_json(metrics_json, {"n_examples": n, "any_action_rate": (any_action / n) if n else 0.0, "agent_backend": agent_backend, "llm_model_id": llm_model_id if llm else None, "llm_calls": int(getattr(llm, "_calls", 0)) if llm else 0, "llm_failures": int(getattr(llm, "_failures", 0)) if llm else 0, "inference_context_schema": INFERENCE_CONTEXT_SCHEMA, "target_isolation_verified": True})
    log(f"Reasoner stage done: n={n}")


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

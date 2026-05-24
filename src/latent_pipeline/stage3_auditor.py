import argparse
import os
import time

from src.latent_pipeline.common import iter_jsonl, write_jsonl, write_json, log
from src.latent_pipeline.prompts import auditor_system_prompt, auditor_user_prompt


RISK_MAP = [
    {"risk_tokens": ["shock", "hemodynamic", "hypoperfusion", "vasopressor"], "action_tokens": ["vasopressor", "fluid", "perfusion"], "prob_key": "vasopressor_signal"},
    {"risk_tokens": ["respiratory", "hypox", "ventilation", "oxygen"], "action_tokens": ["respiratory", "oxygen", "ventilation"], "prob_key": "resp_support_signal"},
    {"risk_tokens": ["aki", "renal", "dialysis"], "action_tokens": ["renal", "aki", "dialysis"], "prob_key": "renal_support_signal"},
]


def _tokens_from_rule(rule):
    toks = set()
    risk = str(rule.get("risk", "")).lower()
    name = str(rule.get("name", "")).lower()
    cat = str(rule.get("category", "")).lower()
    toks.update(risk.replace("-", "_").split("_"))
    toks.update(name.replace("-", " ").split())
    toks.update(cat.replace("-", "_").split("_"))
    for cf in rule.get("counterfactual_candidates", []) or []:
        toks.update(str(cf).lower().replace("-", "_").split("_"))
    return {t for t in toks if len(t) >= 3}


def _rule_addressed(rule, actions_text):
    # Direct lexical match from risk/name/category/counterfactual hints.
    rtoks = _tokens_from_rule(rule)
    if any(t in actions_text for t in rtoks):
        return True
    # Domain mapping fallback.
    risk = str(rule.get("risk", "")).lower()
    for m in RISK_MAP:
        if any(tok in risk for tok in m["risk_tokens"]):
            return any(tok in actions_text for tok in m["action_tokens"])
    return False


def _det_audit(active_rules, pred):
    actions_text = " ".join(pred.get("predicted_actions", [])).lower()
    probs = pred.get("risk_probs", {}) if isinstance(pred.get("risk_probs", {}), dict) else {}
    issues = []
    critical_issues = 0

    if not pred.get("predicted_actions", []):
        issues.append("critical: empty predicted_actions")
        critical_issues += 1

    for rid, rule in active_rules.items():
        risk = str(rule.get("risk", "")).lower()
        sev = str(rule.get("severity", "")).lower()
        for m in RISK_MAP:
            if not any(tok in risk for tok in m["risk_tokens"]):
                continue
            aligned = _rule_addressed(rule, actions_text)
            p = float(probs.get(m["prob_key"], 0.0))
            if (not aligned) and sev in {"high", "critical"} and p >= 0.5:
                issues.append(f"critical: missing action for high-severity rule={rid} risk={rule.get('risk','')}")
                critical_issues += 1
            elif (not aligned) and p >= 0.5:
                issues.append(f"warning: weak action-risk alignment rule={rid} risk={rule.get('risk','')}")
            break

    return {"status": "FAIL" if critical_issues > 0 else "PASS", "issues": issues, "suggested_fixes": []}


def run(input_jsonl: str, output_jsonl: str, metrics_json: str, agent_backend: str = "deterministic", llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct", llm_max_new_tokens: int = 256, llm_temperature: float = 0.1, llm_max_input_tokens: int = 3072):
    rows = []
    fail = 0

    llm = None
    if agent_backend == "llm":
        from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig
        llm = LLMBackend(LLMConfig(model_id=llm_model_id, max_new_tokens=llm_max_new_tokens, temperature=llm_temperature, max_input_tokens=llm_max_input_tokens))
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "50"))
    t0 = time.time()
    n_seen = 0

    for ex in iter_jsonl(input_jsonl):
        n_seen += 1
        det = _det_audit(ex.get("active_rules", {}), ex.get("reasoner_prediction", {}))
        packet = ex.get("packet", {}) if isinstance(ex.get("packet", {}), dict) else {}
        indiv_prev = ex.get("individual_protocol_state_prev", packet.get("individual_protocol_state_prev", {}))
        facts_current = packet.get("facts", [])
        if llm is not None:
            out = llm.generate_json(
                auditor_system_prompt(),
                auditor_user_prompt(
                    ex.get("active_rules", {}),
                    ex.get("reasoner_prediction", {}),
                    indiv_prev,
                    facts_current,
                ),
            )
            status = str(out.get("status", "PASS")).upper()
            if status not in {"PASS", "FAIL"}:
                status = "PASS"
            llm_audit = {"status": status, "issues": out.get("issues", []), "suggested_fixes": out.get("suggested_fixes", [])}
            # Deterministic audit is source of truth; keep LLM output as auxiliary notes.
            audit = {
                "status": det["status"],
                "issues": det["issues"],
                "suggested_fixes": det.get("suggested_fixes", []),
                "llm_status": llm_audit["status"],
                "llm_issues": llm_audit.get("issues", []),
            }
        else:
            audit = det

        fail += int(audit["status"] == "FAIL")
        ex["audit"] = audit
        ex["auditor_inputs"] = {
            "individual_protocol_state_prev": indiv_prev or {},
            "n_facts_current": len(facts_current or []),
        }
        ex["stage3_prediction"] = {"audit_status": audit["status"], "n_issues": len(audit["issues"])}
        ex["stage3_ground_truth"] = ex.get("ground_truth_targets", {})
        rows.append(ex)
        if n_seen % progress_every == 0:
            dt = max(1e-6, time.time() - t0)
            log(f"Auditor progress: {n_seen} rows ({n_seen/dt:.2f} rows/s), llm_failures={int(getattr(llm, '_failures', 0)) if llm else 0}")

    write_jsonl(output_jsonl, rows)
    pred_gt_path = os.path.join(os.path.dirname(output_jsonl), "stage3_predictions_ground_truth.jsonl")
    pred_gt_rows = [{"example_id": r.get("example_id"), "source_dataset": r.get("source_dataset"), "patient_id": r.get("patient_id"), "encounter_id": r.get("encounter_id"), "anchor_time": r.get("anchor_time"), "stage3_prediction": r.get("stage3_prediction", {}), "stage3_ground_truth": r.get("stage3_ground_truth", {})} for r in rows]
    write_jsonl(pred_gt_path, pred_gt_rows)
    n = len(rows)
    write_json(metrics_json, {"n_examples": n, "fail_rate": (fail / n) if n else 0.0, "agent_backend": agent_backend, "llm_model_id": llm_model_id if llm else None, "llm_calls": int(getattr(llm, "_calls", 0)) if llm else 0, "llm_failures": int(getattr(llm, "_failures", 0)) if llm else 0})
    log(f"Auditor stage done: n={n}")


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

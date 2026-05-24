from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.latent_pipeline.common import iter_jsonl, log, write_json, write_jsonl
from src.latent_pipeline.prompts import (
    auditor_system_prompt,
    auditor_user_prompt,
    reasoner_system_prompt,
    reasoner_user_prompt,
    router_system_prompt,
    router_user_prompt,
    steward_system_prompt,
    steward_user_prompt,
)
from src.latent_pipeline.protocol_utils import feature_map_from_facts, load_protocol, rule_score
from src.latent_pipeline.stage2_reasoner import _deterministic_plan
from src.latent_pipeline.stage3_auditor import _det_audit
from src.latent_pipeline.stage4_steward import _state_delta, _update_state


def _to_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _facts_from_example(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = ex.get("protocol_observations", [])
    if isinstance(facts, list) and len(facts) > 0:
        return facts
    # Fallback for stage3-style rows exposing only *_value fields.
    out: List[Dict[str, Any]] = []
    for feat in [
        "map",
        "rr",
        "spo2",
        "creatinine",
        "lactate",
        "hr",
        "temp",
        "wbc",
        "bicarbonate",
        "sodium",
        "potassium",
        "glucose",
    ]:
        v = _to_float(ex.get(f"{feat}_value"), None)
        if v is None:
            continue
        out.append(
            {
                "feature": feat,
                "value_last": v,
                "trend": "unknown",
                "count": 1,
                "abnormal_flag_last": "unknown",
            }
        )
    return out


def _stable_step_order(ex: Dict[str, Any]) -> Tuple:
    return (
        str(ex.get("source_dataset", "")),
        str(ex.get("patient_id", "")),
        str(ex.get("encounter_id", "")),
        int(ex.get("step_index", 0) or 0),
        str(ex.get("anchor_time", "")),
        int(ex.get("step_id", 0) or 0),
    )


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


def _extract_missing_rule_ids(issues: List[str]) -> List[str]:
    out: List[str] = []
    for s in issues or []:
        m = re.search(r"rule=([A-Za-z0-9_\-]+)", str(s))
        if m:
            out.append(m.group(1))
    # dedup preserving order
    return list(dict.fromkeys(out))


def _run_router(
    llm,
    rules: Dict[str, Dict],
    rules_brief: List[Dict[str, Any]],
    packet: Dict[str, Any],
    facts: List[Dict[str, Any]],
    max_rules: int,
) -> Tuple[List[str], Dict[str, Dict]]:
    if llm is None:
        return _det_router(rules, facts, max_rules)
    out = llm.generate_json(router_system_prompt(), router_user_prompt(packet, rules_brief, max_rules=max_rules))
    sel = [rid for rid in out.get("selected_rule_ids", []) if rid in rules][:max_rules]
    if not sel:
        sel, _ = _det_router(rules, facts, max_rules)
    active = {rid: rules[rid] for rid in sel if rid in rules}
    return sel, active


def _run_reasoner(llm, packet: Dict[str, Any], selected_rule_ids: List[str], active_rules: Dict[str, Dict]) -> Dict[str, Any]:
    if llm is None:
        return _deterministic_plan(packet, selected_rule_ids)
    out = llm.generate_json(reasoner_system_prompt(), reasoner_user_prompt(packet, selected_rule_ids, active_rules))
    raw_risk = out.get("risk_probs", out) if isinstance(out, dict) else {}
    aliases = {
        "vasopressor_signal": ["vasopressor_signal", "vaso", "vaso_signal", "vasopressor", "hemodynamic_risk"],
        "resp_support_signal": ["resp_support_signal", "resp", "resp_signal", "respiratory_support_signal", "respiratory_risk"],
        "renal_support_signal": ["renal_support_signal", "renal", "renal_signal", "renal_risk"],
        "any_deterioration": ["any_deterioration", "deterioration", "overall_risk", "global_risk"],
    }
    risk_norm: Dict[str, float] = {}
    if isinstance(raw_risk, dict):
        for canonical, keys in aliases.items():
            v = None
            for k in keys:
                if k in raw_risk:
                    v = raw_risk.get(k)
                    break
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip().replace("%", "")
                try:
                    fv = float(s)
                    if "%" in v:
                        fv /= 100.0
                    v = fv
                except Exception:
                    continue
            vv = _to_float(v, 0.0)
            risk_norm[canonical] = float(max(0.0, min(1.0, float(vv if vv is not None else 0.0))))
    if "any_deterioration" not in risk_norm and any(k in risk_norm for k in ["vasopressor_signal", "resp_support_signal", "renal_support_signal"]):
        risk_norm["any_deterioration"] = float(
            max(risk_norm.get("vasopressor_signal", 0.0), risk_norm.get("resp_support_signal", 0.0), risk_norm.get("renal_support_signal", 0.0))
        )
    pred = {
        "next_bundle_type": out.get("next_bundle_type", "Mixed"),
        "predicted_actions": out.get("predicted_actions", [])[:5],
        "risk_probs": risk_norm,
        "citations": [r for r in out.get("citations", []) if r in set(selected_rule_ids)],
        "counterfactual_notes": out.get("counterfactual_notes", []),
        "rationale": out.get("rationale", ""),
    }
    if not pred["predicted_actions"]:
        pred = _deterministic_plan(packet, selected_rule_ids)
    return pred


def _run_auditor(
    llm,
    active_rules: Dict[str, Dict],
    reasoner_prediction: Dict[str, Any],
    individual_protocol_state_prev: Dict[str, Any] | None = None,
    facts_current: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    det = _det_audit(active_rules, reasoner_prediction)
    if llm is None:
        return det
    out = llm.generate_json(
        auditor_system_prompt(),
        auditor_user_prompt(active_rules, reasoner_prediction, individual_protocol_state_prev, facts_current),
    )
    status = str(out.get("status", "PASS")).upper()
    if status not in {"PASS", "FAIL"}:
        status = "PASS"
    return {
        "status": det["status"],
        "issues": det["issues"],
        "suggested_fixes": det.get("suggested_fixes", []),
        "llm_status": status,
        "llm_issues": out.get("issues", []),
        "llm_suggested_fixes": out.get("suggested_fixes", []),
    }


def _run_steward(
    llm,
    prev: Dict[str, Any] | None,
    reasoner_prediction: Dict[str, Any],
    audit: Dict[str, Any],
    selected_rule_ids: List[str],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if llm is None:
        nxt = _update_state(prev, reasoner_prediction, audit)
        return nxt, _state_delta(prev, nxt)

    out = llm.generate_json(
        steward_system_prompt(),
        steward_user_prompt(prev or {}, reasoner_prediction, audit, selected_rule_ids),
    )
    nxt = out.get("state_next", {})
    keys = ["hemodynamic_state", "respiratory_state", "renal_state", "metabolic_state"]
    if not all(k in nxt for k in keys):
        nxt = _update_state(prev, reasoner_prediction, audit)
        return nxt, _state_delta(prev, nxt)

    for k in keys:
        nxt[k] = int(max(0, min(5, int(nxt.get(k, 0)))))
    nxt["active_protocol_prediction"] = list(dict.fromkeys([r for r in nxt.get("active_protocol_prediction", []) if isinstance(r, str)]))
    return nxt, _state_delta(prev, nxt)


def run(
    input_jsonl: str,
    protocol_path: str,
    out_dir: str,
    max_rules: int = 3,
    max_audit_retries: int = 1,
    fail_policy: str = "conservative_continue",
    agent_backend: str = "deterministic",
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    llm_max_new_tokens: int = 256,
    llm_temperature: float = 0.1,
    llm_max_input_tokens: int = 3072,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rules = load_protocol(protocol_path)
    rules_brief = [
        {"rule_id": rid, "name": r.get("name", ""), "risk": r.get("risk", ""), "trigger": r.get("trigger", [])}
        for rid, r in rules.items()
    ]

    llm = None
    if agent_backend == "llm":
        from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig

        llm = LLMBackend(
            LLMConfig(
                model_id=llm_model_id,
                max_new_tokens=llm_max_new_tokens,
                temperature=llm_temperature,
                max_input_tokens=llm_max_input_tokens,
            )
        )

    rows = list(iter_jsonl(input_jsonl))
    rows.sort(key=_stable_step_order)

    by_traj: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in rows:
        k = (str(ex.get("source_dataset")), str(ex.get("patient_id")), str(ex.get("encounter_id")))
        by_traj[k].append(ex)

    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []
    stage4_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    n_total = 0
    n_fail = 0
    n_retry_steps = 0
    n_retry_success = 0
    t0 = time.time()
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "50"))

    for traj_key, seq in by_traj.items():
        prev_state = None
        prev_audit = None
        for ex in seq:
            n_total += 1
            facts = _facts_from_example(ex)

            attempt = 0
            final = None
            attempt_logs: List[Dict[str, Any]] = []
            extra_retry_feedback: List[str] = []

            while True:
                packet = {
                    "facts": facts,
                    "latent_state": ex.get("latent_state_current", {}),
                    "risk_targets": ex.get("targets", {}),
                    "counterfactual_candidates": ex.get("counterfactual_candidates", []),
                    "individual_protocol_state_prev": prev_state or {},
                    "previous_audit_summary": prev_audit or {},
                    "audit_retry_feedback": extra_retry_feedback,
                }

                sel, active = _run_router(llm, rules, rules_brief, packet, facts, max_rules=max_rules)
                pred = _run_reasoner(llm, packet, sel, active)
                audit = _run_auditor(
                    llm,
                    active,
                    pred,
                    individual_protocol_state_prev=prev_state or {},
                    facts_current=facts,
                )

                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "selected_rule_ids": sel,
                        "active_rule_ids": list(active.keys()),
                        "reasoner_prediction": pred,
                        "audit": audit,
                    }
                )

                if audit.get("status") != "FAIL":
                    if attempt > 0:
                        n_retry_success += 1
                    final = {"packet": packet, "selected_rule_ids": sel, "active_rules": active, "reasoner_prediction": pred, "audit": audit}
                    break

                if attempt >= max_audit_retries:
                    final = {"packet": packet, "selected_rule_ids": sel, "active_rules": active, "reasoner_prediction": pred, "audit": audit}
                    break

                attempt += 1
                n_retry_steps += 1
                missing_rules = _extract_missing_rule_ids(audit.get("issues", []))
                extra_retry_feedback = list(
                    dict.fromkeys(
                        extra_retry_feedback
                        + [
                            "auditor_fail_repair_needed",
                            *[f"must_address_rule:{rid}" for rid in missing_rules],
                            *[f"issue:{x}" for x in audit.get("issues", [])[:4]],
                        ]
                    )
                )

            # Optional hard-fail policy.
            if final is None:
                continue
            if final["audit"].get("status") == "FAIL":
                n_fail += 1
                if fail_policy == "halt_step":
                    raise RuntimeError(
                        f"Audit failed after retries for trajectory={traj_key}, step_index={ex.get('step_index')}"
                    )

            nxt, delta = _run_steward(
                llm,
                prev=prev_state,
                reasoner_prediction=final["reasoner_prediction"],
                audit=final["audit"],
                selected_rule_ids=final["selected_rule_ids"],
            )

            # Build stage rows to keep compatibility with existing evaluators.
            base_example_id = ex.get(
                "example_id",
                f"{ex.get('source_dataset')}_{ex.get('patient_id')}_{ex.get('encounter_id')}_{ex.get('step_index', 0)}",
            )

            s1 = {
                "record_id": int(n_total),
                "example_id": base_example_id,
                "source_dataset": ex.get("source_dataset"),
                "patient_id": ex.get("patient_id"),
                "encounter_id": ex.get("encounter_id"),
                "anchor_time": ex.get("anchor_time"),
                "counterfactual_candidates": ex.get("counterfactual_candidates", []),
                "packet": final["packet"],
                "selected_rule_ids": final["selected_rule_ids"],
                "active_rules": final["active_rules"],
                "ground_truth_targets": ex.get("targets", {}),
                "stage1_prediction": {
                    "selected_rule_ids": final["selected_rule_ids"],
                    "n_selected_rules": len(final["selected_rule_ids"]),
                    "active_rule_ids": list(final["active_rules"].keys()),
                },
                "stage1_ground_truth": ex.get("targets", {}),
            }
            stage1_rows.append(s1)

            acts_text = " ".join(final["reasoner_prediction"].get("predicted_actions", [])).lower()
            risk_probs = final["reasoner_prediction"].get("risk_probs", {})
            vaso_pred = int((risk_probs.get("vasopressor_signal", 0) >= 0.5) or ("vasopressor" in acts_text))
            resp_pred = int((risk_probs.get("resp_support_signal", 0) >= 0.5) or ("respiratory" in acts_text or "oxygen" in acts_text))
            renal_pred = int((risk_probs.get("renal_support_signal", 0) >= 0.5) or ("renal" in acts_text or "aki" in acts_text))
            pred_proxy = {
                "vasopressor_signal": vaso_pred,
                "resp_support_signal": resp_pred,
                "renal_support_signal": renal_pred,
                "any_deterioration": int(vaso_pred or resp_pred or renal_pred),
            }

            s2 = {
                **s1,
                "reasoner_prediction": final["reasoner_prediction"],
                "stage2_prediction": pred_proxy,
                "stage2_ground_truth": ex.get("targets", {}),
            }
            stage2_rows.append(s2)

            s3 = {
                **s2,
                "audit": final["audit"],
                "auditor_inputs": {
                    "individual_protocol_state_prev": prev_state or {},
                    "n_facts_current": len(facts or []),
                },
                "stage3_prediction": {
                    "audit_status": final["audit"].get("status", "PASS"),
                    "n_issues": len(final["audit"].get("issues", [])),
                },
                "stage3_ground_truth": ex.get("targets", {}),
            }
            stage3_rows.append(s3)

            s4 = {
                **s3,
                "individual_protocol_state_prev": prev_state,
                "individual_protocol_state_next": nxt,
                "individual_protocol_state_delta": delta,
                "state_update_source": "steward",
                "stage4_prediction": {"state_next": nxt, "state_delta": delta},
                "stage4_ground_truth": ex.get("targets", {}),
                "temporal_loop": {
                    "retry_attempts_used": attempt,
                    "attempt_logs": attempt_logs,
                },
            }
            stage4_rows.append(s4)

            trace_rows.append(
                {
                    "example_id": base_example_id,
                    "source_dataset": ex.get("source_dataset"),
                    "patient_id": ex.get("patient_id"),
                    "encounter_id": ex.get("encounter_id"),
                    "anchor_time": ex.get("anchor_time"),
                    "step_index": ex.get("step_index"),
                    "attempts_used": attempt,
                    "final_audit_status": final["audit"].get("status", "PASS"),
                    "selected_rule_ids": final["selected_rule_ids"],
                    "state_prev": prev_state,
                    "state_next": nxt,
                }
            )

            prev_state = nxt
            prev_audit = final["audit"]

            if n_total % progress_every == 0:
                dt = max(1e-6, time.time() - t0)
                log(
                    f"Temporal-loop progress: {n_total} rows ({n_total/dt:.2f} rows/s), "
                    f"retry_steps={n_retry_steps}, unresolved_fails={n_fail}"
                )

    # Write outputs in same filenames used by existing tooling.
    s1_path = os.path.join(out_dir, "stage1_router.jsonl")
    s2_path = os.path.join(out_dir, "stage2_reasoner.jsonl")
    s3_path = os.path.join(out_dir, "stage3_auditor.jsonl")
    s4_path = os.path.join(out_dir, "stage4_steward.jsonl")
    write_jsonl(s1_path, stage1_rows)
    write_jsonl(s2_path, stage2_rows)
    write_jsonl(s3_path, stage3_rows)
    write_jsonl(s4_path, stage4_rows)

    # Compatibility prediction-ground-truth files.
    write_jsonl(
        os.path.join(out_dir, "stage1_predictions_ground_truth.jsonl"),
        [
            {
                "example_id": r.get("example_id"),
                "source_dataset": r.get("source_dataset"),
                "patient_id": r.get("patient_id"),
                "encounter_id": r.get("encounter_id"),
                "anchor_time": r.get("anchor_time"),
                "stage1_prediction": r.get("stage1_prediction", {}),
                "stage1_ground_truth": r.get("stage1_ground_truth", {}),
            }
            for r in stage1_rows
        ],
    )
    write_jsonl(
        os.path.join(out_dir, "stage2_predictions_ground_truth.jsonl"),
        [
            {
                "record_id": r.get("record_id"),
                "example_id": r.get("example_id"),
                "source_dataset": r.get("source_dataset"),
                "patient_id": r.get("patient_id"),
                "encounter_id": r.get("encounter_id"),
                "anchor_time": r.get("anchor_time"),
                "stage2_prediction": r.get("stage2_prediction", {}),
                "stage2_risk_probs": r.get("reasoner_prediction", {}).get("risk_probs", {}),
                "stage2_ground_truth": r.get("stage2_ground_truth", {}),
            }
            for r in stage2_rows
        ],
    )
    write_jsonl(
        os.path.join(out_dir, "stage3_predictions_ground_truth.jsonl"),
        [
            {
                "example_id": r.get("example_id"),
                "source_dataset": r.get("source_dataset"),
                "patient_id": r.get("patient_id"),
                "encounter_id": r.get("encounter_id"),
                "anchor_time": r.get("anchor_time"),
                "stage3_prediction": r.get("stage3_prediction", {}),
                "stage3_ground_truth": r.get("stage3_ground_truth", {}),
            }
            for r in stage3_rows
        ],
    )
    write_jsonl(
        os.path.join(out_dir, "stage4_predictions_ground_truth.jsonl"),
        [
            {
                "example_id": r.get("example_id"),
                "source_dataset": r.get("source_dataset"),
                "patient_id": r.get("patient_id"),
                "encounter_id": r.get("encounter_id"),
                "anchor_time": r.get("anchor_time"),
                "stage4_prediction": r.get("stage4_prediction", {}),
                "stage4_ground_truth": r.get("stage4_ground_truth", {}),
            }
            for r in stage4_rows
        ],
    )

    write_jsonl(os.path.join(out_dir, "temporal_handoff_trace.jsonl"), trace_rows)

    n = len(stage4_rows)
    metrics = {
        "runner_mode": "temporal_loop",
        "n_examples": n,
        "n_trajectories": len(by_traj),
        "audit_fail_rate_final": (n_fail / n) if n else 0.0,
        "retry_step_rate": (n_retry_steps / n) if n else 0.0,
        "retry_success_rate": (n_retry_success / n_retry_steps) if n_retry_steps else 0.0,
        "max_audit_retries": int(max_audit_retries),
        "fail_policy": fail_policy,
        "agent_backend": agent_backend,
        "llm_model_id": llm_model_id if llm else None,
        "llm_calls": int(getattr(llm, "_calls", 0)) if llm else 0,
        "llm_failures": int(getattr(llm, "_failures", 0)) if llm else 0,
    }
    write_json(os.path.join(out_dir, "metrics_temporal_loop.json"), metrics)

    # Also write stage metrics names for compatibility.
    write_json(os.path.join(out_dir, "metrics_router.json"), {"n_examples": n, "runner_mode": "temporal_loop", **metrics})
    write_json(os.path.join(out_dir, "metrics_reasoner.json"), {"n_examples": n, "runner_mode": "temporal_loop", **metrics})
    write_json(os.path.join(out_dir, "metrics_auditor.json"), {"n_examples": n, "runner_mode": "temporal_loop", **metrics})
    write_json(os.path.join(out_dir, "metrics_steward.json"), {"n_examples": n, "runner_mode": "temporal_loop", **metrics})
    log(f"Temporal-loop run done: n={n}, trajectories={len(by_traj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--protocol-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-rules", type=int, default=3)
    ap.add_argument("--max-audit-retries", type=int, default=1)
    ap.add_argument("--fail-policy", choices=["conservative_continue", "halt_step"], default="conservative_continue")
    ap.add_argument("--agent-backend", choices=["deterministic", "llm"], default=os.environ.get("AGENT_BACKEND", "deterministic"))
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    ap.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    ap.add_argument("--llm-max-input-tokens", type=int, default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "3072")))
    args = ap.parse_args()

    run(
        input_jsonl=args.input_jsonl,
        protocol_path=args.protocol_path,
        out_dir=args.out_dir,
        max_rules=args.max_rules,
        max_audit_retries=args.max_audit_retries,
        fail_policy=args.fail_policy,
        agent_backend=args.agent_backend,
        llm_model_id=args.llm_model_id,
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_temperature=args.llm_temperature,
        llm_max_input_tokens=args.llm_max_input_tokens,
    )


if __name__ == "__main__":
    main()

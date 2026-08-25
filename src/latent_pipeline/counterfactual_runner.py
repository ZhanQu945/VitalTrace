from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.latent_pipeline.common import iter_jsonl, log, write_json
from src.latent_pipeline.inference_context import build_inference_packet
from src.latent_pipeline.prediction_targets import SUPPORT_TARGETS, add_composite_probability
from src.latent_pipeline.protocol_utils import load_protocol
from src.latent_pipeline.stage4_steward import STATE_KEYS
from src.latent_pipeline.temporal_loop_runner import (
    _run_auditor,
    _run_reasoner,
    _run_router,
    _run_steward,
    _extract_missing_rule_ids,
)


EVALUATION_METHOD = "standardized_recovery_model_rerun_v2"
SCENARIOS = (
    ("map_low_to_normal", "vasopressor_signal"),
    ("remove_lactate_rise", "vasopressor_signal"),
    ("improve_creatinine", "renal_support_signal"),
    ("improve_oxygenation", "resp_support_signal"),
)


def _clone_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(fact) for fact in facts]


def _apply_cf(scenario: str, facts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool, str]:
    perturbed = _clone_facts(facts)
    feature_map = {str(fact.get("feature", "")).lower(): fact for fact in perturbed}

    if scenario == "map_low_to_normal":
        fact = feature_map.get("map")
        if fact is None or fact.get("value_last") is None or float(fact["value_last"]) >= 65.0:
            return perturbed, False, "map_not_low"
        fact.update(value_last=75.0, trend="stable", abnormal_flag_last="normal")
        return perturbed, True, "applied"

    if scenario == "remove_lactate_rise":
        fact = feature_map.get("lactate")
        if fact is None:
            return perturbed, False, "no_lactate"
        if str(fact.get("trend", "")).lower() not in {"rising", "increasing", "up"}:
            return perturbed, False, "lactate_not_rising"
        fact["trend"] = "stable"
        if fact.get("value_last") is not None:
            fact["value_last"] = min(float(fact["value_last"]), 2.0)
        fact["abnormal_flag_last"] = "normal"
        return perturbed, True, "applied"

    if scenario == "improve_creatinine":
        fact = feature_map.get("creatinine")
        if fact is None:
            return perturbed, False, "no_creatinine"
        trend = str(fact.get("trend", "")).lower()
        flag = str(fact.get("abnormal_flag_last", "")).lower()
        value = fact.get("value_last")
        is_high = value is not None and float(value) > 1.2
        if trend not in {"rising", "increasing", "up"} and flag not in {"high", "critical_high"} and not is_high:
            return perturbed, False, "creatinine_not_high_or_rising"
        if value is not None:
            fact["value_last"] = min(float(value), 1.2)
        fact.update(trend="stable", abnormal_flag_last="normal")
        return perturbed, True, "applied"

    if scenario == "improve_oxygenation":
        spo2 = feature_map.get("spo2")
        rr = feature_map.get("rr")
        changed = False
        if spo2 is not None and spo2.get("value_last") is not None and float(spo2["value_last"]) < 92.0:
            spo2.update(value_last=95.0, trend="stable", abnormal_flag_last="normal")
            changed = True
        if rr is not None and rr.get("value_last") is not None and float(rr["value_last"]) > 24.0:
            rr.update(value_last=20.0, trend="stable", abnormal_flag_last="normal")
            changed = True
        if not changed:
            return perturbed, False, "oxygenation_not_abnormal"
        return perturbed, True, "applied"

    return perturbed, False, "unknown_scenario"


def _make_llm(agent_backend: str, model_id: str, max_new_tokens: int, temperature: float, max_input_tokens: int):
    if agent_backend != "llm":
        return None
    from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig

    return LLMBackend(
        LLMConfig(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_input_tokens=max_input_tokens,
        )
    )


def _call_reasoner(
    llm,
    packet,
    selected,
    active,
    reasoner_output_mode: str,
    enable_rare_label_gate: bool,
    rare_label_gate_threshold: float,
    rare_label_gate_cap: float,
):
    kwargs = {}
    if "reasoner_output_mode" in inspect.signature(_run_reasoner).parameters:
        kwargs["reasoner_output_mode"] = reasoner_output_mode
    if "enable_rare_label_gate" in inspect.signature(_run_reasoner).parameters:
        kwargs.update(
            enable_rare_label_gate=enable_rare_label_gate,
            rare_label_gate_threshold=rare_label_gate_threshold,
            rare_label_gate_cap=rare_label_gate_cap,
        )
    return _run_reasoner(llm, packet, selected, active, **kwargs)


def run(
    out_dir: str,
    protocol_json: str,
    agent_backend: str = "llm",
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    llm_max_new_tokens: int = 256,
    llm_temperature: float = 0.1,
    llm_max_input_tokens: int = 3072,
    max_rules: int = 3,
    reasoner_output_mode: str = "constrained_json",
    max_audit_retries: int = 1,
    enable_rare_label_gate: bool = True,
    rare_label_gate_threshold: float = 0.5,
    rare_label_gate_cap: float = 0.45,
) -> None:
    stage4_path = os.path.join(out_dir, "stage4_steward.jsonl")
    rows = list(iter_jsonl(stage4_path))
    if any(not row.get("target_isolation_verified", False) for row in rows):
        raise ValueError(
            "Counterfactual evaluation requires target-free corrected stage outputs. "
            "Rerun the main pipeline before recomputing counterfactual metrics."
        )
    if any("temporal_loop" not in row for row in rows):
        raise ValueError(
            "Standardized counterfactual evaluation currently requires temporal-loop artifacts."
        )
    if any(
        row.get("reasoner_prediction", {}).get("any_deterioration_definition")
        != "max_support_probability"
        for row in rows
    ):
        raise ValueError(
            "Counterfactual evaluation requires the three-support-plus-composite endpoint schema."
        )
    if any(row.get("reasoner_prediction", {}).get("baseline") for row in rows):
        raise ValueError(
            "Counterfactual model reruns require the Vital Trace inference backend; "
            "baseline artifacts need a baseline-specific perturbation adapter."
        )

    rules = load_protocol(protocol_json)
    rules_brief = [
        {"rule_id": rule_id, "name": rule.get("name", ""), "risk": rule.get("risk", ""), "trigger": rule.get("trigger", [])}
        for rule_id, rule in rules.items()
    ]
    llm = _make_llm(
        agent_backend,
        llm_model_id,
        llm_max_new_tokens,
        llm_temperature,
        llm_max_input_tokens,
    )

    results: List[Dict[str, Any]] = []
    progress_every = int(os.environ.get("STAGE_PROGRESS_EVERY", "25"))
    for row_index, row in enumerate(rows, start=1):
        base_packet = row.get("packet", {})
        facts = base_packet.get("facts", [])
        base_probs = add_composite_probability(row.get("reasoner_prediction", {}).get("risk_probs", {}))
        base_rules = set(row.get("selected_rule_ids", []))
        base_state = row.get("individual_protocol_state_next", {}) or {}
        prev_state = row.get("individual_protocol_state_prev", {}) or {}

        for scenario, target in SCENARIOS:
            cf_facts, applied, reason = _apply_cf(scenario, facts)
            result: Dict[str, Any] = {
                "example_id": row.get("example_id"),
                "source_dataset": row.get("source_dataset"),
                "patient_id": row.get("patient_id"),
                "encounter_id": row.get("encounter_id"),
                "anchor_time": row.get("anchor_time"),
                "scenario": scenario,
                "target": target,
                "expected_direction": "decrease",
                "applied": int(applied),
                "apply_reason": reason,
                "evaluation_method": EVALUATION_METHOD,
            }
            if not applied:
                results.append(result)
                continue

            retry_feedback: List[str] = []
            attempt = 0
            while True:
                cf_packet = build_inference_packet(
                    facts=cf_facts,
                    latent_state=base_packet.get("latent_state", {}),
                    counterfactual_candidates=base_packet.get("counterfactual_candidates", []),
                    individual_protocol_state_prev=prev_state,
                    previous_audit_summary=base_packet.get("previous_audit_summary", {}),
                    audit_retry_feedback=retry_feedback,
                )
                selected, active = _run_router(llm, rules, rules_brief, cf_packet, cf_facts, max_rules)
                prediction = _call_reasoner(
                    llm,
                    cf_packet,
                    selected,
                    active,
                    reasoner_output_mode,
                    enable_rare_label_gate,
                    rare_label_gate_threshold,
                    rare_label_gate_cap,
                )
                audit = _run_auditor(llm, active, prediction, prev_state, cf_facts)
                if audit.get("status") != "FAIL" or attempt >= max_audit_retries:
                    break
                attempt += 1
                missing_rules = _extract_missing_rule_ids(audit.get("issues", []))
                retry_feedback = list(
                    dict.fromkeys(
                        ["auditor_fail_repair_needed"]
                        + [f"must_address_rule:{rule_id}" for rule_id in missing_rules]
                        + [f"issue:{issue}" for issue in audit.get("issues", [])[:4]]
                    )
                )
            cf_probs = add_composite_probability(prediction.get("risk_probs", {}))
            cf_state, _ = _run_steward(llm, prev_state, prediction, audit, selected)

            target_delta = float(cf_probs[target] - base_probs[target])
            result.update(
                {
                    "base_prob_target": float(base_probs[target]),
                    "cf_prob_target": float(cf_probs[target]),
                    "delta_prob_target": target_delta,
                    "direction_consistent": int(target_delta < -1e-8),
                    "nonincrease_consistent": int(target_delta <= 1e-8),
                    "base_risk_probs": base_probs,
                    "cf_risk_probs": cf_probs,
                    "base_selected_rule_ids": sorted(base_rules),
                    "cf_selected_rule_ids": selected,
                    "protocol_activation_changed": int(base_rules != set(selected)),
                    "base_individual_state": base_state,
                    "cf_individual_state": cf_state,
                    "individual_state_changed": int(
                        any(int(base_state.get(key, 0)) != int(cf_state.get(key, 0)) for key in STATE_KEYS)
                    ),
                    "audit_retry_attempts_used": int(attempt),
                }
            )
            results.append(result)

        if row_index % progress_every == 0:
            log(f"Counterfactual progress: {row_index}/{len(rows)} source rows")

    frame = pd.DataFrame(results)
    applied = frame[frame["applied"] == 1].copy() if not frame.empty else frame
    probability_changes = {
        target: (
            float(np.mean([
                record["cf_risk_probs"][target] - record["base_risk_probs"][target]
                for record in results
                if record.get("applied") == 1
            ]))
            if any(record.get("applied") == 1 for record in results)
            else None
        )
        for target in SUPPORT_TARGETS
    }
    per_scenario = {}
    for scenario, _ in SCENARIOS:
        subset = applied[applied["scenario"] == scenario] if not applied.empty else applied
        per_scenario[scenario] = {
            "n_applied": int(len(subset)),
            "directional_consistency_rate": float(subset["direction_consistent"].mean()) if len(subset) else None,
            "nonincrease_consistency_rate": float(subset["nonincrease_consistent"].mean()) if len(subset) else None,
            "mean_target_probability_delta": float(subset["delta_prob_target"].mean()) if len(subset) else None,
        }

    metrics = {
        "evaluation_method": EVALUATION_METHOD,
        "agent_backend": agent_backend,
        "llm_model_id": llm_model_id if llm is not None else None,
        "reasoner_output_mode": reasoner_output_mode,
        "max_audit_retries": int(max_audit_retries),
        "enable_rare_label_gate": bool(enable_rare_label_gate),
        "rare_label_gate_threshold": float(rare_label_gate_threshold),
        "rare_label_gate_cap": float(rare_label_gate_cap),
        "standardized_scenarios": [scenario for scenario, _ in SCENARIOS],
        "n_source_rows": int(len(rows)),
        "n_scenario_rows": int(len(frame)),
        "n_applied": int(len(applied)),
        "n_directional_checks": int(len(applied)),
        "directional_consistency_rate": float(applied["direction_consistent"].mean()) if len(applied) else None,
        "nonincrease_consistency_rate": float(applied["nonincrease_consistent"].mean()) if len(applied) else None,
        "average_probability_change": probability_changes,
        "avg_probability_change": probability_changes,
        "protocol_activation_change_rate": float(applied["protocol_activation_changed"].mean()) if len(applied) else None,
        "individual_protocol_state_change_rate": float(applied["individual_state_changed"].mean()) if len(applied) else None,
        "individual_state_change_rate": float(applied["individual_state_changed"].mean()) if len(applied) else None,
        "per_scenario": per_scenario,
        "llm_calls": int(getattr(llm, "_calls", 0)) if llm is not None else 0,
        "llm_failures": int(getattr(llm, "_failures", 0)) if llm is not None else 0,
    }

    os.makedirs(out_dir, exist_ok=True)
    frame.to_csv(os.path.join(out_dir, "counterfactual_results.csv"), index=False)
    frame.to_parquet(os.path.join(out_dir, "counterfactual_results.parquet"), index=False)
    write_json(os.path.join(out_dir, "counterfactual_metrics.json"), metrics)
    write_json(os.path.join(out_dir, "counterfactual_summary.json"), metrics)
    log(f"Counterfactual model-rerun evaluation done: applied={len(applied)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--agent-backend", choices=["deterministic", "llm"], default="llm")
    parser.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    parser.add_argument("--llm-max-input-tokens", type=int, default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "3072")))
    parser.add_argument("--max-rules", type=int, default=3)
    parser.add_argument("--reasoner-output-mode", choices=["constrained_json", "freeform_text"], default="constrained_json")
    parser.add_argument("--max-audit-retries", type=int, default=1)
    parser.add_argument("--enable-rare-label-gate", type=lambda value: str(value).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--rare-label-gate-threshold", type=float, default=0.5)
    parser.add_argument("--rare-label-gate-cap", type=float, default=0.45)
    arguments = parser.parse_args()
    run(
        arguments.out_dir,
        arguments.protocol_json,
        arguments.agent_backend,
        arguments.llm_model_id,
        arguments.llm_max_new_tokens,
        arguments.llm_temperature,
        arguments.llm_max_input_tokens,
        arguments.max_rules,
        arguments.reasoner_output_mode,
        arguments.max_audit_retries,
        arguments.enable_rare_label_gate,
        arguments.rare_label_gate_threshold,
        arguments.rare_label_gate_cap,
    )

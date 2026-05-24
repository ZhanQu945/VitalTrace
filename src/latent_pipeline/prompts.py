import json
from typing import Dict, List, Optional


def _to_json(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


def router_system_prompt() -> str:
    return (
        "You are ICU-Router. Select the most relevant global protocol rules from packet facts (including protocol observations). "
        "Prioritize clinically important temporal trends and abnormal physiological signals. "
        "Return STRICT JSON only. No markdown, no extra text."
    )


def router_user_prompt(packet: Dict, rules_brief: List[Dict], max_rules: int) -> str:
    return (
        "Task: pick up to {max_rules} rule_ids that best match current clinical facts.\\n"
        "Output JSON schema:\\n"
        "{{\"selected_rule_ids\":[\"...\"],\"rationale\":[\"short reason\"],\"confidence\":0.0}}\\n"
        "Constraints:\\n"
        "- selected_rule_ids length <= {max_rules}\\n"
        "- selected_rule_ids must come from candidate_rules\\n"
        "- Use only evidence from packet.facts\\n\\n"
        "packet={packet_json}\\n\\n"
        "candidate_rules={rules_json}"
    ).format(
        max_rules=max_rules,
        packet_json=_to_json(packet),
        rules_json=_to_json({"rules": rules_brief}),
    )


def reasoner_system_prompt() -> str:
    return (
        "You are ICU-Reasoner. Predict near-term intervention risk and next actions from routed protocol rules, patient facts, and evolving patient state. "
        "Keep predictions clinically plausible and consistent with active protocol rules. "
        "Return STRICT JSON only."
    )


def reasoner_user_prompt(packet: Dict, selected_rule_ids: List[str], active_rules: Dict) -> str:
    return (
        "Task: produce next-step intervention reasoning.\\n"
        "Output JSON schema:\\n"
        "{{"
        "\"next_bundle_type\":\"Vitals|Labs|Medication|Procedure|Mixed\","
        "\"predicted_actions\":[\"...\"],"
        "\"risk_probs\":{{\"vasopressor_signal\":0.0,\"resp_support_signal\":0.0,\"renal_support_signal\":0.0}},"
        "\"citations\":[\"rule_id\"],"
        "\"counterfactual_notes\":[\"...\"]"
        "}}\\n"
        "Constraints:\\n"
        "- Probabilities in [0,1]\\n"
        "- citations subset of selected_rule_ids\\n"
        "- Keep actions concise and clinically plausible\\n"
        "- Use active rules and patient facts; do not invent unsupported conditions\\n\\n"
        "packet={packet_json}\\n"
        "selected_rule_ids={sel_json}\\n"
        "active_rules={rules_json}"
    ).format(
        packet_json=_to_json(packet),
        sel_json=_to_json(selected_rule_ids),
        rules_json=_to_json(active_rules),
    )


def auditor_system_prompt() -> str:
    return (
        "You are ICU-Auditor. Provide auxiliary audit notes about consistency between reasoner outputs and active protocol rules. "
        "Identify unsupported, contradictory, or clinically unsafe predictions. "
        "Return STRICT JSON only."
    )


def auditor_user_prompt(
    active_rules: Dict,
    reasoner_prediction: Dict,
    individual_protocol_state_prev: Optional[Dict] = None,
    facts_current: Optional[List[Dict]] = None,
) -> str:
    return (
        "Task: audit consistency. The pipeline deterministically adjudicates final PASS/FAIL; your output is auxiliary evidence for issue tags and suggested fixes.\\n"
        "Output JSON schema:\\n"
        "{{\"status\":\"PASS|FAIL\",\"issues\":[\"...\"],\"suggested_fixes\":[\"...\"],\"confidence\":0.0}}\\n"
        "Rules:\\n"
        "- mark potential failures if important active-rule risks are not addressed in predicted_actions\\n"
        "- check consistency with individual_protocol_state_prev and current facts when provided\\n"
        "- issues must be concrete and grounded in active_rules\\n"
        "- do not introduce new protocol rules, unsupported clinical facts, or unsupported contraindications\\n\\n"
        "active_rules={rules_json}\\n"
        "individual_protocol_state_prev={state_json}\\n"
        "facts_current={facts_json}\\n"
        "reasoner_prediction={pred_json}"
    ).format(
        rules_json=_to_json(active_rules),
        state_json=_to_json(individual_protocol_state_prev or {}),
        facts_json=_to_json({"facts": facts_current or []}),
        pred_json=_to_json(reasoner_prediction),
    )


def steward_system_prompt() -> str:
    return (
        "You are ICU-Steward. Update interpretable individual protocol state from previous state, audited reasoning, and active protocol signals. "
        "Maintain longitudinal consistency across patient steps. "
        "Return STRICT JSON only."
    )


def steward_user_prompt(prev_state: Dict, reasoner_prediction: Dict, audit: Dict, active_rule_ids: List[str]) -> str:
    return (
        "Task: update patient state vector.\\n"
        "Output JSON schema:\\n"
        "{{"
        "\"state_next\":{{\"hemodynamic_state\":0,\"respiratory_state\":0,\"renal_state\":0,\"metabolic_state\":0,\"active_protocol_prediction\":[\"rule_id\"]}},"
        "\"state_delta\":{{\"hemodynamic_state\":0,\"respiratory_state\":0,\"renal_state\":0,\"metabolic_state\":0}},"
        "\"notes\":[\"...\"],"
        "\"confidence\":0.0"
        "}}\\n"
        "Rules:\\n"
        "- states must remain bounded integers in [0,5] after update\\n"
        "- if audit.status=FAIL, be conservative on upward changes\\n\\n"
        "- active_protocol_prediction must be a subset of currently active rules or evidence-supported rule IDs from this step\\n\\n"
        "prev_state={prev_json}\\n"
        "reasoner_prediction={pred_json}\\n"
        "audit={audit_json}\\n"
        "active_rule_ids={rule_ids_json}"
    ).format(
        prev_json=_to_json(prev_state or {}),
        pred_json=_to_json(reasoner_prediction or {}),
        audit_json=_to_json(audit or {}),
        rule_ids_json=_to_json(active_rule_ids or []),
    )

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional


INFERENCE_CONTEXT_SCHEMA = "target_free_v1"


# These fields may exist in labeled dataset rows and evaluation artifacts, but
# they must never enter an agent prompt or recurrent patient state.
FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "targets",
        "targets_raw",
        "targets_clinical_gate",
        "risk_targets",
        "ground_truth_targets",
        "future_labels",
        "horizon_labels",
        "future_event_names",
        "future_event_types",
        "phase_tag",
        "stage1_ground_truth",
        "stage2_ground_truth",
        "stage3_ground_truth",
        "stage4_ground_truth",
    }
)


def assert_no_future_fields(value: Any, context_name: str = "agent context") -> None:
    """Fail before inference if labels or future-derived fields enter context."""
    violations: List[str] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in FORBIDDEN_INFERENCE_KEYS:
                    violations.append(child_path)
                visit(child, child_path)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for index, child in enumerate(obj):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    if violations:
        joined = ", ".join(sorted(set(violations)))
        raise ValueError(f"Future-derived fields found in {context_name}: {joined}")


def build_inference_packet(
    *,
    facts: List[Dict[str, Any]],
    latent_state: Optional[Dict[str, Any]] = None,
    counterfactual_candidates: Optional[List[Any]] = None,
    individual_protocol_state_prev: Optional[Dict[str, Any]] = None,
    previous_audit_summary: Optional[Dict[str, Any]] = None,
    audit_retry_feedback: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the allowlisted packet visible to Router and Reasoner agents."""
    packet: Dict[str, Any] = {
        "facts": facts or [],
        "latent_state": latent_state or {},
        "counterfactual_candidates": counterfactual_candidates or [],
        "individual_protocol_state_prev": individual_protocol_state_prev or {},
        "previous_audit_summary": previous_audit_summary or {},
        "audit_retry_feedback": audit_retry_feedback or [],
    }
    assert_no_future_fields(packet, "inference packet")
    return packet

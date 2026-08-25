from __future__ import annotations

from typing import Any, Dict


SUPPORT_TARGETS = (
    "vasopressor_signal",
    "resp_support_signal",
    "renal_support_signal",
)
COMPOSITE_TARGET = "any_deterioration"
COMPOSITE_DEFINITION = "max_support_probability"


def add_composite_probability(risk_probs: Dict[str, Any]) -> Dict[str, float]:
    """Return three support probabilities plus their max-probability composite."""
    normalized: Dict[str, float] = {}
    for target in SUPPORT_TARGETS:
        try:
            value = float(risk_probs.get(target, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        normalized[target] = max(0.0, min(1.0, value))
    normalized[COMPOSITE_TARGET] = max(normalized.values(), default=0.0)
    return normalized


def add_composite_label(targets: Dict[str, Any]) -> Dict[str, int]:
    """Return binary support labels plus their logical-OR composite."""
    normalized = {target: int(bool(targets.get(target, 0))) for target in SUPPORT_TARGETS}
    normalized[COMPOSITE_TARGET] = int(any(normalized.values()))
    return normalized

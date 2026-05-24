from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CORE_VITALS = {"map", "rr", "spo2", "hr", "temp"}
CORE_LABS = {"lactate", "creatinine", "wbc", "bicarbonate", "sodium", "potassium", "glucose"}
CORE_FEATURES = CORE_VITALS | CORE_LABS

FEATURE_ALIASES = {
    "map": ["map", "mean blood pressure", "arterial blood pressure mean", "systemicmean", "noninvasivemean"],
    "rr": ["rr", "respiratory rate", "respiration", "resp rate"],
    "spo2": ["spo2", "o2 saturation", "sao2", "pulse ox", "pulseox"],
    "hr": ["hr", "heart rate", "heartrate", "pulse"],
    "temp": ["temp", "temperature"],
    "lactate": ["lactate"],
    "creatinine": ["creatinine"],
    "wbc": ["wbc", "white blood cell"],
    "bicarbonate": ["bicarbonate", "hco3"],
    "sodium": ["sodium", "na"],
    "potassium": ["potassium", "k "],
    "glucose": ["glucose"],
}

THRESHOLDS = {
    "map": (65.0, 110.0),
    "rr": (10.0, 24.0),
    "spo2": (90.0, 100.0),
    "hr": (50.0, 110.0),
    "temp": (36.0, 38.0),
    "lactate": (0.0, 2.0),
    "creatinine": (0.0, 1.5),
    "wbc": (4.0, 12.0),
    "bicarbonate": (22.0, 28.0),
    "sodium": (135.0, 145.0),
    "potassium": (3.5, 5.5),
    "glucose": (70.0, 180.0),
}

INTERVENTION_PATTERNS = {
    "VASOPRESSOR_START_OR_TITRATION": re.compile(r"(?:norepinephrine|noradrenaline|levophed|vasopressin|epinephrine|adrenaline|phenylephrine|neosynephrine|dopamine|dobutamine)", re.I),
    "FLUID_RESUSCITATION": re.compile(r"(?:fluid\s*bolus|crystalloid|resuscitation\s*bolus)", re.I),
    "OXYGEN_ESCALATION": re.compile(r"(?:oxygen|hfnc|high flow)", re.I),
    "NONINVASIVE_RESP_SUPPORT": re.compile(r"(?:cpap|bipap|bi\s*-?\s*pap|hfnc)", re.I),
    "INVASIVE_RESP_SUPPORT": re.compile(r"(?:intub|endotracheal|ett|mechanical\s*vent|ventilat)", re.I),
    "RENAL_REPLACEMENT_THERAPY": re.compile(r"(?:dialysis|crrt|cvvh|cvvhd|cvvhdf|rrt)", re.I),
    "ANTIMICROBIAL_REVIEW_OR_START": re.compile(r"(?:antibiotic|antimicrobial|vancomycin|piperacillin|cefepime|meropenem)", re.I),
}

TARGET_CONCEPTS = {
    "vasopressor_signal": {"VASOPRESSOR_START_OR_TITRATION", "FLUID_RESUSCITATION"},
    "resp_support_signal": {"OXYGEN_ESCALATION", "NONINVASIVE_RESP_SUPPORT", "INVASIVE_RESP_SUPPORT"},
    "renal_support_signal": {"RENAL_REPLACEMENT_THERAPY"},
}


@dataclass
class MatchResult:
    concepts: List[str]
    provenance: str


def canonical_feature(name: str) -> Optional[str]:
    n = str(name or "").lower()
    for c, aliases in FEATURE_ALIASES.items():
        if any(a in n for a in aliases):
            return c
    return None


def state_from_value(feature: str, value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "missing"
    lo, hi = THRESHOLDS.get(feature, (None, None))
    if lo is not None and value < lo:
        return "low"
    if hi is not None and value > hi:
        return "high"
    return "normal"


def normalize_code(code: str) -> str:
    s = str(code or "").strip().upper()
    s = re.sub(r"[^A-Z0-9.]", "", s)
    if s and "." not in s and re.match(r"^[A-Z][0-9]{2,4}$", s):
        # simple ICD-like normalization fallback
        s = s[:3] + "." + s[3:]
    return s


def parse_payload_code(payload_json: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        o = json.loads(payload_json) if payload_json else {}
    except Exception:
        return None, None
    for k in ["icd_code", "code", "itemid", "drug_code", "treatmentid"]:
        if k in o and o[k] is not None:
            raw = str(o[k])
            return raw, normalize_code(raw)
    return None, None


def map_intervention(event_name: str, payload_json: str) -> MatchResult:
    txt = str(event_name or "")
    raw_code, norm_code = parse_payload_code(payload_json)

    # code-first placeholders: users can extend code dictionaries here
    # default implementation falls back to text patterns but preserves provenance
    if raw_code:
        for concept, pat in INTERVENTION_PATTERNS.items():
            if pat.search(txt):
                return MatchResult([concept], "code_normalized" if norm_code and norm_code != raw_code else "code_exact")

    hits = [c for c, pat in INTERVENTION_PATTERNS.items() if pat.search(txt)]
    if hits:
        return MatchResult(hits, "desc_mapped")
    return MatchResult([], "regex_fallback")


def print_stage_header(stage: str):
    print(f"\n{'='*24} {stage} {'='*24}", flush=True)


def print_examples(df: pd.DataFrame, title: str, cols: List[str], n: int = 3):
    print(f"\n[examples] {title} (n={min(n, len(df))})", flush=True)
    if df.empty:
        print("  <none>", flush=True)
        return
    for i, (_, r) in enumerate(df.head(n).iterrows(), start=1):
        rec = {c: r.get(c, None) for c in cols}
        print(f"  {i}. {rec}", flush=True)

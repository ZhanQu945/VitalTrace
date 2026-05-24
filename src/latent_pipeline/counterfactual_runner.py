from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.latent_pipeline.common import iter_jsonl, write_json
from src.latent_pipeline.protocol_utils import load_protocol, feature_map_from_facts, rule_score


TARGETS = ["vasopressor_signal", "resp_support_signal", "renal_support_signal"]


def _risk_probs_from_facts(facts: List[Dict]) -> Dict[str, float]:
    fmap = feature_map_from_facts(facts)
    map_v = fmap.get("map", {}).get("value_last")
    lac_t = str(fmap.get("lactate", {}).get("trend", "")).lower()
    spo2_v = fmap.get("spo2", {}).get("value_last")
    rr_v = fmap.get("rr", {}).get("value_last")
    cr_v = fmap.get("creatinine", {}).get("value_last")
    cr_flag = str(fmap.get("creatinine", {}).get("abnormal_flag_last", "")).lower()
    cr_t = str(fmap.get("creatinine", {}).get("trend", "")).lower()

    p_vaso = 0.7 if (map_v is not None and map_v < 65) else 0.2
    if lac_t == "rising":
        p_vaso = min(1.0, p_vaso + 0.2)

    p_resp = 0.2
    if spo2_v is not None and spo2_v < 90:
        p_resp += 0.5
    if rr_v is not None and rr_v > 28:
        p_resp += 0.3
    p_resp = min(1.0, p_resp)

    p_renal = 0.15
    if cr_t == "rising" or cr_flag == "high":
        p_renal += 0.45
    if cr_v is not None and cr_v >= 2.0:
        p_renal += 0.15
    p_renal = min(1.0, p_renal)

    return {
        "vasopressor_signal": float(p_vaso),
        "resp_support_signal": float(p_resp),
        "renal_support_signal": float(p_renal),
    }


def _active_rules(rules: Dict, facts: List[Dict]) -> List[str]:
    fmap = feature_map_from_facts(facts)
    out = []
    for rid, rule in rules.items():
        ok, _ = rule_score(rule, fmap)
        if ok:
            out.append(rid)
    return out


def _clone_facts(facts: List[Dict]) -> List[Dict]:
    return [dict(x) for x in facts]


def _apply_cf(scenario: str, facts: List[Dict]) -> Tuple[List[Dict], bool, str]:
    f = _clone_facts(facts)
    fmap = {str(x.get("feature", "")): x for x in f}

    if scenario == "map_low_to_normal":
        m = fmap.get("map")
        if m is None or m.get("value_last") is None or float(m.get("value_last")) >= 65:
            return f, False, "map_not_low"
        m["value_last"] = 75.0
        m["abnormal_flag_last"] = "normal"
        return f, True, "applied"

    if scenario == "remove_lactate_rise":
        l = fmap.get("lactate")
        if l is None:
            return f, False, "no_lactate"
        if str(l.get("trend", "")).lower() != "rising":
            return f, False, "lactate_not_rising"
        l["trend"] = "stable"
        if l.get("value_last") is not None:
            l["value_last"] = min(float(l["value_last"]), 2.0)
        l["abnormal_flag_last"] = "normal"
        return f, True, "applied"

    if scenario == "improve_creatinine":
        c = fmap.get("creatinine")
        if c is None:
            return f, False, "no_creatinine"
        t = str(c.get("trend", "")).lower()
        fl = str(c.get("abnormal_flag_last", "")).lower()
        if t != "rising" and fl != "high":
            return f, False, "creatinine_not_high_rising"
        if c.get("value_last") is not None:
            c["value_last"] = max(0.8, float(c["value_last"]) * 0.8)
        c["trend"] = "stable"
        c["abnormal_flag_last"] = "normal"
        return f, True, "applied"

    if scenario == "improve_oxygenation":
        s = fmap.get("spo2")
        r = fmap.get("rr")
        changed = False
        if s is not None and s.get("value_last") is not None and float(s.get("value_last")) < 92:
            s["value_last"] = 95.0
            s["abnormal_flag_last"] = "normal"
            changed = True
        if r is not None and r.get("value_last") is not None and float(r.get("value_last")) > 24:
            r["value_last"] = 20.0
            r["abnormal_flag_last"] = "normal"
            changed = True
        if not changed:
            return f, False, "oxygenation_not_abnormal"
        return f, True, "applied"

    return f, False, "unknown_scenario"


def run(out_dir: str, protocol_json: str):
    stage1_path = os.path.join(out_dir, "stage1_router.jsonl")
    rows = [o for o in iter_jsonl(stage1_path)]
    rules = load_protocol(protocol_json)

    scenarios = [
        ("map_low_to_normal", "vasopressor_signal", -1),
        ("remove_lactate_rise", "vasopressor_signal", -1),
        ("improve_creatinine", "renal_support_signal", -1),
        ("improve_oxygenation", "resp_support_signal", -1),
    ]

    out_rows = []
    directional = []
    prob_changes = {k: [] for k in TARGETS}
    proto_change = []
    state_change = []

    for o in rows:
        facts = o.get("packet", {}).get("facts", [])
        base_probs = _risk_probs_from_facts(facts)
        base_rules = set(_active_rules(rules, facts))
        base_state = {
            "hemodynamic_state": int(base_probs["vasopressor_signal"] >= 0.5),
            "respiratory_state": int(base_probs["resp_support_signal"] >= 0.5),
            "renal_state": int(base_probs["renal_support_signal"] >= 0.5),
        }

        for sc, target, exp_dir in scenarios:
            cf_facts, applied, reason = _apply_cf(sc, facts)
            cf_probs = _risk_probs_from_facts(cf_facts)
            cf_rules = set(_active_rules(rules, cf_facts))
            cf_state = {
                "hemodynamic_state": int(cf_probs["vasopressor_signal"] >= 0.5),
                "respiratory_state": int(cf_probs["resp_support_signal"] >= 0.5),
                "renal_state": int(cf_probs["renal_support_signal"] >= 0.5),
            }

            d_target = float(cf_probs[target] - base_probs[target])
            ok_dir = int(np.sign(d_target) == exp_dir) if applied else None
            if ok_dir is not None:
                directional.append(ok_dir)

            for k in TARGETS:
                prob_changes[k].append(float(cf_probs[k] - base_probs[k]))
            proto_change.append(int(base_rules != cf_rules))
            state_change.append(int(base_state != cf_state))

            out_rows.append({
                "example_id": o.get("example_id"),
                "source_dataset": o.get("source_dataset"),
                "patient_id": o.get("patient_id"),
                "encounter_id": o.get("encounter_id"),
                "anchor_time": o.get("anchor_time"),
                "scenario": sc,
                "applied": int(applied),
                "apply_reason": reason,
                "target": target,
                "expected_direction": exp_dir,
                "base_prob_target": float(base_probs[target]),
                "cf_prob_target": float(cf_probs[target]),
                "delta_prob_target": d_target,
                "direction_consistent": ok_dir,
                "protocol_activation_changed": int(base_rules != cf_rules),
                "individual_state_changed": int(base_state != cf_state),
            })

    df = pd.DataFrame(out_rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "counterfactual_results.csv"), index=False)
    df.to_parquet(os.path.join(out_dir, "counterfactual_results.parquet"), index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_applied": int(df["applied"].sum()) if len(df) else 0,
        "directional_consistency_rate": float(np.mean(directional)) if directional else None,
        "average_probability_change": {k: (float(np.mean(v)) if v else None) for k, v in prob_changes.items()},
        "protocol_activation_change_rate": float(np.mean(proto_change)) if proto_change else None,
        "individual_protocol_state_change_rate": float(np.mean(state_change)) if state_change else None,
    }
    write_json(os.path.join(out_dir, "counterfactual_summary.json"), summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol-json", required=True)
    args = ap.parse_args()
    run(args.out_dir, args.protocol_json)

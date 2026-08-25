from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Tuple


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_ci_pair(ci_obj: Any) -> Tuple[Any, Any]:
    if isinstance(ci_obj, (list, tuple)) and len(ci_obj) >= 2:
        return ci_obj[0], ci_obj[1]
    return None, None


def mean_or_none(vals: List[Any]) -> Any:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def infer_dataset(*parts: str) -> str:
    txt = " ".join(str(x or "") for x in parts).lower()
    has_mimic = "mimic" in txt
    has_eicu = "eicu" in txt
    if has_mimic and not has_eicu:
        return "mimic"
    if has_eicu and not has_mimic:
        return "eicu"
    if has_mimic and has_eicu:
        return "mixed"
    return "unknown"


def infer_exp_family(out_dir: str) -> str:
    name = os.path.basename(out_dir).lower()
    if "single" in name:
        return "single_baseline"
    if "ablation" in name:
        return "ablation"
    if "freeform" in name:
        return "freeform"
    if "vitaltrace" in name or "trace" in name:
        return "vitaltrace"
    return "staged"


def collect_one(out_dir: str) -> Dict[str, Any] | None:
    overall = read_json(os.path.join(out_dir, "metrics_overall.json"))
    if not overall:
        return None
    if overall.get("evaluation_schema") != "corrected_evaluation_v2":
        return None

    cal = read_json(os.path.join(out_dir, "calibrated_metrics.json"))
    stg = read_json(os.path.join(out_dir, "stagewise_metrics.json"))
    cf = read_json(os.path.join(out_dir, "counterfactual_metrics.json"))
    if cf.get("evaluation_method") != "standardized_recovery_model_rerun_v2":
        cf = {}
    temporal = read_json(os.path.join(out_dir, "temporal_metrics.json"))
    pcons = read_json(os.path.join(out_dir, "protocol_consistency_metrics.json"))
    eff = read_json(os.path.join(out_dir, "efficiency_metrics.json"))
    evt = read_json(os.path.join(out_dir, "event_level_metrics.json"))

    auroc_l, auroc_u = safe_ci_pair(overall.get("macro_auroc_95ci"))
    auprc_l, auprc_u = safe_ci_pair(overall.get("macro_auprc_95ci"))

    labels = [
        ("vasopressor_signal", "vaso"),
        ("resp_support_signal", "resp"),
        ("renal_support_signal", "renal"),
        ("any_deterioration", "deterioration"),
    ]
    per_label_aurocs: List[Any] = []
    per_label_auprcs: List[Any] = []
    per_target: Dict[str, Any] = {}
    for lbl, short in labels:
        dd = evt.get(lbl, {}) if isinstance(evt.get(lbl, {}), dict) else {}
        per_label_aurocs.append(dd.get("auroc"))
        per_label_auprcs.append(dd.get("auprc"))
        per_target[f"{short}_auroc"] = dd.get("auroc")
        per_target[f"{short}_auprc"] = dd.get("auprc")
        per_target[f"{short}_f1"] = dd.get("f1")
        per_target[f"{short}_ece"] = dd.get("ece")
        per_target[f"{short}_brier"] = dd.get("brier")

    row = {
        "run_id": os.path.basename(out_dir),
        "exp_family": infer_exp_family(out_dir),
        "dataset": infer_dataset(out_dir),
        "out_dir": out_dir,
        "n_examples": overall.get("n_examples"),
        "macro_auroc": overall.get("macro_auroc"),
        "macro_auroc_ci_low": auroc_l,
        "macro_auroc_ci_high": auroc_u,
        "macro_auprc": overall.get("macro_auprc"),
        "macro_auprc_ci_low": auprc_l,
        "macro_auprc_ci_high": auprc_u,
        "bootstrap_replicates": overall.get("bootstrap_replicates"),
        "bootstrap_sampling": overall.get("bootstrap_sampling"),
        "micro_f1": overall.get("micro_f1"),
        "micro_ece": overall.get("micro_ece"),
        "micro_brier": overall.get("micro_brier"),
        "cal_micro_f1": cal.get("micro_f1"),
        "cal_micro_ece": cal.get("micro_ece"),
        "cal_micro_precision": cal.get("micro_precision"),
        "cal_micro_recall": cal.get("micro_recall"),
        "auditor_fail_rate": (stg.get("stage3", {}) or {}).get("fail_rate"),
        "auditor_pass_rate": (stg.get("stage3", {}) or {}).get("pass_rate"),
        "recall_at_6h": temporal.get("recall_at_6h"),
        "recall_at_12h": temporal.get("recall_at_12h"),
        "lead_time_h_mean": temporal.get("lead_time_h_mean"),
        "directional_consistency_rate": cf.get("directional_consistency_rate"),
        "counterfactual_evaluation_method": cf.get("evaluation_method"),
        "rule_violation_rate": pcons.get("rule_violation_rate"),
        "rule_activation_precision": pcons.get("rule_activation_precision"),
        "rule_activation_recall": pcons.get("rule_activation_recall"),
        "llm_calls_per_example": eff.get("llm_calls_per_example"),
        "llm_failure_rate_per_call": eff.get("llm_failure_rate_per_call"),
        "per_label_mean_auroc": mean_or_none(per_label_aurocs),
        "per_label_mean_auprc": mean_or_none(per_label_auprcs),
    }
    row.update(per_target)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", required=True, help="Root folder containing experiment subfolders.")
    ap.add_argument("--out-csv", required=True, help="Path to output CSV summary.")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(args.experiments_root, "**", "metrics_overall.json")
    for p in glob.glob(pattern, recursive=True):
        out_dir = os.path.dirname(p)
        r = collect_one(out_dir)
        if r:
            rows.append(r)

    rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("run_id", ""))))
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        else:
            w = csv.DictWriter(f, fieldnames=["run_id"])
            w.writeheader()

    print(f"saved: {args.out_csv}")
    print(f"n_rows: {len(rows)}")


if __name__ == "__main__":
    main()

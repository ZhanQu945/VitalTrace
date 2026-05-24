from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict


def rjson(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def main(jobs_tsv: str, out_csv: str):
    rows = []
    with open(jobs_tsv, "r") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobid, run_id, exp_family, out_dir = line.split("\t")
            overall = rjson(os.path.join(out_dir, "metrics_overall.json"))
            cal = rjson(os.path.join(out_dir, "calibrated_metrics.json"))
            stg = rjson(os.path.join(out_dir, "stagewise_metrics.json"))
            cf = rjson(os.path.join(out_dir, "counterfactual_metrics.json"))
            eff = rjson(os.path.join(out_dir, "efficiency_metrics.json"))
            rows.append(
                {
                    "jobid": jobid,
                    "run_id": run_id,
                    "exp_family": exp_family,
                    "out_dir": out_dir,
                    "macro_auroc": overall.get("macro_auroc"),
                    "macro_auprc": overall.get("macro_auprc"),
                    "micro_f1": overall.get("micro_f1"),
                    "micro_ece": overall.get("micro_ece"),
                    "micro_brier": overall.get("micro_brier"),
                    "cal_micro_f1": cal.get("micro_f1"),
                    "cal_micro_ece": cal.get("micro_ece"),
                    "auditor_fail_rate": (stg.get("stage3", {}) or {}).get("fail_rate"),
                    "auditor_pass_rate": (stg.get("stage3", {}) or {}).get("pass_rate"),
                    "directional_consistency_rate": cf.get("directional_consistency_rate"),
                    "protocol_activation_change_rate": cf.get("protocol_activation_change_rate"),
                    "llm_calls_total": eff.get("llm_calls_total"),
                    "llm_failures_total": eff.get("llm_failures_total"),
                    "llm_calls_per_example": eff.get("llm_calls_per_example"),
                }
            )
    rows = sorted(rows, key=lambda x: x["run_id"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["run_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved: {out_csv}")
    print(f"n_rows: {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-tsv", required=True)
    ap.add_argument("--out-csv", required=True)
    ns = ap.parse_args()
    main(ns.jobs_tsv, ns.out_csv)

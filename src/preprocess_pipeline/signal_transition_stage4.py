from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.preprocess_pipeline.signal_transition_common import TARGET_CONCEPTS, map_intervention, print_examples, print_stage_header


def read_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def labels_for_step(step_time, fut: pd.DataFrame):
    concepts = set()
    for arr in fut["intervention_concepts"].tolist():
        for c in arr:
            concepts.add(c)
    y_vaso = int(len(concepts & TARGET_CONCEPTS["vasopressor_signal"]) > 0)
    y_resp = int(len(concepts & TARGET_CONCEPTS["resp_support_signal"]) > 0)
    y_renal = int(len(concepts & TARGET_CONCEPTS["renal_support_signal"]) > 0)
    y_any = int(y_vaso or y_resp or y_renal)
    return {
        "vasopressor_signal": y_vaso,
        "resp_support_signal": y_resp,
        "renal_support_signal": y_renal,
        "any_deterioration": y_any,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-steps", required=True)
    ap.add_argument("--input-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizons", default="6,12,24,48")
    ap.add_argument("--progress-every-patients", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header("STAGE 4: INTERVENTION LABEL GENERATION")

    steps = pd.DataFrame(read_jsonl(args.input_steps))
    ev = pd.read_parquet(args.input_events)
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev[ev["event_time"].notna()].copy()

    # ensure intervention concepts present
    if "intervention_concepts" not in ev.columns:
        concepts = []
        provs = []
        for _, r in ev.iterrows():
            if str(r.get("event_type", "")) in {"medication", "procedure"}:
                m = map_intervention(str(r.get("event_name", "")), str(r.get("payload_json", "")))
                concepts.append(m.concepts)
                provs.append(m.provenance)
            else:
                concepts.append([])
                provs.append("na")
        ev["intervention_concepts"] = concepts
        ev["intervention_match_provenance"] = provs

    steps["anchor_time"] = pd.to_datetime(steps["anchor_time"], errors="coerce")
    key = ["source_dataset", "patient_id", "encounter_id"]

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    summary = {"horizons": {}, "n_steps_input": int(len(steps))}

    for h in horizons:
        out_rows = []
        groups = list(steps.groupby(key))
        n_groups = len(groups)
        t0 = time.time()
        for gi, (k, sg) in enumerate(groups, start=1):
            if args.progress_every_patients > 0 and (gi % args.progress_every_patients == 0):
                print(
                    f"[progress][stage4][h{h}] trajectories={gi}/{n_groups} rows_emitted={len(out_rows)} elapsed_s={time.time()-t0:.1f}",
                    flush=True,
                )
            eg = ev[(ev["source_dataset"] == k[0]) & (ev["patient_id"] == k[1]) & (ev["encounter_id"] == k[2])]
            eg = eg[eg["event_type"].isin(["medication", "procedure"])].copy()
            for _, r in sg.sort_values("anchor_time").iterrows():
                t = r["anchor_time"]
                fut = eg[(eg["event_time"] > t) & (eg["event_time"] <= t + pd.Timedelta(hours=h))]
                y = labels_for_step(t, fut)
                rec = dict(r)
                rec["horizon_hours"] = int(h)
                rec["targets"] = y
                out_rows.append(rec)

        out_df = pd.DataFrame(out_rows)
        out_path = os.path.join(args.out_dir, f"stage4_labeled_steps_h{h}.jsonl")
        with open(out_path, "w") as f:
            for _, r in out_df.iterrows():
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

        prev = {
            lab: float(out_df["targets"].map(lambda x: x.get(lab, 0)).mean()) if len(out_df) else 0.0
            for lab in ["vasopressor_signal", "resp_support_signal", "renal_support_signal", "any_deterioration"]
        }
        summary["horizons"][str(h)] = {
            "n_rows": int(len(out_df)),
            "prevalence": prev,
        }
        print(f"h={h} rows={len(out_df):,} prevalence={ {k: round(v,4) for k,v in prev.items()} }")

        pos = out_df[out_df["targets"].map(lambda x: x.get("any_deterioration", 0) == 1)]
        neg = out_df[out_df["targets"].map(lambda x: x.get("any_deterioration", 0) == 0)]
        print_examples(pos, f"h{h} strict positive", ["source_dataset", "patient_id", "encounter_id", "step_id", "anchor_time"]) 
        print_examples(neg, f"h{h} negative", ["source_dataset", "patient_id", "encounter_id", "step_id", "anchor_time"]) 

    out_summary = os.path.join(args.out_dir, "stage4_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.preprocess_pipeline.signal_transition_common import (
    CORE_FEATURES,
    THRESHOLDS,
    canonical_feature,
    print_examples,
    print_stage_header,
    state_from_value,
)


def extreme_value(feature: str, vals: pd.Series):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if len(v) == 0:
        return None
    if feature in {"map", "spo2", "bicarbonate", "sodium", "potassium", "glucose"}:
        lo, hi = THRESHOLDS.get(feature, (None, None))
        if lo is None or hi is None:
            return float(v.iloc[-1])
        mid = (lo + hi) / 2.0
        idx = (v - mid).abs().idxmax()
        return float(v.loc[idx])
    if feature in {"rr", "lactate", "creatinine", "wbc", "temp", "hr"}:
        return float(v.max())
    return float(v.iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--progress-every-patients", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header("STAGE 3: TRANSITION-BASED STEP CONSTRUCTION")

    df = pd.read_parquet(args.input_events)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[df["event_time"].notna()].copy()
    df["feature"] = df["event_name"].map(canonical_feature)
    sig = df[df["feature"].isin(CORE_FEATURES)].copy()
    sig["value_num"] = pd.to_numeric(sig["value_num"], errors="coerce")

    out_rows = []
    transitions = defaultdict(int)
    steps_per_traj = []

    key_cols = ["source_dataset", "patient_id", "encounter_id"]
    traj_groups = list(sig.sort_values(key_cols + ["event_time"]).groupby(key_cols))
    n_traj = len(traj_groups)
    t0 = time.time()
    for idx, (key, g) in enumerate(traj_groups, start=1):
        if args.progress_every_patients > 0 and (idx % args.progress_every_patients == 0):
            print(
                f"[progress][stage3] trajectories={idx}/{n_traj} steps_emitted={len(out_rows)} elapsed_s={time.time()-t0:.1f}",
                flush=True,
            )
        g = g.sort_values("event_time")
        current_state = {f: "missing" for f in CORE_FEATURES}
        last_seen_time = {f: None for f in CORE_FEATURES}
        step_events = []
        step_id = 0

        for _, r in g.iterrows():
            f = r["feature"]
            v = r["value_num"]
            t = r["event_time"]
            new_state = state_from_value(f, v)
            changed = new_state != current_state.get(f, "missing")

            if changed and step_events:
                # flush old step
                sg = pd.DataFrame(step_events)
                rec = {
                    "source_dataset": key[0], "patient_id": int(key[1]), "encounter_id": int(key[2]),
                    "step_id": int(step_id),
                    "anchor_time": str(sg["event_time"].max()),
                    "t_start": str(sg["event_time"].min()),
                    "t_end": str(sg["event_time"].max()),
                    "n_events": int(len(sg)),
                }
                for sf in sorted(CORE_FEATURES):
                    sf_rows = sg[sg["feature"] == sf]
                    rec[f"{sf}_value"] = extreme_value(sf, sf_rows["value_num"]) if len(sf_rows) else None
                    rec[f"{sf}_measured_in_step"] = int(len(sf_rows) > 0)
                    rec[f"{sf}_n_meas"] = int(len(sf_rows))
                out_rows.append(rec)
                step_id += 1
                step_events = []

            if changed:
                transitions[f"{f}:{current_state.get(f,'missing')}->{new_state}"] += 1
                current_state[f] = new_state

            step_events.append({
                "event_time": t,
                "feature": f,
                "value_num": v,
            })
            last_seen_time[f] = t

        if step_events:
            sg = pd.DataFrame(step_events)
            rec = {
                "source_dataset": key[0], "patient_id": int(key[1]), "encounter_id": int(key[2]),
                "step_id": int(step_id),
                "anchor_time": str(sg["event_time"].max()),
                "t_start": str(sg["event_time"].min()),
                "t_end": str(sg["event_time"].max()),
                "n_events": int(len(sg)),
            }
            for sf in sorted(CORE_FEATURES):
                sf_rows = sg[sg["feature"] == sf]
                rec[f"{sf}_value"] = extreme_value(sf, sf_rows["value_num"]) if len(sf_rows) else None
                rec[f"{sf}_measured_in_step"] = int(len(sf_rows) > 0)
                rec[f"{sf}_n_meas"] = int(len(sf_rows))
            out_rows.append(rec)
            step_id += 1

        steps_per_traj.append(step_id)

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["source_dataset", "patient_id", "encounter_id", "step_id"]) if not out.empty else out

    s = pd.Series(steps_per_traj) if steps_per_traj else pd.Series(dtype=float)
    stats = {
        "n_trajectories": int(len(steps_per_traj)),
        "n_steps": int(len(out)),
        "steps_per_trajectory_mean": float(s.mean()) if len(s) else 0.0,
        "steps_per_trajectory_median": float(s.median()) if len(s) else 0.0,
        "steps_per_trajectory_p95": float(s.quantile(0.95)) if len(s) else 0.0,
        "steps_per_trajectory_max": int(s.max()) if len(s) else 0,
        "transition_counts": dict(sorted(transitions.items(), key=lambda kv: kv[1], reverse=True)[:200]),
    }

    print(f"trajectories={stats['n_trajectories']:,} steps={stats['n_steps']:,}")
    print("steps/trajectory:", {k: stats[k] for k in ["steps_per_trajectory_mean", "steps_per_trajectory_median", "steps_per_trajectory_p95", "steps_per_trajectory_max"]})
    print("top transitions:", dict(list(stats["transition_counts"].items())[:20]))

    print_examples(out, "stable/merged sample", ["source_dataset", "patient_id", "encounter_id", "step_id", "anchor_time", "map_value", "rr_value", "spo2_value"])
    print_examples(out.sort_values("rr_value", ascending=False, na_position="last"), "deterioration sample", ["source_dataset", "patient_id", "encounter_id", "step_id", "rr_value", "spo2_value", "map_value"])
    print_examples(out.sort_values("spo2_value", ascending=False, na_position="last"), "recovery-like sample", ["source_dataset", "patient_id", "encounter_id", "step_id", "spo2_value", "rr_value", "map_value"])

    out_jsonl = os.path.join(args.out_dir, "stage3_transition_steps.jsonl")
    out_summary = os.path.join(args.out_dir, "stage3_summary.json")

    with open(out_jsonl, "w") as f:
        for _, r in out.iterrows():
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
    with open(out_summary, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"saved: {out_jsonl}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()

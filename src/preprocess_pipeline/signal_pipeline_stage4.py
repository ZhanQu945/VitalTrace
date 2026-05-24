from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from src.preprocess_pipeline.signal_transition_common import print_examples, print_stage_header


KEY = ["source_dataset", "patient_id", "encounter_id"]


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "eicu"])
    ap.add_argument("--stage3-steps", required=True)
    ap.add_argument("--stage3-labeled-main", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--require-full-core-signals", action="store_true")
    ap.add_argument("--full-core-mode", choices=["all_steps", "any_step"], default="all_steps")
    ap.add_argument("--full-core-fallback", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--export-latent-debug-jsonl", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header(f"STAGE 4 ({args.dataset}): Debug Subset Sampling")

    steps = read_jsonl(args.stage3_steps)
    lab = read_jsonl(args.stage3_labeled_main)

    if steps.empty or lab.empty:
        raise RuntimeError("Stage4 sampling received empty stage3 inputs.")

    core_cols = ["map_value", "rr_value", "spo2_value", "creatinine_value", "lactate_value"]
    full_filter_applied = False
    eligible_enc_before = int(steps[KEY].drop_duplicates().shape[0])
    eligible_enc_after = eligible_enc_before

    if args.require_full_core_signals:
        for c in core_cols:
            if c not in steps.columns:
                raise RuntimeError(f"Missing required core column in stage3 steps: {c}")
        full_filter_applied = True
        row_full = steps[core_cols].notna().all(axis=1)
        if args.full_core_mode == "all_steps":
            mask_df = (
                steps.assign(_row_full=row_full)
                .groupby(KEY, as_index=False)["_row_full"]
                .all()
                .rename(columns={"_row_full": "_enc_ok"})
            )
        else:
            mask_df = (
                steps.assign(_row_full=row_full)
                .groupby(KEY, as_index=False)["_row_full"]
                .any()
                .rename(columns={"_row_full": "_enc_ok"})
            )
        keep_enc = mask_df[mask_df["_enc_ok"]][KEY]
        eligible_enc_after = int(keep_enc.drop_duplicates().shape[0])
        if eligible_enc_after == 0 and args.full_core_fallback:
            full_filter_applied = False
            eligible_enc_after = eligible_enc_before
        else:
            steps = steps.merge(keep_enc, on=KEY, how="inner")
            lab = lab.merge(keep_enc, on=KEY, how="inner")
            if eligible_enc_after == 0:
                raise RuntimeError(
                    f"No encounters remain after full-core filter mode={args.full_core_mode}. "
                    "Relax filter or regenerate stage3 with denser carry-forward."
                )

    # Rank encounters by (1) intervention label density, (2) full-signal coverage, (3) trajectory richness.
    core_cols = ["map_value", "rr_value", "spo2_value", "creatinine_value", "lactate_value"]
    step_stats = (
        steps.groupby(KEY, as_index=False)
        .agg(
            n_steps=("step_id", "count"),
            full_core_steps=("step_id", lambda s: 0),  # placeholder filled below
            core_obs_sum=("step_id", lambda s: 0),     # placeholder filled below
        )
    )
    core_nonmissing = steps[core_cols].notna().sum(axis=1) if set(core_cols).issubset(set(steps.columns)) else pd.Series([0] * len(steps))
    tmp = steps[KEY].copy()
    tmp["core_nonmissing"] = core_nonmissing.values
    tmp["is_full_core"] = (tmp["core_nonmissing"] == 5).astype(int)
    cov = tmp.groupby(KEY, as_index=False).agg(full_core_steps=("is_full_core", "sum"), core_obs_sum=("core_nonmissing", "sum"))
    step_stats = step_stats.drop(columns=["full_core_steps", "core_obs_sum"]).merge(cov, on=KEY, how="left")
    step_stats["full_core_steps"] = step_stats["full_core_steps"].fillna(0).astype(int)
    step_stats["core_obs_sum"] = step_stats["core_obs_sum"].fillna(0).astype(int)
    step_stats["full_core_step_ratio"] = step_stats["full_core_steps"] / step_stats["n_steps"].clip(lower=1)

    def _any_pos(t):
        return int(isinstance(t, dict) and int(t.get("any_deterioration", 0)) == 1)

    lab_enc = lab.copy()
    lab_enc["any_pos"] = lab_enc["targets"].map(_any_pos) if "targets" in lab_enc.columns else 0
    label_stats = lab_enc.groupby(KEY, as_index=False).agg(
        n_label_rows=("any_pos", "count"),
        n_any_pos=("any_pos", "sum"),
    )
    label_stats["label_pos_ratio"] = label_stats["n_any_pos"] / label_stats["n_label_rows"].clip(lower=1)

    rank_df = step_stats.merge(label_stats, on=KEY, how="left")
    for c in ["n_label_rows", "n_any_pos", "label_pos_ratio"]:
        rank_df[c] = rank_df[c].fillna(0)
    rank_df["rank_score"] = (
        4.0 * rank_df["label_pos_ratio"]
        + 2.0 * rank_df["full_core_step_ratio"]
        + 0.5 * (rank_df["n_steps"].clip(upper=200) / 200.0)
    )
    rank_df = rank_df.sort_values(
        ["rank_score", "n_any_pos", "full_core_step_ratio", "n_steps"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    top_k = min(int(args.top_k), len(rank_df))
    enc = rank_df.head(top_k)[KEY].copy()
    # If top_k < sample_n requested, continue by ranked fill-in.
    n = min(int(args.sample_n), len(rank_df))
    if n > top_k:
        enc = rank_df.head(n)[KEY].copy()

    steps_s = steps.merge(enc, on=KEY, how="inner")
    lab_s = lab.merge(enc, on=KEY, how="inner")

    out_steps = os.path.join(args.out_dir, "stage4_sample_steps.jsonl")
    out_lab = os.path.join(args.out_dir, "stage4_sample_labeled_main.jsonl")
    out_summary = os.path.join(args.out_dir, "stage4_sample_summary.json")
    out_rank = os.path.join(args.out_dir, "stage4_ranked_encounters.csv")

    write_jsonl(steps_s, out_steps)
    write_jsonl(lab_s, out_lab)
    rank_df.to_csv(out_rank, index=False)

    out_latent_debug = None
    if args.export_latent_debug_jsonl:
        # Export a latent-pipeline-compatible JSONL using step snapshots as protocol_observations.
        obs_features = [
            "map", "rr", "spo2", "creatinine", "lactate",
            "hr", "temp", "wbc", "bicarbonate", "sodium", "potassium", "glucose",
        ]
        by_key = defaultdict(list)
        for _, r in steps_s.sort_values(KEY + ["step_id"]).iterrows():
            k = (r["source_dataset"], r["patient_id"], r["encounter_id"])
            by_key[k].append(r.to_dict())
        tgt_map = {
            (r["source_dataset"], r["patient_id"], r["encounter_id"], int(r.get("step_id", 0))): r.get("targets", {})
            for _, r in lab_s.iterrows()
        }
        out_latent_debug = os.path.join(args.out_dir, "stage4_sample_for_latent_debug.jsonl")
        with open(out_latent_debug, "w") as f:
            for k, rows in by_key.items():
                prev_vals = {}
                for row in rows:
                    obs = []
                    for feat in obs_features:
                        v = row.get(f"{feat}_value")
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            continue
                        trend = "stable"
                        if feat in prev_vals:
                            if v > prev_vals[feat]:
                                trend = "rising"
                            elif v < prev_vals[feat]:
                                trend = "decreasing"
                        prev_vals[feat] = v
                        obs.append({
                            "feature": feat,
                            "value_last": float(v),
                            "value_mean": float(v),
                            "trend": trend,
                            "abnormal_flag_last": "abnormal" if feat in {"map", "rr", "spo2", "creatinine", "lactate"} else "normal",
                        })
                    step_id = int(row.get("step_id", 0))
                    ex = {
                        "example_id": f"{k[0]}_{k[1]}_{k[2]}_{step_id}",
                        "source_dataset": k[0],
                        "patient_id": k[1],
                        "encounter_id": k[2],
                        "anchor_time": row.get("anchor_time"),
                        "step_id": step_id,
                        "protocol_observations": obs,
                        "targets": tgt_map.get((k[0], k[1], k[2], step_id), {"any_deterioration": 0}),
                    }
                    f.write(json.dumps(ex, default=str) + "\n")

    summary = {
        "dataset": args.dataset,
        "sample_n_requested": int(args.sample_n),
        "sample_n_selected": int(n),
        "top_k": int(top_k),
        "steps_selected": int(len(steps_s)),
        "labeled_rows_selected": int(len(lab_s)),
        "full_core_filter_applied": bool(full_filter_applied),
        "full_core_mode": args.full_core_mode if full_filter_applied else None,
        "eligible_encounters_before_filter": eligible_enc_before,
        "eligible_encounters_after_filter": eligible_enc_after,
        "ranked_csv": out_rank,
        "latent_debug_jsonl": out_latent_debug,
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    print_examples(enc, "sampled encounters", KEY)
    print_examples(steps_s, "sample steps", ["source_dataset", "patient_id", "encounter_id", "step_id", "anchor_time"])
    print_examples(lab_s, "sample labels", ["source_dataset", "patient_id", "encounter_id", "step_id", "horizon_hours", "targets"])

    print(f"saved: {out_steps}")
    print(f"saved: {out_lab}")
    print(f"saved: {out_rank}")
    if out_latent_debug:
        print(f"saved: {out_latent_debug}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()

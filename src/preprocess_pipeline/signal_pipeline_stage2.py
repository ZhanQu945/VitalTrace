from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from data.preprocess_longitudinal import extract_mimic_labs, extract_mimic_vitals, extract_eicu_labs, extract_eicu_vitals
from src.preprocess_pipeline.signal_transition_common import CORE_FEATURES, canonical_feature, print_examples, print_stage_header, state_from_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "eicu"])
    ap.add_argument("--mimic-root", default="./data/mimic")
    ap.add_argument("--eicu-root", default="./data/eicu")
    ap.add_argument("--stage1-cohort", required=True)
    ap.add_argument("--stage1-interventions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-signal-features", type=int, default=4)
    ap.add_argument("--progress-every-patients", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header(f"STAGE 2 ({args.dataset}): Signal Extraction + Quality/Consistency")
    t0 = time.time()

    cohort = pd.read_parquet(args.stage1_cohort)
    iv = pd.read_parquet(args.stage1_interventions)
    print(f"loaded stage1 cohort rows={len(cohort):,} interventions rows={len(iv):,}")

    if args.dataset == "mimic":
        print("loading raw MIMIC labs...")
        t_labs = time.time()
        labs = extract_mimic_labs(args.mimic_root, top_n_labs=0)
        print(f"loaded MIMIC labs rows={len(labs):,} elapsed_s={time.time()-t_labs:.1f}")
        print("loading raw MIMIC vitals...")
        t_vit = time.time()
        vit = extract_mimic_vitals(args.mimic_root, max_vitals_per_hour=0)
        print(f"loaded MIMIC vitals rows={len(vit):,} elapsed_s={time.time()-t_vit:.1f}")
    else:
        print("loading raw eICU labs...")
        t_labs = time.time()
        labs = extract_eicu_labs(args.eicu_root)
        print(f"loaded eICU labs rows={len(labs):,} elapsed_s={time.time()-t_labs:.1f}")
        print("loading raw eICU vitals...")
        t_vit = time.time()
        vit = extract_eicu_vitals(args.eicu_root, max_vitals_per_hour=0)
        print(f"loaded eICU vitals rows={len(vit):,} elapsed_s={time.time()-t_vit:.1f}")
    sig = pd.concat([labs, vit], ignore_index=True)
    print(f"raw signal rows loaded={len(sig):,} elapsed_s={time.time()-t0:.1f}")

    k = cohort[["source_dataset", "patient_id", "encounter_id"]].copy()
    print("joining signals to stage1 cohort keys...")
    sig = sig.merge(k, on=["source_dataset", "patient_id", "encounter_id"], how="inner")
    print(f"post cohort join signal rows={len(sig):,}")
    print("canonicalizing signal feature names...")
    sig["feature"] = sig["event_name"].map(canonical_feature)
    sig = sig[sig["feature"].isin(CORE_FEATURES)].copy()
    sig["value_num"] = pd.to_numeric(sig["value_num"], errors="coerce")
    print(f"post core-feature filter signal rows={len(sig):,}")

    # coverage filter
    key = ["source_dataset", "patient_id", "encounter_id"]
    print("computing per-encounter signal feature coverage...")
    cov = sig.groupby(key)["feature"].nunique().reset_index(name="n_signal_features")
    keep = cohort.merge(cov, on=key, how="left")
    keep["n_signal_features"] = keep["n_signal_features"].fillna(0).astype(int)
    keep = keep[keep["n_signal_features"] >= int(args.min_signal_features)].copy()
    print(f"post coverage filter encounters={len(keep):,}")

    print("restricting signals/interventions to retained coverage cohort...")
    sig = sig.merge(keep[key], on=key, how="inner")
    iv = iv.merge(keep[key], on=key, how="inner")
    print(f"rows after coverage restrict: signals={len(sig):,} interventions={len(iv):,}")

    # consistency: abnormal but no intervention at encounter level
    # Compute states in encounter batches so we can log progress regularly.
    key = ["source_dataset", "patient_id", "encounter_id"]
    sig = sig.sort_values(key + ["event_time"]).copy()
    enc = sig[key].drop_duplicates().reset_index(drop=True)
    total_enc = len(enc)
    step_n = max(1, int(args.progress_every_patients))
    chunks = []
    for start in range(0, total_enc, step_n):
        end = min(start + step_n, total_enc)
        kchunk = enc.iloc[start:end]
        schunk = sig.merge(kchunk, on=key, how="inner")
        schunk["state"] = schunk.apply(lambda r: state_from_value(str(r["feature"]), r["value_num"]), axis=1)
        chunks.append(schunk)
        if end % step_n == 0 or end == total_enc:
            print(f"progress encounters processed={end:,}/{total_enc:,}")
    sig = pd.concat(chunks, ignore_index=True)
    abn = sig[sig["state"].isin(["low", "high"])].groupby(key).size().reset_index(name="n_abn")
    ints = iv.groupby(key).size().reset_index(name="n_int")
    qc = keep.merge(abn, on=key, how="left").merge(ints, on=key, how="left")
    qc[["n_abn", "n_int"]] = qc[["n_abn", "n_int"]].fillna(0)
    bad = qc[(qc["n_abn"] > 0) & (qc["n_int"] == 0)]
    good = qc[~((qc["n_abn"] > 0) & (qc["n_int"] == 0))].copy()
    print(f"post consistency filter encounters={len(good):,} removed={len(bad):,}")

    sig = sig.merge(good[key], on=key, how="inner")
    iv = iv.merge(good[key], on=key, how="inner")
    keep = keep.merge(good[key], on=key, how="inner")

    events = pd.concat([sig, iv], ignore_index=True).sort_values(["source_dataset", "patient_id", "encounter_id", "event_time"])

    print(f"retained encounters={len(keep):,} signal rows={len(sig):,} intervention rows={len(iv):,}")
    print(f"removed inconsistent encounters={len(bad):,}")
    print("signal coverage:", sig.groupby("feature")[key].apply(lambda g: g.drop_duplicates().shape[0]).to_dict())

    print_examples(keep, "retained encounters", ["source_dataset", "patient_id", "encounter_id", "los_h", "mortality", "n_signal_features"])
    print_examples(sig.sort_values("event_time"), "signal rows", ["source_dataset", "patient_id", "encounter_id", "event_time", "feature", "value_num", "abnormal_flag"])
    print_examples(bad, "removed inconsistent", ["source_dataset", "patient_id", "encounter_id", "n_abn", "n_int"])

    cohort_path = os.path.join(args.out_dir, "stage2_cohort.parquet")
    events_path = os.path.join(args.out_dir, "stage2_events.parquet")
    summary_path = os.path.join(args.out_dir, "stage2_summary.json")

    keep.to_parquet(cohort_path, index=False)
    events.to_parquet(events_path, index=False)
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "cohort_input": int(len(cohort)),
            "cohort_retained": int(len(keep)),
            "signal_rows_retained": int(len(sig)),
            "intervention_rows_retained": int(len(iv)),
            "removed_inconsistent_encounters": int(len(bad)),
        }, f, indent=2)

    print(f"saved: {cohort_path}")
    print(f"saved: {events_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()

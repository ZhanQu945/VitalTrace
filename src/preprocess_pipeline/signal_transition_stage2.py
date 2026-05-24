from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.preprocess_pipeline.signal_transition_common import (
    CORE_FEATURES,
    TARGET_CONCEPTS,
    canonical_feature,
    map_intervention,
    print_examples,
    print_stage_header,
    state_from_value,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-signal-features", type=int, default=4)
    ap.add_argument("--require-any-intervention", action="store_true")
    ap.add_argument("--progress-every-patients", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header("STAGE 2: SIGNAL/INTERVENTION CONSISTENCY FILTER")

    df = pd.read_parquet(args.input_events)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[df["event_time"].notna()].copy()

    df["feature"] = df["event_name"].map(canonical_feature)
    signal_mask = df["feature"].isin(CORE_FEATURES)
    intervention_mask = df["event_type"].astype(str).isin(["medication", "procedure"])
    keep = df[signal_mask | intervention_mask].copy()

    # intervention concept mapping
    t0 = time.time()
    total_patients = keep[["source_dataset", "patient_id"]].drop_duplicates().shape[0]
    seen_patients = set()
    concepts, provs = [], []
    for i, (_, r) in enumerate(keep.iterrows(), start=1):
        pkey = (str(r.get("source_dataset", "")), str(r.get("patient_id", "")))
        if pkey not in seen_patients:
            seen_patients.add(pkey)
            if args.progress_every_patients > 0 and (len(seen_patients) % args.progress_every_patients == 0):
                print(
                    f"[progress][stage2] patients={len(seen_patients)}/{total_patients} rows={i}/{len(keep)} elapsed_s={time.time()-t0:.1f}",
                    flush=True,
                )
        if str(r.get("event_type", "")) in {"medication", "procedure"}:
            m = map_intervention(str(r.get("event_name", "")), str(r.get("payload_json", "")))
            concepts.append(m.concepts)
            provs.append(m.provenance)
        else:
            concepts.append([])
            provs.append("na")
    keep["intervention_concepts"] = concepts
    keep["intervention_match_provenance"] = provs

    key = ["source_dataset", "patient_id", "encounter_id"]

    # per-encounter signal coverage
    sig_cov = keep[keep["feature"].isin(CORE_FEATURES)].groupby(key)["feature"].nunique().reset_index(name="n_signal_features")

    # per-encounter intervention availability
    inter = keep[keep["intervention_concepts"].map(lambda x: len(x) > 0)].groupby(key).size().reset_index(name="n_interventions")

    enc = keep[key].drop_duplicates()
    enc = enc.merge(sig_cov, on=key, how="left").merge(inter, on=key, how="left")
    enc["n_signal_features"] = enc["n_signal_features"].fillna(0).astype(int)
    enc["n_interventions"] = enc["n_interventions"].fillna(0).astype(int)

    good = enc[enc["n_signal_features"] >= int(args.min_signal_features)].copy()
    if args.require_any_intervention:
        good = good[good["n_interventions"] > 0].copy()

    out = keep.merge(good[key], on=key, how="inner")

    # stats
    print(f"input events={len(df):,} | kept signal/intervention rows={len(keep):,} | retained rows={len(out):,}")
    print("event type counts (retained):", out["event_type"].value_counts().to_dict())

    # coverage
    cov = out[out["feature"].isin(CORE_FEATURES)].groupby("feature")[key].apply(lambda g: g.drop_duplicates().shape[0]).to_dict()
    n_enc = out[key].drop_duplicates().shape[0]
    cov_rate = {k: float(v / n_enc) if n_enc else 0.0 for k, v in cov.items()}
    print("signal coverage rates:", {k: round(v, 4) for k, v in sorted(cov_rate.items())})

    # provenance stats
    prov = out[out["event_type"].isin(["medication", "procedure"])]["intervention_match_provenance"].value_counts().to_dict()
    print("intervention match provenance:", prov)

    # abnormal episodes with no intervention concept at encounter level
    sig = out[out["feature"].isin(CORE_FEATURES)].copy()
    sig["value_num"] = pd.to_numeric(sig["value_num"], errors="coerce")
    sig["state"] = sig.apply(lambda r: state_from_value(str(r["feature"]), r["value_num"]), axis=1)
    abn = sig[sig["state"].isin(["low", "high"])].groupby(key).size().reset_index(name="n_abn")
    inter_e = out[out["intervention_concepts"].map(lambda x: len(x) > 0)].groupby(key).size().reset_index(name="n_int")
    cons = enc.merge(abn, on=key, how="left").merge(inter_e, on=key, how="left")
    cons[["n_abn", "n_int"]] = cons[["n_abn", "n_int"]].fillna(0)
    bad_cons = cons[(cons["n_abn"] > 0) & (cons["n_int"] == 0)]
    print(f"inconsistent encounters (abnormal signals but zero mapped interventions): {len(bad_cons):,}")

    print_examples(good, "retained encounters", ["source_dataset", "patient_id", "encounter_id", "n_signal_features", "n_interventions"])
    print_examples(out[out["intervention_match_provenance"].isin(["code_exact", "code_normalized"])], "code-first match examples", ["source_dataset", "patient_id", "encounter_id", "event_name", "intervention_match_provenance"])
    print_examples(out[(out["event_type"].isin(["medication", "procedure"])) & (out["intervention_concepts"].map(len) == 0)], "unmatched intervention examples", ["source_dataset", "patient_id", "encounter_id", "event_name", "payload_json"])

    audit = out[out["event_type"].isin(["medication", "procedure"])][key + ["event_time", "event_name", "payload_json", "intervention_concepts", "intervention_match_provenance"]].copy()

    out_events = os.path.join(args.out_dir, "stage2_filtered_events.parquet")
    out_summary = os.path.join(args.out_dir, "stage2_summary.json")
    out_audit = os.path.join(args.out_dir, "stage2_intervention_mapping_audit.csv")
    out.to_parquet(out_events, index=False)
    audit.to_csv(out_audit, index=False)
    with open(out_summary, "w") as f:
        json.dump(
            {
                "input_rows": int(len(df)),
                "kept_signal_or_intervention_rows": int(len(keep)),
                "retained_rows": int(len(out)),
                "retained_encounters": int(n_enc),
                "provenance_counts": prov,
                "coverage_rates": cov_rate,
                "inconsistent_encounters": int(len(bad_cons)),
            },
            f,
            indent=2,
        )

    print(f"saved: {out_events}")
    print(f"saved: {out_audit}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()

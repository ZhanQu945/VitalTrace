from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from data_v2.preprocess_longitudinal import extract_mimic_meds, extract_mimic_procs, extract_eicu_meds, extract_eicu_procs
from src.preprocess_pipeline.signal_transition_common import TARGET_CONCEPTS, map_intervention, print_examples, print_stage_header


def _build_base_cohort(dataset: str, mimic_root: str, eicu_root: str) -> pd.DataFrame:
    if dataset == "mimic":
        adm = pd.read_csv(os.path.join(mimic_root, "admissions.csv.gz"), usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"]) 
        adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
        adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
        adm = adm.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"]).copy()
        adm["los_h"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 3600.0
        adm["mortality"] = adm["deathtime"].notna().astype(int)
        adm = adm.sort_values(["subject_id", "admittime", "hadm_id"])
        one = adm.groupby("subject_id", as_index=False).head(1)
        return pd.DataFrame({
            "source_dataset": "mimic",
            "patient_id": one["subject_id"].astype(int),
            "encounter_id": one["hadm_id"].astype(int),
            "admit_time": one["admittime"].astype(str),
            "discharge_time": one["dischtime"].astype(str),
            "los_h": one["los_h"].astype(float),
            "mortality": one["mortality"].astype(int),
        })
    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), low_memory=False)
    p = p[["uniquepid", "patientunitstayid", "hospitaladmitoffset", "unitdischargeoffset", "hospitaldischargestatus"]].copy()
    p["hospitaladmitoffset"] = pd.to_numeric(p["hospitaladmitoffset"], errors="coerce")
    p["unitdischargeoffset"] = pd.to_numeric(p["unitdischargeoffset"], errors="coerce")
    p = p.dropna(subset=["uniquepid", "patientunitstayid"]).copy()
    p = p.sort_values(["uniquepid", "patientunitstayid"])
    one = p.groupby("uniquepid", as_index=False).head(1)
    los_h = (one["unitdischargeoffset"].fillna(0.0) / 60.0).clip(lower=0.0)
    mort = one["hospitaldischargestatus"].astype(str).str.lower().str.contains("expired").astype(int)
    # IMPORTANT:
    # Downstream raw event extractors in this repo use patientunitstayid as both
    # patient_id and encounter_id for eICU event rows. To keep joins correct, we
    # align stage-1 keys to that convention and preserve uniquepid separately.
    return pd.DataFrame({
        "source_dataset": "eicu",
        "patient_id": one["patientunitstayid"].astype(int),
        "encounter_id": one["patientunitstayid"].astype(int),
        "source_person_id": one["uniquepid"].astype(str),
        "admit_time": one["hospitaladmitoffset"].astype(str),
        "discharge_time": one["unitdischargeoffset"].astype(str),
        "los_h": los_h.astype(float),
        "mortality": mort.astype(int),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "eicu"])
    ap.add_argument("--mimic-root", default="./data/mimic")
    ap.add_argument("--eicu-root", default="./data/eicu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-relevant-interventions", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header(f"STAGE 1 ({args.dataset}): Cohort + Intervention Gate")

    cohort = _build_base_cohort(args.dataset, args.mimic_root, args.eicu_root)
    print(f"base cohort patients/encounters={len(cohort):,}")

    if args.dataset == "mimic":
        meds = extract_mimic_meds(args.mimic_root)
        procs = extract_mimic_procs(args.mimic_root)
    else:
        meds = extract_eicu_meds(args.eicu_root)
        procs = extract_eicu_procs(args.eicu_root)
    iv = pd.concat([meds, procs], ignore_index=True)

    k = cohort[["source_dataset", "patient_id", "encounter_id"]].copy()
    iv = iv.merge(k, on=["source_dataset", "patient_id", "encounter_id"], how="inner")

    concepts = []
    provs = []
    for _, r in iv.iterrows():
        m = map_intervention(str(r.get("event_name", "")), str(r.get("payload_json", "")))
        concepts.append(m.concepts)
        provs.append(m.provenance)
    iv["intervention_concepts"] = concepts
    iv["intervention_match_provenance"] = provs

    rel_set = set().union(*TARGET_CONCEPTS.values())
    iv["is_relevant"] = iv["intervention_concepts"].map(lambda xs: int(len(set(xs) & rel_set) > 0))

    cnt = iv.groupby(["source_dataset", "patient_id", "encounter_id"], as_index=False)["is_relevant"].sum().rename(columns={"is_relevant": "n_relevant_interventions"})
    cohort2 = cohort.merge(cnt, on=["source_dataset", "patient_id", "encounter_id"], how="left")
    cohort2["n_relevant_interventions"] = cohort2["n_relevant_interventions"].fillna(0).astype(int)
    keep = cohort2[cohort2["n_relevant_interventions"] >= int(args.min_relevant_interventions)].copy()

    iv_keep = iv.merge(keep[["source_dataset", "patient_id", "encounter_id"]], on=["source_dataset", "patient_id", "encounter_id"], how="inner")

    print(f"retained cohort={len(keep):,} / {len(cohort):,}")
    print("provenance:", iv_keep["intervention_match_provenance"].value_counts().to_dict())

    print_examples(keep, "retained encounters", ["source_dataset", "patient_id", "encounter_id", "los_h", "mortality", "n_relevant_interventions"])
    print_examples(iv_keep[iv_keep["is_relevant"] == 1], "relevant intervention rows", ["source_dataset", "patient_id", "encounter_id", "event_name", "intervention_concepts", "intervention_match_provenance"])
    print_examples(cohort2[cohort2["n_relevant_interventions"] == 0], "dropped no-relevant-intervention", ["source_dataset", "patient_id", "encounter_id", "los_h", "mortality"])

    cohort_path = os.path.join(args.out_dir, "stage1_cohort.parquet")
    iv_path = os.path.join(args.out_dir, "stage1_interventions.parquet")
    summary_path = os.path.join(args.out_dir, "stage1_summary.json")

    keep.to_parquet(cohort_path, index=False)
    iv_keep.to_parquet(iv_path, index=False)
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "cohort_input": int(len(cohort)),
            "cohort_retained": int(len(keep)),
            "intervention_rows_retained": int(len(iv_keep)),
            "min_relevant_interventions": int(args.min_relevant_interventions),
            "provenance_counts": iv_keep["intervention_match_provenance"].value_counts().to_dict(),
        }, f, indent=2)

    print(f"saved: {cohort_path}")
    print(f"saved: {iv_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()

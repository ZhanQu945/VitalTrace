from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.preprocess_pipeline.signal_transition_common import print_examples, print_stage_header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-events", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header("STAGE 1: COHORT-FIRST FILTERING")

    df = pd.read_parquet(args.input_events)
    n_rows0 = len(df)
    n_pat0 = df[["source_dataset", "patient_id"]].drop_duplicates().shape[0]
    n_enc0 = df[["source_dataset", "patient_id", "encounter_id"]].drop_duplicates().shape[0]

    df = df.dropna(subset=["source_dataset", "patient_id", "encounter_id", "event_time"]).copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[df["event_time"].notna()].copy()

    # choose one encounter per (source, patient): earliest first event
    enc_first = (
        df.groupby(["source_dataset", "patient_id", "encounter_id"], as_index=False)["event_time"]
        .min()
        .rename(columns={"event_time": "enc_first_time"})
    )
    enc_first = enc_first.sort_values(["source_dataset", "patient_id", "enc_first_time", "encounter_id"])
    keep_enc = enc_first.groupby(["source_dataset", "patient_id"], as_index=False).head(1)

    k = keep_enc[["source_dataset", "patient_id", "encounter_id"]].copy()
    out = df.merge(k, on=["source_dataset", "patient_id", "encounter_id"], how="inner")

    cohort = out[["source_dataset", "patient_id", "encounter_id"]].drop_duplicates().copy()

    n_rows1 = len(out)
    n_pat1 = cohort[["source_dataset", "patient_id"]].drop_duplicates().shape[0]
    n_enc1 = len(cohort)

    print(f"input rows={n_rows0:,} patients={n_pat0:,} encounters={n_enc0:,}")
    print(f"retained rows={n_rows1:,} patients={n_pat1:,} encounters={n_enc1:,}")
    print(f"dropped rows={n_rows0-n_rows1:,} dropped encounters={n_enc0-n_enc1:,}")

    print_examples(cohort, "retained cohort", ["source_dataset", "patient_id", "encounter_id"])

    dropped = enc_first.merge(k, on=["source_dataset", "patient_id", "encounter_id"], how="left", indicator=True)
    dropped = dropped[dropped["_merge"] == "left_only"]
    print_examples(dropped, "dropped duplicate encounters", ["source_dataset", "patient_id", "encounter_id", "enc_first_time"])

    multi = enc_first.groupby(["source_dataset", "patient_id"]).size().reset_index(name="n_enc")
    multi = multi[multi["n_enc"] > 1]
    print_examples(multi, "multi-encounter boundary cases", ["source_dataset", "patient_id", "n_enc"])

    out_events = os.path.join(args.out_dir, "stage1_events.parquet")
    out_cohort = os.path.join(args.out_dir, "stage1_cohort.parquet")
    out_summary = os.path.join(args.out_dir, "stage1_summary.json")

    out.to_parquet(out_events, index=False)
    cohort.to_parquet(out_cohort, index=False)
    with open(out_summary, "w") as f:
        json.dump(
            {
                "input_rows": int(n_rows0),
                "input_patients": int(n_pat0),
                "input_encounters": int(n_enc0),
                "retained_rows": int(n_rows1),
                "retained_patients": int(n_pat1),
                "retained_encounters": int(n_enc1),
            },
            f,
            indent=2,
        )

    print(f"saved: {out_events}")
    print(f"saved: {out_cohort}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()

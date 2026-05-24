from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from src.preprocess_pipeline.signal_transition_common import (
    CORE_FEATURES,
    THRESHOLDS,
    TARGET_CONCEPTS,
    print_examples,
    print_stage_header,
    state_from_value,
)


CORE_ORDER = sorted(CORE_FEATURES)
KEY_COLS = ["source_dataset", "patient_id", "encounter_id"]
CORE_SPLIT_FEATURES = {"map", "rr", "spo2", "creatinine", "lactate"}
PHYSIO_BOUNDS = {
    "map": (20.0, 220.0),
    "rr": (4.0, 80.0),
    "spo2": (40.0, 100.0),
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 20.0),
    "hr": (20.0, 260.0),
    "temp": (30.0, 43.0),
    "wbc": (0.1, 200.0),
    "bicarbonate": (1.0, 60.0),
    "sodium": (90.0, 190.0),
    "potassium": (1.0, 12.0),
    "glucose": (10.0, 1500.0),
}


def sanitize_value(feature: str, value: float):
    lo_hi = PHYSIO_BOUNDS.get(feature)
    if lo_hi is None:
        return value
    lo, hi = lo_hi
    if value < lo or value > hi:
        return None
    return value


def extreme_value(feature: str, vals: List[float]):
    s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(s) == 0:
        return None
    if feature in {"map", "spo2"}:
        return float(s.min())
    if feature in {"rr", "lactate", "creatinine", "wbc", "temp", "hr"}:
        return float(s.max())
    lo, hi = THRESHOLDS.get(feature, (None, None))
    if lo is None or hi is None:
        return float(s.iloc[-1])
    mid = (lo + hi) / 2.0
    idx = (s - mid).abs().idxmax()
    return float(s.loc[idx])


def _to_ts(v):
    t = pd.to_datetime(v, errors="coerce")
    return t if pd.notna(t) else None


def build_steps_for_encounter(enc_sig: pd.DataFrame, carryforward_max_age_h: float) -> List[dict]:
    if enc_sig.empty:
        return []

    enc_sig = enc_sig.sort_values(["event_time", "feature"]).copy()

    source_dataset = enc_sig["source_dataset"].iloc[0]
    patient_id = enc_sig["patient_id"].iloc[0]
    encounter_id = enc_sig["encounter_id"].iloc[0]

    current_state = {f: "missing" for f in CORE_ORDER}
    current_snapshot = {f: None for f in CORE_ORDER}
    last_update_time = {f: None for f in CORE_ORDER}

    buf_vals = defaultdict(list)
    step_start_time = None
    step_anchor_time = None
    step_id = 0
    out = []

    def finalize_step(sid: int):
        nonlocal buf_vals, step_start_time, step_anchor_time, current_snapshot, last_update_time
        if step_anchor_time is None:
            return

        rec = {
            "source_dataset": source_dataset,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "step_id": sid,
            "step_start_time": str(step_start_time),
            "anchor_time": str(step_anchor_time),
        }

        for f in CORE_ORDER:
            if len(buf_vals[f]) > 0:
                v = extreme_value(f, buf_vals[f])
                current_snapshot[f] = v
                last_update_time[f] = step_anchor_time

            v = current_snapshot[f]
            t_last = last_update_time[f]
            stale = False
            age_h = None
            if t_last is not None:
                age_h = (step_anchor_time - t_last).total_seconds() / 3600.0
                if (carryforward_max_age_h > 0) and (age_h > carryforward_max_age_h):
                    stale = True
            if stale:
                rec[f"{f}_value"] = None
                rec[f"{f}_age_h"] = float(age_h)
            else:
                rec[f"{f}_value"] = v
                rec[f"{f}_age_h"] = float(age_h) if age_h is not None and not math.isnan(age_h) else 0.0

        out.append(rec)
        buf_vals = defaultdict(list)
        step_start_time = None
        step_anchor_time = None

    for _, r in enc_sig.iterrows():
        f = str(r["feature"])
        if f not in CORE_SPLIT_FEATURES:
            continue

        t = _to_ts(r["event_time"])
        if t is None:
            continue

        v = pd.to_numeric(pd.Series([r["value_num"]]), errors="coerce").iloc[0]
        if pd.isna(v):
            continue
        v = float(v)
        v = sanitize_value(f, v)
        if v is None:
            continue

        st = state_from_value(f, v)
        prev = current_state[f]

        # New step only on threshold-state crossing between known states.
        should_split = (prev != "missing") and (st != "missing") and (st != prev)

        if should_split and (step_anchor_time is not None):
            finalize_step(step_id)
            step_id += 1

        if step_start_time is None:
            step_start_time = t
        step_anchor_time = t

        buf_vals[f].append(v)
        if st != "missing":
            current_state[f] = st

    if step_anchor_time is not None:
        finalize_step(step_id)

    return out


def label_from_future(fut_iv: pd.DataFrame):
    concepts = set()
    if len(fut_iv):
        for arr in fut_iv["intervention_concepts"].tolist():
            if arr is None:
                continue
            try:
                for c in arr:
                    concepts.add(c)
            except Exception:
                continue
    yv = int(len(concepts & TARGET_CONCEPTS["vasopressor_signal"]) > 0)
    yr = int(len(concepts & TARGET_CONCEPTS["resp_support_signal"]) > 0)
    yk = int(len(concepts & TARGET_CONCEPTS["renal_support_signal"]) > 0)
    return {
        "vasopressor_signal": yv,
        "resp_support_signal": yr,
        "renal_support_signal": yk,
        "any_deterioration": int(yv or yr or yk),
    }


def core_state_and_score(step_row: dict):
    core5 = ["map", "rr", "spo2", "creatinine", "lactate"]
    severe_rules = {
        "map": lambda v: v is not None and v < 55.0,
        "spo2": lambda v: v is not None and v < 88.0,
        "rr": lambda v: v is not None and v >= 30.0,
        "lactate": lambda v: v is not None and v >= 4.0,
        "creatinine": lambda v: v is not None and v >= 2.5,
    }
    abnormal = 0
    severe = 0
    observed = 0
    for f in core5:
        v = step_row.get(f"{f}_value")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        observed += 1
        st = state_from_value(f, v)
        if st in {"low", "high"}:
            abnormal += 1
        if severe_rules[f](v):
            severe += 1
    score = abnormal + 2 * severe
    return observed, abnormal, severe, score


def select_shared_window_indices(
    steps_df: pd.DataFrame,
    iv_df: pd.DataFrame,
    horizons: List[int],
    window_k: int,
) -> pd.DataFrame:
    """Select one contiguous shared window [start, start+k) per encounter for all horizons."""
    if window_k <= 0 or steps_df.empty:
        return pd.DataFrame(columns=KEY_COLS + ["window_start", "window_len", "orig_steps", "window_score"])

    h_for_selection = min(horizons) if len(horizons) else 6
    recs = []
    for k, sg in steps_df.sort_values(KEY_COLS + ["anchor_time"]).groupby(KEY_COLS):
        ivg = iv_df[(iv_df["source_dataset"] == k[0]) & (iv_df["patient_id"] == k[1]) & (iv_df["encounter_id"] == k[2])]
        eg = sg.reset_index(drop=True).copy()
        n = len(eg)
        if n == 0:
            continue
        # Horizon-agnostic enough: use the shortest horizon for local actionability.
        y_any = []
        y_vaso = []
        y_resp = []
        y_renal = []
        for _, r in eg.iterrows():
            t = r["anchor_time"]
            fut = ivg[(ivg["event_time"] > t) & (ivg["event_time"] <= t + pd.Timedelta(hours=h_for_selection))]
            y = label_from_future(fut)
            y_any.append(int(y.get("any_deterioration", 0)))
            y_vaso.append(int(y.get("vasopressor_signal", 0)))
            y_resp.append(int(y.get("resp_support_signal", 0)))
            y_renal.append(int(y.get("renal_support_signal", 0)))

        eg["_sel_step_score"] = (
            3.0 * pd.Series(y_any, dtype=float)
            + 1.5 * pd.Series(y_vaso, dtype=float)
            + 1.2 * pd.Series(y_resp, dtype=float)
            + 1.2 * pd.Series(y_renal, dtype=float)
            + 0.4 * eg["core5_abnormal_count"].fillna(0.0)
            + 0.8 * eg["core5_severe_count"].fillna(0.0)
            + 0.2 * eg["core5_observed_count"].fillna(0.0)
        )
        def window_pattern_score(sub: pd.DataFrame) -> float:
            m = len(sub)
            if m == 0:
                return -1e9
            a_end = max(1, int(round(m * 0.25)))
            d_end = max(a_end + 1, int(round(m * 0.60)))
            i_end = max(d_end + 1, int(round(m * 0.85)))
            early = sub.iloc[:a_end]
            det = sub.iloc[a_end:d_end]
            ivn = sub.iloc[d_end:i_end]
            rec = sub.iloc[i_end:]

            def mean_or0(s):
                return float(s.mean()) if len(s) else 0.0

            early_risk = mean_or0(early["_y_any"])
            det_risk = mean_or0(det["_y_any"])
            iv_risk = mean_or0(ivn["_y_any"])
            rec_risk = mean_or0(rec["_y_any"])

            early_inst = mean_or0(early["instability_score"])
            det_inst = mean_or0(det["instability_score"])
            rec_inst = mean_or0(rec["instability_score"])

            # Encourage first positive away from step 0 (pre-event context).
            pos_idx = sub.index[sub["_y_any"] > 0].tolist()
            first_pos = int(pos_idx[0] - sub.index[0]) if len(pos_idx) else m

            # Core template: low-ish early, rise, intervention-heavy middle, recovery tail.
            score = 0.0
            score += 2.0 * (det_risk - early_risk)
            score += 2.2 * (iv_risk - early_risk)
            score += 1.8 * max(0.0, iv_risk - rec_risk)
            score += 1.2 * (det_inst - early_inst)
            score += 1.2 * max(0.0, det_inst - rec_inst)

            # Label and signal density bonuses.
            score += 0.8 * mean_or0(sub["_y_vaso"])
            score += 2.4 * mean_or0(sub["_y_resp"])
            score += 0.6 * mean_or0(sub["_y_renal"])
            score += 0.2 * mean_or0(sub["core5_observed_count"])
            score += 0.25 * mean_or0(sub["core5_severe_count"])

            # Penalties: always-positive windows and immediate-positive windows.
            score -= 1.8 * mean_or0(sub["_y_any"] > 0.95)
            if first_pos <= 1:
                score -= 1.0
            elif 2 <= first_pos <= max(3, int(m * 0.4)):
                score += 0.8
            return float(score)

        eg["_y_any"] = pd.Series(y_any, dtype=float)
        eg["_y_vaso"] = pd.Series(y_vaso, dtype=float)
        eg["_y_resp"] = pd.Series(y_resp, dtype=float)
        eg["_y_renal"] = pd.Series(y_renal, dtype=float)

        if n <= window_k:
            start = 0
            keep_len = n
            best = eg.copy()
            score = window_pattern_score(best)
            first_pos = int(best.index[best["_y_any"] > 0][0]) if (best["_y_any"] > 0).any() else n
        else:
            best_score = -1e18
            best = None
            best_start = 0
            best_first_pos = n
            # If encounter has respiratory positives anywhere, enforce selecting a window that contains at least one.
            encounter_has_resp = bool(pd.Series(y_resp).sum() > 0)
            fallback = None
            fallback_score = -1e18
            fallback_start = 0
            fallback_first_pos = n
            for st in range(0, n - window_k + 1):
                sub = eg.iloc[st : st + window_k].copy()
                sc = window_pattern_score(sub)
                has_resp = bool(sub["_y_resp"].sum() > 0)
                if sc > fallback_score:
                    fallback_score = sc
                    fallback = sub
                    fallback_start = st
                    posf = sub.index[sub["_y_any"] > 0].tolist()
                    fallback_first_pos = int(posf[0] - sub.index[0]) if len(posf) else window_k
                if encounter_has_resp and (not has_resp):
                    continue
                if sc > best_score:
                    best_score = sc
                    best = sub
                    best_start = st
                    pos = sub.index[sub["_y_any"] > 0].tolist()
                    best_first_pos = int(pos[0] - sub.index[0]) if len(pos) else window_k
            if best is None:
                best = fallback
                best_score = fallback_score
                best_start = fallback_start
                best_first_pos = fallback_first_pos
            start = int(best_start)
            keep_len = int(window_k)
            score = float(best_score)
            first_pos = int(best_first_pos)
        recs.append(
            {
                "source_dataset": k[0],
                "patient_id": int(k[1]),
                "encounter_id": int(k[2]),
                "window_start": int(start),
                "window_len": int(keep_len),
                "orig_steps": int(n),
                "window_score": float(score),
                "first_positive_offset": int(first_pos),
                "window_any_rate": float(best["_y_any"].mean()) if best is not None and len(best) else 0.0,
                "window_vaso_rate": float(best["_y_vaso"].mean()) if best is not None and len(best) else 0.0,
                "window_resp_rate": float(best["_y_resp"].mean()) if best is not None and len(best) else 0.0,
                "window_renal_rate": float(best["_y_renal"].mean()) if best is not None and len(best) else 0.0,
                "window_instability_mean": float(best["instability_score"].mean()) if best is not None and len(best) else 0.0,
                "selection_horizon_hours": int(h_for_selection),
            }
        )
    return pd.DataFrame(recs)


def encounter_rank_from_stage2(
    cohort_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Rank encounters for debug quality using only stage2 outputs."""
    if top_k <= 0:
        return cohort_df.iloc[0:0].copy()

    key_df = cohort_df[KEY_COLS].drop_duplicates()
    ev = events_df.merge(key_df, on=KEY_COLS, how="inner")

    sig = ev[ev["feature"].isin(CORE_FEATURES)].copy()
    iv = ev[ev["feature"].isna()].copy()
    if "intervention_concepts" in iv.columns:
        iv = iv[iv["intervention_concepts"].map(lambda x: hasattr(x, "__len__") and len(x) > 0)].copy()

    # Signal-level quality.
    sig_stats = (
        sig.groupby(KEY_COLS)
        .agg(
            n_signal_rows=("feature", "size"),
            n_unique_features=("feature", "nunique"),
            n_core5_unique=("feature", lambda s: int(len(set(s) & {"map", "rr", "spo2", "creatinine", "lactate"}))),
            n_abnormal=("state", lambda s: int(pd.Series(s).isin(["low", "high"]).sum())),
        )
        .reset_index()
    )

    # Temporal coverage and gaps.
    t_agg = sig.groupby(KEY_COLS).agg(tmin=("event_time", "min"), tmax=("event_time", "max")).reset_index()
    t_agg["span_hours"] = (
        (pd.to_datetime(t_agg["tmax"], errors="coerce") - pd.to_datetime(t_agg["tmin"], errors="coerce"))
        .dt.total_seconds()
        .div(3600.0)
        .fillna(0.0)
    )
    sig = sig.sort_values(KEY_COLS + ["event_time"])
    sig["prev_time"] = sig.groupby(KEY_COLS)["event_time"].shift(1)
    sig["gap_h"] = (
        (pd.to_datetime(sig["event_time"], errors="coerce") - pd.to_datetime(sig["prev_time"], errors="coerce"))
        .dt.total_seconds()
        .div(3600.0)
    )
    gap_stats = sig.groupby(KEY_COLS).agg(max_gap_h=("gap_h", "max"), median_gap_h=("gap_h", "median")).reset_index()

    # Intervention density/diversity.
    def _concepts(arr) -> set:
        if arr is None:
            return set()
        if isinstance(arr, (list, tuple, set)):
            return {str(x) for x in arr if x is not None and str(x) != "nan"}
        if hasattr(arr, "tolist"):
            try:
                vals = arr.tolist()
                if isinstance(vals, list):
                    return {str(x) for x in vals if x is not None and str(x) != "nan"}
            except Exception:
                pass
        # Fallback: scalar-like entries become one concept token.
        s = str(arr)
        if s and s != "nan":
            return {s}
        return set()

    iv_stats = (
        iv.groupby(KEY_COLS)
        .agg(
            n_iv_rows=("intervention_concepts", "size"),
            n_iv_concepts=("intervention_concepts", lambda s: int(len({c for arr in s for c in _concepts(arr)}))),
            n_iv_resp_rows=(
                "intervention_concepts",
                lambda s: int(
                    sum(
                        1
                        for arr in s
                        if len(_concepts(arr) & TARGET_CONCEPTS["resp_support_signal"]) > 0
                    )
                ),
            ),
            n_iv_renal_rows=(
                "intervention_concepts",
                lambda s: int(
                    sum(
                        1
                        for arr in s
                        if len(_concepts(arr) & TARGET_CONCEPTS["renal_support_signal"]) > 0
                    )
                ),
            ),
            n_iv_vaso_rows=(
                "intervention_concepts",
                lambda s: int(
                    sum(
                        1
                        for arr in s
                        if len(_concepts(arr) & TARGET_CONCEPTS["vasopressor_signal"]) > 0
                    )
                ),
            ),
        )
        .reset_index()
    )

    rank_df = key_df.merge(sig_stats, on=KEY_COLS, how="left").merge(t_agg[KEY_COLS + ["span_hours"]], on=KEY_COLS, how="left")
    rank_df = rank_df.merge(gap_stats, on=KEY_COLS, how="left").merge(iv_stats, on=KEY_COLS, how="left")
    rank_df = rank_df.fillna(
        {
            "n_signal_rows": 0,
            "n_unique_features": 0,
            "n_core5_unique": 0,
            "n_abnormal": 0,
            "span_hours": 0.0,
            "max_gap_h": 9999.0,
            "median_gap_h": 9999.0,
            "n_iv_rows": 0,
            "n_iv_concepts": 0,
            "n_iv_resp_rows": 0,
            "n_iv_renal_rows": 0,
            "n_iv_vaso_rows": 0,
        }
    )

    # Weighted score favors complete core signals + abnormalities + interventions + continuity.
    rank_df["rank_score"] = (
        3.0 * rank_df["n_core5_unique"]
        + 0.002 * rank_df["n_signal_rows"]
        + 0.05 * rank_df["n_abnormal"]
        + 0.5 * rank_df["n_iv_rows"]
        + 2.0 * rank_df["n_iv_concepts"]
        + 0.8 * rank_df["n_iv_resp_rows"]
        + 0.5 * rank_df["n_iv_renal_rows"]
        + 0.3 * rank_df["n_iv_vaso_rows"]
        + 0.03 * rank_df["span_hours"].clip(lower=0.0, upper=48.0)
        - 0.3 * rank_df["max_gap_h"].clip(lower=0.0, upper=48.0)
        - 0.15 * rank_df["median_gap_h"].clip(lower=0.0, upper=24.0)
    )

    rank_df = rank_df.sort_values(
        ["rank_score", "n_core5_unique", "n_iv_rows", "source_dataset", "patient_id", "encounter_id"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return rank_df.head(top_k).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic", "eicu"])
    ap.add_argument("--stage2-cohort", required=True)
    ap.add_argument("--stage2-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizons", default="6,12,24,48")
    ap.add_argument("--progress-every-patients", type=int, default=100)
    ap.add_argument("--los-mode", default="all", choices=["all", "lt48"])
    ap.add_argument("--carryforward-max-age-hours", type=float, default=24.0)
    ap.add_argument("--qc-min-steps-per-encounter", type=float, default=8.0)
    ap.add_argument("--qc-max-steps-per-encounter", type=float, default=220.0)
    ap.add_argument("--qc-min-nonmissing-core-features", type=float, default=3.0)
    ap.add_argument("--qc-min-any-deterioration-rate", type=float, default=0.05)
    ap.add_argument("--qc-max-any-deterioration-rate", type=float, default=0.98)
    ap.add_argument("--qc-strict", action="store_true")
    ap.add_argument("--reuse-checkpoints", action="store_true")
    ap.add_argument("--simplified-top-k", type=int, default=0)
    ap.add_argument("--meaningful-steps-per-encounter", type=int, default=0)
    ap.add_argument("--simplified-min-resp-encounters", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_stage_header(f"STAGE 3 ({args.dataset}): Transition + Label + QC")

    horizons = [int(x) for x in re.split(r"[,\s|;]+", str(args.horizons).strip()) if str(x).strip()]
    print(f"horizons={horizons}")

    cohort_ckpt = os.path.join(args.out_dir, "stage3_policy_cohort.parquet")
    ev_ckpt = os.path.join(args.out_dir, "stage3_policy_events.parquet")
    steps_ckpt_parquet = os.path.join(args.out_dir, "stage3_steps.parquet")
    step_path = os.path.join(args.out_dir, "stage3_steps.jsonl")

    selected_ckpt = os.path.join(args.out_dir, "stage3_selected_encounters.parquet")
    ranking_ckpt = os.path.join(args.out_dir, "stage3_selection_ranking.parquet")

    if args.reuse_checkpoints and os.path.exists(cohort_ckpt) and os.path.exists(ev_ckpt):
        print("Reusing policy-filtered cohort/events checkpoints...")
        c = pd.read_parquet(cohort_ckpt)
        ev = pd.read_parquet(ev_ckpt)
        ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
        ev = ev[ev["event_time"].notna()].copy()
    else:
        cohort = pd.read_parquet(args.stage2_cohort)
        ev = pd.read_parquet(args.stage2_events)
        print(f"[stats] stage2 cohort rows={len(cohort):,}")
        print(f"[stats] stage2 event rows={len(ev):,}")
        ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
        ev = ev[ev["event_time"].notna()].copy()
        print(f"[stats] stage2 event rows with valid time={len(ev):,}")

        # Policy filtering moves from old stage4 into stage3.
        c = cohort.copy()
        if args.los_mode == "lt48":
            c = c[c["los_h"].astype(float) < 48].copy()
        print(f"[stats] cohort after los_mode={args.los_mode}: {len(c):,}")
        key_df = c[KEY_COLS].drop_duplicates()

        ev = ev.merge(key_df, on=KEY_COLS, how="inner")
        print(f"[stats] events after los_mode encounter join: {len(ev):,}")

        # Simplified debug mode: select top-k encounters directly from stage2 signals/interventions.
        if args.simplified_top_k > 0:
            ranked_all = encounter_rank_from_stage2(c, ev, top_k=max(int(args.simplified_top_k) * 50, 200))
            need_k = int(args.simplified_top_k)
            need_resp = max(0, int(args.simplified_min_resp_encounters))
            selected_idx = []
            resp_pool = ranked_all[ranked_all["n_iv_resp_rows"] > 0]
            if need_resp > 0 and len(resp_pool) > 0:
                selected_idx.extend(resp_pool.head(need_resp).index.tolist())
            for idx, _ in ranked_all.iterrows():
                if idx in selected_idx:
                    continue
                selected_idx.append(idx)
                if len(selected_idx) >= need_k:
                    break
            selected_idx = selected_idx[:need_k]
            ranked = ranked_all.loc[selected_idx].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)
            sel_keys = ranked[KEY_COLS].drop_duplicates()
            c = c.merge(sel_keys, on=KEY_COLS, how="inner")
            ev = ev.merge(sel_keys, on=KEY_COLS, how="inner")
            ranked.to_parquet(ranking_ckpt, index=False)
            sel_keys.to_parquet(selected_ckpt, index=False)
            print(f"simplified_top_k enabled: selected {len(sel_keys):,} encounters")
            print(
                f"[stats] selection diversity: resp_encounters={(ranked['n_iv_resp_rows'] > 0).sum():,}, "
                f"renal_encounters={(ranked['n_iv_renal_rows'] > 0).sum():,}, "
                f"vaso_encounters={(ranked['n_iv_vaso_rows'] > 0).sum():,}"
            )
            print(
                f"[stats] selected LOS summary hours: "
                f"min={c['los_h'].min():.2f}, p50={c['los_h'].median():.2f}, max={c['los_h'].max():.2f}"
            )
            if len(ranked):
                show_cols = KEY_COLS + ["rank_score", "n_core5_unique", "n_iv_rows", "n_iv_concepts", "max_gap_h", "median_gap_h"]
                print("[stats] top-ranked encounters:")
                print(ranked[show_cols].head(10).to_string(index=False))
            print(f"saved: {ranking_ckpt}")
            print(f"saved: {selected_ckpt}")

        c.to_parquet(cohort_ckpt, index=False)
        ev.to_parquet(ev_ckpt, index=False)
        print(f"saved checkpoint: {cohort_ckpt}")
        print(f"saved checkpoint: {ev_ckpt}")

    if "c" not in locals():
        # when reusing checkpoints, c was loaded from file above
        c = pd.read_parquet(cohort_ckpt)
    sig = ev[ev["feature"].isin(CORE_FEATURES)].copy() if "feature" in ev.columns else pd.DataFrame(columns=ev.columns)
    iv = ev[ev["feature"].isna()].copy() if "feature" in ev.columns else pd.DataFrame(columns=ev.columns)
    print(f"[stats] signal rows pre-stage3={len(sig):,}")
    print(f"[stats] intervention rows pre-stage3={len(iv):,}")

    # Keep only intervention rows with mapped target concepts.
    if "intervention_concepts" in iv.columns:
        iv = iv[iv["intervention_concepts"].map(lambda x: hasattr(x, "__len__") and len(x) > 0)].copy()
    print(f"[stats] intervention rows with mapped concepts={len(iv):,}")

    # Build or reuse transition steps checkpoint.
    if args.reuse_checkpoints and os.path.exists(steps_ckpt_parquet):
        print("Reusing steps checkpoint...")
        steps_df = pd.read_parquet(steps_ckpt_parquet)
        steps_df["anchor_time"] = pd.to_datetime(steps_df["anchor_time"], errors="coerce")
        steps_df = steps_df[steps_df["anchor_time"].notna()].copy()
    else:
        steps = []
        groups = list(sig.sort_values(KEY_COLS + ["event_time"]).groupby(KEY_COLS))
        print(f"encounters for step construction={len(groups):,}")
        for gi, (_, g) in enumerate(groups, start=1):
            if args.progress_every_patients > 0 and gi % args.progress_every_patients == 0:
                print(f"[progress][stage3-step] trajectories={gi}/{len(groups)} steps={len(steps)}", flush=True)
            steps.extend(build_steps_for_encounter(g, carryforward_max_age_h=float(args.carryforward_max_age_hours)))

        steps_df = pd.DataFrame(steps)
        if len(steps_df) == 0:
            print("No steps produced; exiting.", file=sys.stderr)
            sys.exit(2)

        steps_df["anchor_time"] = pd.to_datetime(steps_df["anchor_time"], errors="coerce")
        steps_df = steps_df[steps_df["anchor_time"].notna()].copy()
        # Step-level clinical annotations (lightweight for LLM task difficulty control)
        ann = steps_df.apply(lambda r: core_state_and_score(r.to_dict()), axis=1, result_type="expand")
        ann.columns = ["core5_observed_count", "core5_abnormal_count", "core5_severe_count", "instability_score"]
        steps_df = pd.concat([steps_df.reset_index(drop=True), ann.reset_index(drop=True)], axis=1)
        steps_df.to_parquet(steps_ckpt_parquet, index=False)
        print(f"saved checkpoint: {steps_ckpt_parquet}")
        step_counts_dbg = steps_df.groupby(KEY_COLS).size().astype(float)
        print(
            f"[stats] steps/encounter after build: "
            f"min={step_counts_dbg.min():.1f}, p50={step_counts_dbg.median():.1f}, "
            f"p90={step_counts_dbg.quantile(0.9):.1f}, max={step_counts_dbg.max():.1f}"
        )

    with open(step_path, "w") as f:
        for _, r in steps_df.iterrows():
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
    print(f"saved: {step_path}")

    # Select one shared K-step window per encounter for all horizons.
    shared_windows = None
    if args.meaningful_steps_per_encounter > 0 and len(steps_df):
        shared_windows = select_shared_window_indices(
            steps_df=steps_df,
            iv_df=iv,
            horizons=horizons,
            window_k=int(args.meaningful_steps_per_encounter),
        )
        if len(shared_windows):
            print(
                f"[stats] shared-window selection K={int(args.meaningful_steps_per_encounter)} "
                f"encounters={len(shared_windows):,}"
            )
            print(shared_windows.sort_values("window_score", ascending=False).to_string(index=False))
            shared_windows.to_parquet(os.path.join(args.out_dir, "stage3_shared_windows.parquet"), index=False)

    # Label generation per horizon.
    prev_by_h = {}
    labeled_row_counts = {}
    saved_label_paths = []
    for h in horizons:
        rows = []
        step_groups = list(steps_df.groupby(KEY_COLS))
        for gi, (k, sg) in enumerate(step_groups, start=1):
            if args.progress_every_patients > 0 and gi % args.progress_every_patients == 0:
                print(f"[progress][stage3-label-h{h}] trajectories={gi}/{len(step_groups)} rows={len(rows)}", flush=True)
            ivg = iv[(iv["source_dataset"] == k[0]) & (iv["patient_id"] == k[1]) & (iv["encounter_id"] == k[2])]
            prev_score = None
            for _, r in sg.sort_values("anchor_time").iterrows():
                t = r["anchor_time"]
                fut = ivg[(ivg["event_time"] > t) & (ivg["event_time"] <= t + pd.Timedelta(hours=h))]
                rec = dict(r)
                rec["horizon_hours"] = h
                raw_t = label_from_future(fut)
                # Optional clinically conservative gate: require severe or multi-signal deterioration.
                severe_now = int(rec.get("core5_severe_count", 0) >= 1)
                multisig_now = int(rec.get("core5_abnormal_count", 0) >= 2)
                gate = int(severe_now or multisig_now)
                gated_t = {
                    "vasopressor_signal": int(raw_t["vasopressor_signal"] and gate),
                    "resp_support_signal": int(raw_t["resp_support_signal"] and gate),
                    "renal_support_signal": int(raw_t["renal_support_signal"] and gate),
                }
                gated_t["any_deterioration"] = int(
                    gated_t["vasopressor_signal"] or gated_t["resp_support_signal"] or gated_t["renal_support_signal"]
                )
                rec["targets_raw"] = raw_t
                rec["targets"] = raw_t
                rec["targets_clinical_gate"] = gated_t
                # Coarse phase tag for interpretation (not used as supervision target).
                score = float(rec.get("instability_score", 0.0))
                raw_any = int(raw_t.get("any_deterioration", 0))
                if raw_any == 1:
                    phase = "intervention_window"
                elif prev_score is None:
                    phase = "stable" if score <= 1 else "deterioration"
                elif score > prev_score:
                    phase = "deterioration"
                elif score < prev_score:
                    phase = "recovery"
                else:
                    phase = "stable" if score <= 1 else "deterioration"
                rec["phase_tag"] = phase
                prev_score = score
                rows.append(rec)

        out_df = pd.DataFrame(rows)
        # Apply one shared K-step window per encounter across all horizons.
        if args.meaningful_steps_per_encounter > 0 and len(out_df) and shared_windows is not None and len(shared_windows):
            kept = []
            for _, w in shared_windows.iterrows():
                m = (
                    (out_df["source_dataset"] == w["source_dataset"])
                    & (out_df["patient_id"] == w["patient_id"])
                    & (out_df["encounter_id"] == w["encounter_id"])
                )
                eg = out_df[m].sort_values("anchor_time").reset_index(drop=True)
                if eg.empty:
                    continue
                st = int(w["window_start"])
                ln = int(w["window_len"])
                chunk = eg.iloc[st : st + ln].copy()
                chunk["_selected_window_start"] = st
                chunk["_selected_window_len"] = len(chunk)
                kept.append(chunk)
            if kept:
                before_n = len(out_df)
                out_df = pd.concat(kept, ignore_index=True)
                print(
                    f"[stats] horizon h{h} shared window applied: "
                    f"orig_rows={before_n:,} kept_rows={len(out_df):,} encounters={len(shared_windows):,}"
                )

        out_path = os.path.join(args.out_dir, f"stage3_labeled_h{h}.jsonl")
        with open(out_path, "w") as f:
            for _, r in out_df.iterrows():
                f.write(json.dumps(r.to_dict(), default=str) + "\n")
        saved_label_paths.append(out_path)
        labeled_row_counts[str(h)] = int(len(out_df))

        prev_by_h[str(h)] = {
            lab: float(out_df["targets"].map(lambda x: x.get(lab, 0)).mean()) if len(out_df) else 0.0
            for lab in ["vasopressor_signal", "resp_support_signal", "renal_support_signal", "any_deterioration"]
        }
        prev_by_h[f"{h}_clinical_gate"] = {
            lab: float(out_df["targets_clinical_gate"].map(lambda x: x.get(lab, 0)).mean()) if len(out_df) else 0.0
            for lab in ["vasopressor_signal", "resp_support_signal", "renal_support_signal", "any_deterioration"]
        }
        print(f"[stats] horizon h{h} rows={len(out_df):,} prevalence={prev_by_h[str(h)]}")

    # QC metrics and gates (moved from stage4).
    step_counts = steps_df.groupby(KEY_COLS).size()
    steps_per_enc_median = float(step_counts.median()) if len(step_counts) else 0.0
    steps_per_enc_p90 = float(step_counts.quantile(0.90)) if len(step_counts) else 0.0

    core5_cols = [f"{f}_value" for f in ["map", "rr", "spo2", "creatinine", "lactate"]]
    n_nonmissing_core = steps_df[core5_cols].notna().sum(axis=1) if set(core5_cols).issubset(set(steps_df.columns)) else pd.Series([], dtype=float)
    nonmissing_core_median = float(n_nonmissing_core.median()) if len(n_nonmissing_core) else 0.0

    main_h = "12" if "12" in prev_by_h else (str(horizons[0]) if horizons else "6")
    any_det_rate = float(prev_by_h.get(main_h, {}).get("any_deterioration", 0.0))

    qc_metrics = {
        "steps_per_encounter_median": steps_per_enc_median,
        "steps_per_encounter_p90": steps_per_enc_p90,
        "steps_per_encounter_mean": float(step_counts.mean()) if len(step_counts) else 0.0,
        "steps_per_encounter_max": float(step_counts.max()) if len(step_counts) else 0.0,
        "nonmissing_core5_features_median_per_step": nonmissing_core_median,
        "nonmissing_core5_features_p10_per_step": float(n_nonmissing_core.quantile(0.10)) if len(n_nonmissing_core) else 0.0,
        "nonmissing_core5_features_p90_per_step": float(n_nonmissing_core.quantile(0.90)) if len(n_nonmissing_core) else 0.0,
        "core5_complete_step_share": float((n_nonmissing_core == 5).mean()) if len(n_nonmissing_core) else 0.0,
        "any_deterioration_rate_main_horizon": any_det_rate,
        "main_horizon_for_qc": int(main_h),
    }
    qc_gates = {
        "min_steps_per_encounter": float(args.qc_min_steps_per_encounter),
        "max_steps_per_encounter": float(args.qc_max_steps_per_encounter),
        "min_nonmissing_core5_features": float(args.qc_min_nonmissing_core_features),
        "min_any_deterioration_rate": float(args.qc_min_any_deterioration_rate),
        "max_any_deterioration_rate": float(args.qc_max_any_deterioration_rate),
    }

    qc_fail_reasons = []
    if steps_per_enc_median < qc_gates["min_steps_per_encounter"]:
        qc_fail_reasons.append(f"steps_per_encounter_median {steps_per_enc_median:.2f} < {qc_gates['min_steps_per_encounter']:.2f}")
    if steps_per_enc_median > qc_gates["max_steps_per_encounter"]:
        qc_fail_reasons.append(f"steps_per_encounter_median {steps_per_enc_median:.2f} > {qc_gates['max_steps_per_encounter']:.2f}")
    if nonmissing_core_median < qc_gates["min_nonmissing_core5_features"]:
        qc_fail_reasons.append(
            f"nonmissing_core5_features_median_per_step {nonmissing_core_median:.2f} < {qc_gates['min_nonmissing_core5_features']:.2f}"
        )
    if any_det_rate < qc_gates["min_any_deterioration_rate"]:
        qc_fail_reasons.append(f"any_deterioration_rate_main_horizon {any_det_rate:.4f} < {qc_gates['min_any_deterioration_rate']:.4f}")
    if any_det_rate > qc_gates["max_any_deterioration_rate"]:
        qc_fail_reasons.append(f"any_deterioration_rate_main_horizon {any_det_rate:.4f} > {qc_gates['max_any_deterioration_rate']:.4f}")

    qc_pass = len(qc_fail_reasons) == 0

    s = c["los_h"].astype(float)
    summary = {
        "dataset": args.dataset,
        "los_mode": args.los_mode,
        "n_encounters": int(len(c)),
        "n_steps": int(len(steps_df)),
        "los_median_h": float(s.median()) if len(s) else 0.0,
        "los_lt_48h_share": float((s < 48).mean()) if len(s) else 0.0,
        "mortality_share": float(c["mortality"].mean()) if "mortality" in c.columns and len(c) else 0.0,
        "label_prevalence_by_horizon": prev_by_h,
        "labeled_rows_by_horizon": labeled_row_counts,
        "carryforward_max_age_hours": float(args.carryforward_max_age_hours),
        "simplified_top_k": int(args.simplified_top_k),
        "qc_metrics": qc_metrics,
        "qc_gates": qc_gates,
        "qc_pass": qc_pass,
        "qc_fail_reasons": qc_fail_reasons,
    }

    summary_path = os.path.join(args.out_dir, "stage3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    print(f"QC status: {'PASS' if qc_pass else 'FAIL'}")
    if qc_fail_reasons:
        for r in qc_fail_reasons:
            print(f"  - {r}")

    print_examples(steps_df, "transition steps", ["source_dataset", "patient_id", "encounter_id", "step_id", "anchor_time", "map_value", "rr_value", "spo2_value", "lactate_value", "creatinine_value"])
    print_examples(c.sort_values("los_h", ascending=False), "LOS examples", ["source_dataset", "patient_id", "encounter_id", "los_h", "mortality"])
    print_examples(c.sort_values("mortality", ascending=False), "mortality examples", ["source_dataset", "patient_id", "encounter_id", "los_h", "mortality"])

    if not os.path.exists(step_path) or os.path.getsize(step_path) == 0:
        print(f"Missing/empty required file: {step_path}", file=sys.stderr)
        sys.exit(2)
    for p in saved_label_paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"Missing/empty required horizon file: {p}", file=sys.stderr)
            sys.exit(2)
    print(f"saved: {step_path}")
    for p in saved_label_paths:
        print(f"saved: {p}")
    print(f"saved: {summary_path}")

    if args.qc_strict and (not qc_pass):
        print("Stage 3 strict QC enabled and gates failed; exiting with status 2.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

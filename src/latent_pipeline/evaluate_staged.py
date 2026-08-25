from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from src.latent_pipeline.common import iter_jsonl, write_json
from src.latent_pipeline.prediction_targets import add_composite_label, add_composite_probability
from src.latent_pipeline.protocol_utils import load_protocol, feature_map_from_facts, rule_score

LABELS = ["vasopressor_signal", "resp_support_signal", "renal_support_signal", "any_deterioration"]
STATE_KEYS = [
    "hemodynamic_state",
    "respiratory_state",
    "renal_state",
    "metabolic_state",
    "systemic_inflammation_state",
]
RISK_STATE_TO_TARGET = {
    "hemodynamic_state": "vasopressor_signal",
    "respiratory_state": "resp_support_signal",
    "renal_state": "renal_support_signal",
    "metabolic_state": "any_deterioration",
    "systemic_inflammation_state": "any_deterioration",
}
COMMON_STAGE_FIELDS = {
    "record_id",
    "example_id",
    "source_dataset",
    "patient_id",
    "encounter_id",
    "anchor_time",
    "step_id",
    "packet",
    "selected_rule_ids",
    "active_rules",
    "ground_truth_targets",
    "counterfactual_candidates",
    "inference_context_schema",
    "target_isolation_verified",
    "protocol_observations",
    "router_output",
    "reasoner_prediction",
    "audit",
    "individual_protocol_state_prev",
    "individual_protocol_state_next",
    "individual_protocol_state_delta",
    "state_version",
    "state_update_source",
    "stage1_prediction",
    "stage1_ground_truth",
    "stage2_prediction",
    "stage2_ground_truth",
    "stage3_prediction",
    "stage3_ground_truth",
    "stage4_prediction",
    "stage4_ground_truth",
    "llm_status",
    "llm_issues",
    "auditor_inputs",
    "temporal_loop",
}


def _join_cols(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    if "record_id" in df_left.columns and "record_id" in df_right.columns:
        return ["record_id"]
    return ["example_id"]


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray):
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray):
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _bin_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5):
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (y_score >= thr).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = None
    return out


def _safe_specificity(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5):
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (y_score >= thr).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    d = tn + fp
    if d == 0:
        return None
    return float(tn / d)


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    if len(y_true) == 0:
        return None
    y_true = y_true.astype(float)
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_score = np.clip(y_score, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_score >= lo) & (y_score <= hi)
        else:
            m = (y_score >= lo) & (y_score < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_score[m]))
        ece += float(np.sum(m) / n) * abs(acc - conf)
    return float(ece)


def _safe_metrics_block(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Optional[float]]:
    out = {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "f1": None,
        "precision": None,
        "recall": None,
        "specificity": _safe_specificity(y_true, y_score),
        "brier": None,
        "ece": _ece(y_true, y_score),
    }
    if len(y_true) > 0:
        bm = _bin_metrics(y_true, y_score)
        out["f1"] = bm["f1"]
        out["precision"] = bm["precision"]
        out["recall"] = bm["recall"]
        out["brier"] = bm["brier"]
    return out


def _resample_patient_rows(
    pred_df: pd.DataFrame,
    patient_key: pd.Series,
    sampled_patient_keys: List[str],
) -> pd.DataFrame:
    """Concatenate sampled patient clusters while preserving multiplicity."""
    groups = {
        key: pred_df.loc[patient_key == key]
        for key in patient_key.unique().tolist()
    }
    return pd.concat(
        [groups[key] for key in sampled_patient_keys],
        ignore_index=True,
    )


def _bootstrap_patient_ci(pred_df: pd.DataFrame, n_boot: int = 1000, seed: int = 13) -> Dict[str, Optional[List[float]]]:
    if pred_df.empty:
        return {"macro_auroc_95ci": None, "macro_auprc_95ci": None, "macro_f1_95ci": None}

    patient_key = pred_df["source_dataset"].astype(str) + "||" + pred_df["patient_id"].astype(str)
    uniq = patient_key.unique().tolist()
    if len(uniq) == 0:
        return {"macro_auroc_95ci": None, "macro_auprc_95ci": None, "macro_f1_95ci": None}

    rng = np.random.default_rng(seed)
    m_aurocs, m_auprcs, m_f1s = [], [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        bdf = _resample_patient_rows(pred_df, patient_key, sampled.tolist())
        if bdf.empty:
            continue
        au_l, ap_l, f1_l = [], [], []
        for lab in LABELS:
            yt = bdf[f"gt_{lab}"].to_numpy(dtype=int)
            yp = bdf[f"pred_{lab}"].to_numpy(dtype=float)
            au = _safe_auroc(yt, yp)
            ap = _safe_auprc(yt, yp)
            f1v = _bin_metrics(yt, yp)["f1"] if len(yt) else None
            if au is not None:
                au_l.append(au)
            if ap is not None:
                ap_l.append(ap)
            if f1v is not None:
                f1_l.append(f1v)
        if au_l:
            m_aurocs.append(float(np.mean(au_l)))
        if ap_l:
            m_auprcs.append(float(np.mean(ap_l)))
        if f1_l:
            m_f1s.append(float(np.mean(f1_l)))

    def _ci(v: List[float]) -> Optional[List[float]]:
        if not v:
            return None
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "macro_auroc_95ci": _ci(m_aurocs),
        "macro_auprc_95ci": _ci(m_auprcs),
        "macro_f1_95ci": _ci(m_f1s),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_unit": "patient",
        "bootstrap_sampling": "clusters_with_replacement_preserving_multiplicity",
    }


def _parse_time_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _hours_between(t1: pd.Timestamp, t2: pd.Timestamp) -> Optional[float]:
    if pd.isna(t1) or pd.isna(t2):
        return None
    h = float((t2 - t1).total_seconds() / 3600.0)
    # Guard against corrupted timelines (e.g., cross-year jumps in deidentified offsets).
    if abs(h) > (24.0 * 30.0):
        return None
    return h


def _early_warning_metrics(pred_df: pd.DataFrame, label: str, horizons: List[int]) -> Dict[str, Optional[float]]:
    if pred_df.empty:
        return {f"event_recall_at_{h}h": None for h in horizons} | {"median_lead_time_h": None, "mean_lead_time_h": None, "false_alarms_per_24h": None}

    d = pred_df.copy()
    d["anchor_time_parsed"] = _parse_time_col(d["anchor_time"])
    d["pred_bin"] = (d[f"pred_{label}"] >= 0.5).astype(int)
    d["gt_bin"] = d[f"gt_{label}"].astype(int)
    d = d.sort_values(["source_dataset", "patient_id", "encounter_id", "anchor_time_parsed", "step_id"], na_position="last")

    recalls = {h: [] for h in horizons}
    lead_times = []
    false_alarm_hours = []
    false_alarm_counts = []

    for _, g in d.groupby(["source_dataset", "patient_id", "encounter_id"], dropna=False):
        g = g.reset_index(drop=True)
        t = g["anchor_time_parsed"].tolist()
        gt = g["gt_bin"].tolist()
        pr = g["pred_bin"].tolist()
        pos_idx = [i for i, v in enumerate(gt) if v == 1]
        if pos_idx:
            hit_flags = {h: False for h in horizons}
            for j in pos_idx:
                best_lead = None
                for i in range(j):
                    if pr[i] != 1:
                        continue
                    hdiff = _hours_between(t[i], t[j])
                    if hdiff is None or hdiff < 0:
                        continue
                    if best_lead is None or hdiff < best_lead:
                        best_lead = hdiff
                    for h in horizons:
                        if hdiff <= float(h):
                            hit_flags[h] = True
                if best_lead is not None:
                    lead_times.append(best_lead)
            for h in horizons:
                recalls[h].append(1.0 if hit_flags[h] else 0.0)
        # false alarms per 24h: predicted positive at step i with no event in next 24h
        fa = 0
        monitored_h = 0.0
        for i in range(len(g) - 1):
            hstep = _hours_between(t[i], t[i + 1])
            if hstep is not None and hstep > 0:
                monitored_h += hstep
            if pr[i] != 1:
                continue
            has_future_event = False
            for j in range(i + 1, len(g)):
                if gt[j] != 1:
                    continue
                hdiff = _hours_between(t[i], t[j])
                if hdiff is not None and 0 <= hdiff <= 24.0:
                    has_future_event = True
                    break
            if not has_future_event:
                fa += 1
        if monitored_h > 0:
            false_alarm_counts.append(float(fa))
            false_alarm_hours.append(float(monitored_h))

    out = {f"event_recall_at_{h}h": (float(np.mean(recalls[h])) if recalls[h] else None) for h in horizons}
    out["median_lead_time_h"] = float(np.median(lead_times)) if lead_times else None
    out["mean_lead_time_h"] = float(np.mean(lead_times)) if lead_times else None
    if false_alarm_hours and sum(false_alarm_hours) > 0:
        out["false_alarms_per_24h"] = float(sum(false_alarm_counts) / sum(false_alarm_hours) * 24.0)
    else:
        out["false_alarms_per_24h"] = None
    return out


def _event_level_and_kstep_metrics(pred_df: pd.DataFrame, label: str, k_steps: List[int]) -> Dict[str, Optional[float]]:
    if pred_df.empty:
        out = {
            "n_events": 0,
            "event_detected_rate": None,
            "median_lead_steps": None,
            "mean_lead_steps": None,
            "median_lead_hours": None,
            "mean_lead_hours": None,
            "alerts_total": 0,
            "alerts_with_event_within_ks_steps_rate": {},
            "events_with_alert_in_prev_ks_steps_rate": {},
            "alerts_per_event": None,
        }
        for k in k_steps:
            out["alerts_with_event_within_ks_steps_rate"][f"k{k}"] = None
            out["events_with_alert_in_prev_ks_steps_rate"][f"k{k}"] = None
        return out

    d = pred_df.copy()
    d["anchor_time_parsed"] = _parse_time_col(d["anchor_time"])
    d["pred_bin"] = (d[f"pred_{label}"] >= 0.5).astype(int)
    d["gt_bin"] = d[f"gt_{label}"].astype(int)
    d = d.sort_values(["source_dataset", "patient_id", "encounter_id", "anchor_time_parsed", "step_id"], na_position="last")

    n_events = 0
    detected_events = 0
    lead_steps_all: List[int] = []
    lead_hours_all: List[float] = []
    alerts_total = 0
    alert_hits = {k: 0 for k in k_steps}
    event_hits_prevk = {k: 0 for k in k_steps}

    for _, g in d.groupby(["source_dataset", "patient_id", "encounter_id"], dropna=False):
        g = g.reset_index(drop=True)
        t = g["anchor_time_parsed"].tolist()
        gt = g["gt_bin"].tolist()
        pr = g["pred_bin"].tolist()
        n = len(g)
        alerts_total += int(np.sum(pr))

        # Event onsets (new positive segments)
        event_onsets = []
        for i in range(n):
            if gt[i] == 1 and (i == 0 or gt[i - 1] == 0):
                event_onsets.append(i)

        n_events += len(event_onsets)
        for ev in event_onsets:
            best_step = None
            best_h = None
            for i in range(ev):
                if pr[i] != 1:
                    continue
                ds = ev - i
                dh = _hours_between(t[i], t[ev])
                if dh is None or dh < 0:
                    continue
                if best_step is None or ds < best_step:
                    best_step = ds
                    best_h = dh
            if best_step is not None:
                detected_events += 1
                lead_steps_all.append(int(best_step))
                lead_hours_all.append(float(best_h) if best_h is not None else float(best_step))

            for k in k_steps:
                lo = max(0, ev - k)
                if any(pr[j] == 1 for j in range(lo, ev)):
                    event_hits_prevk[k] += 1

        # Alert-centric: does event occur in next k steps after alert
        pos_idx = [i for i, v in enumerate(pr) if v == 1]
        for i in pos_idx:
            for k in k_steps:
                hi = min(n, i + k + 1)
                if any(gt[j] == 1 for j in range(i + 1, hi)):
                    alert_hits[k] += 1

    out = {
        "n_events": int(n_events),
        "event_detected_rate": (float(detected_events / n_events) if n_events > 0 else None),
        "median_lead_steps": (float(np.median(lead_steps_all)) if lead_steps_all else None),
        "mean_lead_steps": (float(np.mean(lead_steps_all)) if lead_steps_all else None),
        "median_lead_hours": (float(np.median(lead_hours_all)) if lead_hours_all else None),
        "mean_lead_hours": (float(np.mean(lead_hours_all)) if lead_hours_all else None),
        "alerts_total": int(alerts_total),
        "alerts_with_event_within_ks_steps_rate": {},
        "events_with_alert_in_prev_ks_steps_rate": {},
        "alerts_per_event": (float(alerts_total / n_events) if n_events > 0 else None),
    }
    for k in k_steps:
        out["alerts_with_event_within_ks_steps_rate"][f"k{k}"] = (float(alert_hits[k] / alerts_total) if alerts_total > 0 else None)
        out["events_with_alert_in_prev_ks_steps_rate"][f"k{k}"] = (float(event_hits_prevk[k] / n_events) if n_events > 0 else None)
    return out


def _schema_metrics(stage_rows: List[Dict], required_fields: List[str], known_fields: Optional[set] = None) -> Dict[str, float]:
    n = len(stage_rows)
    if n == 0:
        return {
            "schema_valid_rate": 0.0,
            "malformed_output_rate": 0.0,
            "empty_output_rate": 0.0,
            "hallucinated_field_rate": 0.0,
        }
    valid = 0
    empty = 0
    halluc = 0
    for o in stage_rows:
        if all(k in o for k in required_fields):
            valid += 1
        if not o or len(o.keys()) == 0:
            empty += 1
        if known_fields is not None:
            extra = [k for k in o.keys() if k not in known_fields]
            if extra:
                halluc += 1
    return {
        "schema_valid_rate": float(valid / n),
        "malformed_output_rate": float(1.0 - valid / n),
        "empty_output_rate": float(empty / n),
        "hallucinated_field_rate": float(halluc / n) if known_fields is not None else 0.0,
    }


def _cohen_kappa_binary(y1: np.ndarray, y2: np.ndarray) -> Optional[float]:
    if len(y1) == 0 or len(y2) == 0 or len(y1) != len(y2):
        return None
    y1 = y1.astype(int)
    y2 = y2.astype(int)
    po = float(np.mean(y1 == y2))
    p1 = float(np.mean(y1))
    p2 = float(np.mean(y2))
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) < 1e-12:
        return None
    return float((po - pe) / (1.0 - pe))


def _read_json_if_exists(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _sort_key_for_traj(o: Dict):
    step = o.get("step_id", 0)
    try:
        step = int(step)
    except Exception:
        step = 0
    return (str(o.get("anchor_time", "")), step)


def _run_lengths(bin_arr: np.ndarray, value: int = 1) -> List[int]:
    out = []
    cur = 0
    for x in bin_arr.tolist():
        if int(x) == int(value):
            cur += 1
        else:
            if cur > 0:
                out.append(cur)
                cur = 0
    if cur > 0:
        out.append(cur)
    return out


def _compute_stability_and_recovery(s4: List[Dict]) -> Tuple[Dict, Dict]:
    if not s4:
        return {}, {}

    by_traj: Dict[Tuple[str, str, str], List[Dict]] = {}
    for o in s4:
        k = (str(o.get("source_dataset")), str(o.get("patient_id")), str(o.get("encounter_id")))
        by_traj.setdefault(k, []).append(o)

    dim_keys = STATE_KEYS
    stability = {}
    recovery = {}

    for dim in dim_keys:
        flips = 0
        transitions = 0
        oscillations = 0
        osc_den = 0
        abs_deltas = []
        state_vals = []
        pos_run_lengths = []
        rebound_events = 0
        recovery_events = 0
        steps_to_recovery = []
        unstable_starts = 0

        for _, seq in by_traj.items():
            seq = sorted(seq, key=_sort_key_for_traj)
            vals = []
            deltas = []
            for o in seq:
                st = o.get("individual_protocol_state_next", {}) if isinstance(o.get("individual_protocol_state_next", {}), dict) else {}
                de = o.get("individual_protocol_state_delta", {}) if isinstance(o.get("individual_protocol_state_delta", {}), dict) else {}
                try:
                    v = int(st.get(dim, 0))
                except Exception:
                    v = 0
                try:
                    d = int(de.get(dim, 0))
                except Exception:
                    d = 0
                vals.append(v)
                deltas.append(d)
                abs_deltas.append(abs(d))
                state_vals.append(v)
                if d < 0:
                    recovery_events += 1

            if len(vals) < 2:
                continue

            b = (np.array(vals, dtype=int) >= 1).astype(int)
            transitions += max(0, len(b) - 1)
            flips += int(np.sum(b[1:] != b[:-1]))
            pos_run_lengths.extend(_run_lengths(b, value=1))

            # Oscillation pattern: 1-0-1 or 0-1-0
            if len(b) >= 3:
                for i in range(len(b) - 2):
                    tri = (int(b[i]), int(b[i + 1]), int(b[i + 2]))
                    if tri in {(1, 0, 1), (0, 1, 0)}:
                        oscillations += 1
                    osc_den += 1

            # Recovery latency: first transition from unstable (>=1) to negative delta
            first_unstable_idx = None
            first_recovery_idx = None
            for i, v in enumerate(vals):
                if v >= 1:
                    first_unstable_idx = i
                    break
            if first_unstable_idx is not None:
                unstable_starts += 1
                for j in range(first_unstable_idx, len(deltas)):
                    if deltas[j] < 0:
                        first_recovery_idx = j
                        break
                if first_recovery_idx is not None:
                    steps_to_recovery.append(int(first_recovery_idx - first_unstable_idx))

            # Rebound after recovery: negative delta followed by positive delta within next 2 steps
            for i, d in enumerate(deltas):
                if d < 0:
                    w = deltas[i + 1:i + 3]
                    if any(x > 0 for x in w):
                        rebound_events += 1

        stability[dim] = {
            "state_flip_rate": float(flips / transitions) if transitions > 0 else None,
            "oscillation_pattern_rate": float(oscillations / osc_den) if osc_den > 0 else None,
            "mean_abs_state_delta": float(np.mean(abs_deltas)) if abs_deltas else None,
            "state_value_std": float(np.std(state_vals)) if state_vals else None,
            "mean_positive_state_persistence_steps": float(np.mean(pos_run_lengths)) if pos_run_lengths else 0.0,
        }
        recovery[dim] = {
            "recovery_transition_rate": float(recovery_events / len(s4)) if len(s4) > 0 else None,
            "rebound_after_recovery_rate": float(rebound_events / recovery_events) if recovery_events > 0 else 0.0,
            "n_recovery_events": int(recovery_events),
            "n_unstable_trajectories": int(unstable_starts),
            "mean_steps_to_first_recovery_after_instability": float(np.mean(steps_to_recovery)) if steps_to_recovery else None,
            "median_steps_to_first_recovery_after_instability": float(np.median(steps_to_recovery)) if steps_to_recovery else None,
        }

    return stability, recovery


def _state_signal_from_facts(facts: List[Dict]) -> Dict[str, int]:
    fmap = feature_map_from_facts(facts or [])
    # Hemodynamic
    map_v = fmap.get("map", {}).get("value_last")
    lac_v = fmap.get("lactate", {}).get("value_last")
    lac_t = str(fmap.get("lactate", {}).get("trend", "")).lower()
    hemo = int(
        (map_v is not None and float(map_v) < 65.0)
        or (lac_v is not None and float(lac_v) >= 2.0)
        or (lac_t == "rising")
    )
    # Respiratory
    spo2_v = fmap.get("spo2", {}).get("value_last")
    rr_v = fmap.get("rr", {}).get("value_last")
    rr_t = str(fmap.get("rr", {}).get("trend", "")).lower()
    resp = int(
        (spo2_v is not None and float(spo2_v) < 90.0)
        or (rr_v is not None and float(rr_v) > 28.0)
        or (rr_t == "rising")
    )
    # Renal
    cr_v = fmap.get("creatinine", {}).get("value_last")
    cr_t = str(fmap.get("creatinine", {}).get("trend", "")).lower()
    cr_f = str(fmap.get("creatinine", {}).get("abnormal_flag_last", "")).lower()
    renal = int(
        (cr_v is not None and float(cr_v) >= 1.5)
        or (cr_t == "rising")
        or (cr_f == "high")
    )
    # Metabolic
    bicarb_v = fmap.get("bicarbonate", {}).get("value_last")
    gluc_v = fmap.get("glucose", {}).get("value_last")
    na_v = fmap.get("sodium", {}).get("value_last")
    k_v = fmap.get("potassium", {}).get("value_last")
    metab = int(
        (bicarb_v is not None and float(bicarb_v) < 22.0)
        or (gluc_v is not None and (float(gluc_v) < 70.0 or float(gluc_v) > 180.0))
        or (na_v is not None and (float(na_v) < 135.0 or float(na_v) > 145.0))
        or (k_v is not None and (float(k_v) < 3.5 or float(k_v) > 5.5))
    )
    # Systemic inflammation
    wbc_v = fmap.get("wbc", {}).get("value_last")
    inflam = int(wbc_v is not None and (float(wbc_v) < 4.0 or float(wbc_v) > 12.0))
    global_det = int(hemo or resp or renal or metab or inflam)
    return {
        "hemodynamic_state": hemo,
        "respiratory_state": resp,
        "renal_state": renal,
        "metabolic_state": metab,
        "systemic_inflammation_state": inflam,
        "global_deterioration_state": global_det,
    }


def _fit_label_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: Optional[np.ndarray] = None) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_thr = 0.5
    best_f1 = -1.0
    for thr in grid:
        y_pred = (y_score >= float(thr)).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _patient_keys(df: pd.DataFrame) -> pd.Series:
    return df["source_dataset"].astype(str) + "||" + df["patient_id"].astype(str)


def _calibrate_thresholds(pred_df: pd.DataFrame, seed: int = 13, calib_frac: float = 0.2):
    if pred_df.empty:
        return {"thresholds": {k: 0.5 for k in LABELS}, "n_calib_rows": 0, "n_eval_rows": 0, "calibration_patient_frac": float(calib_frac), "seed": int(seed), "eval_df": pred_df}
    keys = _patient_keys(pred_df).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_cal = max(1, int(len(keys) * calib_frac))
    cal_keys = set(keys[:n_cal])
    cal_df = pred_df[_patient_keys(pred_df).isin(cal_keys)].copy()
    eval_df = pred_df[~_patient_keys(pred_df).isin(cal_keys)].copy()
    if eval_df.empty:
        eval_df = pred_df.copy()
    thresholds = {}
    for lab in LABELS:
        yt = cal_df[f"gt_{lab}"].to_numpy(dtype=int)
        yp = cal_df[f"pred_{lab}"].to_numpy(dtype=float)
        thresholds[lab] = _fit_label_threshold(yt, yp)
    return {
        "thresholds": thresholds,
        "n_calib_rows": int(len(cal_df)),
        "n_eval_rows": int(len(eval_df)),
        "calibration_patient_frac": float(calib_frac),
        "seed": int(seed),
        "eval_df": eval_df,
    }


def run(out_dir: str, protocol_json: Optional[str] = None, bootstrap_replicates: int = 1000):
    s1_path = os.path.join(out_dir, "stage1_router.jsonl")
    s2_path = os.path.join(out_dir, "stage2_reasoner.jsonl")
    s3_path = os.path.join(out_dir, "stage3_auditor.jsonl")
    s4_path = os.path.join(out_dir, "stage4_steward.jsonl")

    s1 = [o for o in iter_jsonl(s1_path)]
    s2 = [o for o in iter_jsonl(s2_path)]
    s3 = [o for o in iter_jsonl(s3_path)]
    s4 = [o for o in iter_jsonl(s4_path)]

    if any(not row.get("target_isolation_verified", False) for row in s1 + s2 + s3 + s4):
        raise ValueError(
            "Evaluation requires target-free corrected stage outputs; rerun inference first."
        )
    if any(
        row.get("reasoner_prediction", {}).get("any_deterioration_definition")
        != "max_support_probability"
        for row in s2
    ):
        raise ValueError(
            "Evaluation requires the three-support-plus-max-composite endpoint schema."
        )

    n = len(s2)
    rows = []
    y = {k: [] for k in LABELS}
    p = {k: [] for k in LABELS}

    for o in s2:
        gt = add_composite_label(o.get("stage2_ground_truth", {}))
        pred = o.get("reasoner_prediction", {})
        proxy = o.get("stage2_prediction", {})
        rp = add_composite_probability(pred.get("risk_probs", {}))

        rec = {
            "record_id": o.get("record_id"),
            "example_id": o.get("example_id"),
            "source_dataset": o.get("source_dataset"),
            "patient_id": o.get("patient_id"),
            "encounter_id": o.get("encounter_id"),
            "anchor_time": o.get("anchor_time"),
            "step_id": int(o.get("step_id", 0)) if str(o.get("step_id", "")).isdigit() else 0,
        }

        for lab in LABELS:
            gt_v = int(gt.get(lab, 0))
            # prefer continuous risk_probs; fallback to binary proxy
            pr_v = float(rp.get(lab, proxy.get(lab, 0)))
            if not np.isfinite(pr_v):
                pr_v = 0.0
            pr_v = float(np.clip(pr_v, 0.0, 1.0))
            y[lab].append(gt_v)
            p[lab].append(pr_v)
            rec[f"gt_{lab}"] = gt_v
            rec[f"pred_{lab}"] = pr_v

        rows.append(rec)

    pred_df = pd.DataFrame(rows)

    per_label = []
    aurocs, auprcs = [], []
    for lab in LABELS:
        yt = np.array(y[lab], dtype=int)
        yp = np.array(p[lab], dtype=float)
        auroc = _safe_auroc(yt, yp)
        auprc = _safe_auprc(yt, yp)
        bm = _bin_metrics(yt, yp)
        spec = _safe_specificity(yt, yp)
        ece = _ece(yt, yp)
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
        per_label.append({
            "label": lab,
            "n": int(len(yt)),
            "prevalence": float(yt.mean()) if len(yt) else 0.0,
            "auroc": auroc,
            "auprc": auprc,
            "specificity": spec,
            "ece": ece,
            **bm,
        })

    # micro metrics by flattening
    y_flat = np.concatenate([np.array(y[k], dtype=int) for k in LABELS]) if n else np.array([], dtype=int)
    p_flat = np.concatenate([np.array(p[k], dtype=float) for k in LABELS]) if n else np.array([], dtype=float)

    overall = {
        "n_examples": n,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "micro_auroc": _safe_auroc(y_flat, p_flat) if len(y_flat) else None,
        "micro_auprc": _safe_auprc(y_flat, p_flat) if len(y_flat) else None,
        "micro_f1": float(f1_score(y_flat, (p_flat >= 0.5).astype(int), zero_division=0)) if len(y_flat) else None,
        "micro_precision": float(precision_score(y_flat, (p_flat >= 0.5).astype(int), zero_division=0)) if len(y_flat) else None,
        "micro_recall": float(recall_score(y_flat, (p_flat >= 0.5).astype(int), zero_division=0)) if len(y_flat) else None,
        "micro_specificity": _safe_specificity(y_flat, p_flat) if len(y_flat) else None,
        "micro_brier": float(brier_score_loss(y_flat, p_flat)) if len(y_flat) else None,
        "micro_ece": _ece(y_flat, p_flat) if len(y_flat) else None,
        "prediction_endpoint_schema": "three_support_plus_max_composite_v1",
        "evaluation_schema": "corrected_evaluation_v2",
    }
    overall.update(_bootstrap_patient_ci(pred_df, n_boot=bootstrap_replicates))

    # per-label threshold calibration on held-out patient subset
    cal = _calibrate_thresholds(pred_df, seed=13, calib_frac=0.2)
    thr = cal["thresholds"]
    eval_df = cal["eval_df"]
    per_label_cal = []
    f1s, precs, recs, specs = [], [], [], []
    y_flat_cal, p_flat_cal, yhat_flat_cal = [], [], []
    for lab in LABELS:
        yt = eval_df[f"gt_{lab}"].to_numpy(dtype=int)
        yp = eval_df[f"pred_{lab}"].to_numpy(dtype=float)
        t = float(thr.get(lab, 0.5))
        yhat = (yp >= t).astype(int)
        y_flat_cal.append(yt)
        p_flat_cal.append(yp)
        yhat_flat_cal.append(yhat)
        f1v = float(f1_score(yt, yhat, zero_division=0)) if len(yt) else None
        prv = float(precision_score(yt, yhat, zero_division=0)) if len(yt) else None
        rev = float(recall_score(yt, yhat, zero_division=0)) if len(yt) else None
        spv = _safe_specificity(yt, yp, thr=t)
        if f1v is not None:
            f1s.append(f1v)
        if prv is not None:
            precs.append(prv)
        if rev is not None:
            recs.append(rev)
        if spv is not None:
            specs.append(spv)
        per_label_cal.append(
            {
                "label": lab,
                "threshold": t,
                "n_eval": int(len(yt)),
                "f1": f1v,
                "precision": prv,
                "recall": rev,
                "specificity": spv,
            }
        )
    y_flat_cal = np.concatenate(y_flat_cal) if y_flat_cal else np.array([], dtype=int)
    p_flat_cal = np.concatenate(p_flat_cal) if p_flat_cal else np.array([], dtype=float)
    yhat_flat_cal = np.concatenate(yhat_flat_cal) if yhat_flat_cal else np.array([], dtype=int)
    calibrated_metrics = {
        "n_eval_examples": int(len(eval_df)),
        "thresholds": {k: float(v) for k, v in thr.items()},
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_precision": float(np.mean(precs)) if precs else None,
        "macro_recall": float(np.mean(recs)) if recs else None,
        "macro_specificity": float(np.mean(specs)) if specs else None,
        "micro_f1": float(f1_score(y_flat_cal, yhat_flat_cal, zero_division=0)) if len(y_flat_cal) else None,
        "micro_precision": float(precision_score(y_flat_cal, yhat_flat_cal, zero_division=0)) if len(y_flat_cal) else None,
        "micro_recall": float(recall_score(y_flat_cal, yhat_flat_cal, zero_division=0)) if len(y_flat_cal) else None,
        "micro_specificity": (float(np.sum((y_flat_cal == 0) & (yhat_flat_cal == 0)) / max(1, np.sum(y_flat_cal == 0))) if len(y_flat_cal) else None),
        "micro_auroc": _safe_auroc(y_flat_cal, p_flat_cal) if len(y_flat_cal) else None,
        "micro_auprc": _safe_auprc(y_flat_cal, p_flat_cal) if len(y_flat_cal) else None,
        "micro_ece": _ece(y_flat_cal, p_flat_cal) if len(y_flat_cal) else None,
        "n_calib_rows": int(cal["n_calib_rows"]),
    }

    # stagewise diagnostics
    n_stage3_fail = int(np.sum([str(o.get("audit", {}).get("status", "PASS")).upper() == "FAIL" for o in s3])) if s3 else 0
    n_stage3_pass = int(len(s3) - n_stage3_fail) if s3 else 0
    stagewise = {
        "stage1": {
            "n": len(s1),
            "selection_rate": float(np.mean([len(o.get("selected_rule_ids", [])) > 0 for o in s1])) if s1 else 0.0,
            "avg_selected_rules": float(np.mean([len(o.get("selected_rule_ids", [])) for o in s1])) if s1 else 0.0,
        },
        "stage2": {
            "n": len(s2),
            "any_action_rate": float(np.mean([len(o.get("reasoner_prediction", {}).get("predicted_actions", [])) > 0 for o in s2])) if s2 else 0.0,
            "avg_actions": float(np.mean([len(o.get("reasoner_prediction", {}).get("predicted_actions", [])) for o in s2])) if s2 else 0.0,
        },
        "stage3": {
            "n": len(s3),
            "n_fail": int(n_stage3_fail),
            "n_pass": int(n_stage3_pass),
            "fail_rate": float(n_stage3_fail / len(s3)) if s3 else 0.0,
            "pass_rate": float(n_stage3_pass / len(s3)) if s3 else 0.0,
            "avg_issues": float(np.mean([len(o.get("audit", {}).get("issues", [])) for o in s3])) if s3 else 0.0,
        },
        "stage4": {
            "n": len(s4),
            "avg_abs_state_delta": {
                k: float(np.mean([abs(int(o.get("individual_protocol_state_delta", {}).get(k, 0))) for o in s4])) if s4 else 0.0
                for k in STATE_KEYS
            },
            "state_changed_rate": float(np.mean([
                any(int(o.get("individual_protocol_state_delta", {}).get(k, 0)) != 0 for k in STATE_KEYS)
                for o in s4
            ])) if s4 else 0.0,
        },
    }

    # communication/schema quality
    stagewise["communication_stability"] = {
        "stage1": _schema_metrics(
            s1,
            required_fields=["example_id", "packet", "selected_rule_ids", "stage1_prediction"],
            known_fields=COMMON_STAGE_FIELDS,
        ),
        "stage2": _schema_metrics(
            s2,
            required_fields=["example_id", "reasoner_prediction", "selected_rule_ids", "packet"],
            known_fields=COMMON_STAGE_FIELDS,
        ),
        "stage3": _schema_metrics(
            s3,
            required_fields=["example_id", "audit", "reasoner_prediction", "selected_rule_ids"],
            known_fields=COMMON_STAGE_FIELDS,
        ),
        "stage4": _schema_metrics(
            s4,
            required_fields=["example_id", "individual_protocol_state_prev", "individual_protocol_state_next", "individual_protocol_state_delta"],
            known_fields=COMMON_STAGE_FIELDS,
        ),
    }

    # temporal early-warning on major labels
    temporal = {
        "vasopressor_signal": _early_warning_metrics(pred_df, "vasopressor_signal", [6, 12, 24]),
        "resp_support_signal": _early_warning_metrics(pred_df, "resp_support_signal", [6, 12, 24]),
        "renal_support_signal": _early_warning_metrics(pred_df, "renal_support_signal", [6, 12, 24]),
        "any_deterioration": _early_warning_metrics(pred_df, "any_deterioration", [6, 12, 24]),
    }
    # event-level + k-step tolerant metrics
    event_level = {
        "vasopressor_signal": _event_level_and_kstep_metrics(pred_df, "vasopressor_signal", [1, 2, 3, 4]),
        "resp_support_signal": _event_level_and_kstep_metrics(pred_df, "resp_support_signal", [1, 2, 3, 4]),
        "renal_support_signal": _event_level_and_kstep_metrics(pred_df, "renal_support_signal", [1, 2, 3, 4]),
        "any_deterioration": _event_level_and_kstep_metrics(pred_df, "any_deterioration", [1, 2, 3, 4]),
    }

    # risk-state evaluation (using stage4 state_next vs supervised targets)
    risk_state = {}
    s4_df = pd.DataFrame(s4) if s4 else pd.DataFrame()
    if not s4_df.empty and not pred_df.empty:
        jcols = _join_cols(pred_df, s4_df)
        merged = pred_df.merge(
            s4_df[jcols + ["individual_protocol_state_next"]].drop_duplicates(subset=jcols, keep="first"),
            on=jcols,
            how="left",
        )
        for s_key, t_key in RISK_STATE_TO_TARGET.items():
            scores = []
            for x in merged["individual_protocol_state_next"].tolist():
                if isinstance(x, dict):
                    try:
                        scores.append(float(x.get(s_key, 0)))
                    except Exception:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            yt = merged[f"gt_{t_key}"].to_numpy(dtype=int)
            yp = np.array(scores, dtype=float)
            m = _safe_metrics_block(yt, yp)
            pred_prev = float(np.mean(yp >= 1.0)) if len(yp) else None
            gt_prev = float(np.mean(yt)) if len(yt) else None
            risk_state[s_key] = {
                "target_label": t_key,
                **m,
                "risk_prevalence_pred": pred_prev,
                "risk_prevalence_gt": gt_prev,
                "risk_prevalence_abs_diff": (abs(pred_prev - gt_prev) if (pred_prev is not None and gt_prev is not None) else None),
                "severity_stratification": {
                    "gt_rate_by_state_ge1": float(np.mean(yt[yp >= 1.0])) if np.any(yp >= 1.0) else None,
                    "gt_rate_by_state_eq0": float(np.mean(yt[yp < 1.0])) if np.any(yp < 1.0) else None,
                },
            }

    # protocol consistency (against deterministic trigger ground-truth)
    protocol_consistency = {}
    if protocol_json is not None and os.path.exists(protocol_json):
        rules = load_protocol(protocol_json)
        tp = fp = fn = 0
        unsupported = 0
        all_rule_ids = set(rules.keys())
        for o in s1:
            facts = o.get("packet", {}).get("facts", [])
            fmap = feature_map_from_facts(facts)
            gt_active = set()
            for rid, rule in rules.items():
                try:
                    ok, _ = rule_score(rule, fmap)
                    if ok:
                        gt_active.add(rid)
                except Exception:
                    continue
            pred_active = set([str(x) for x in o.get("selected_rule_ids", []) if isinstance(x, str)])
            unsupported += int(any(r not in all_rule_ids for r in pred_active))
            tp += len(pred_active & gt_active)
            fp += len(pred_active - gt_active)
            fn += len(gt_active - pred_active)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        protocol_consistency = {
            "rule_activation_precision": prec,
            "rule_activation_recall": rec,
            "rule_violation_rate": float(fp / (tp + fp)) if (tp + fp) > 0 else None,
            "unsupported_rule_generation_rate": float(unsupported / len(s1)) if s1 else None,
        }

    # cross-dataset split summary
    cross_dataset = {}
    if not pred_df.empty and "source_dataset" in pred_df.columns:
        for ds, g in pred_df.groupby("source_dataset"):
            ds_metrics = {}
            for lab in LABELS:
                yt = g[f"gt_{lab}"].to_numpy(dtype=int)
                yp = g[f"pred_{lab}"].to_numpy(dtype=float)
                ds_metrics[lab] = _safe_metrics_block(yt, yp)
            macro_auroc_vals = [v["auroc"] for v in ds_metrics.values() if v["auroc"] is not None]
            macro_auprc_vals = [v["auprc"] for v in ds_metrics.values() if v["auprc"] is not None]
            cross_dataset[str(ds)] = {
                "n": int(len(g)),
                "macro_auroc": float(np.mean(macro_auroc_vals)) if macro_auroc_vals else None,
                "macro_auprc": float(np.mean(macro_auprc_vals)) if macro_auprc_vals else None,
                "micro_ece": _ece(
                    np.concatenate([g[f"gt_{k}"].to_numpy(dtype=int) for k in LABELS]),
                    np.concatenate([g[f"pred_{k}"].to_numpy(dtype=float) for k in LABELS]),
                ),
            }
        if "mimic" in cross_dataset and "eicu" in cross_dataset:
            ma = cross_dataset["mimic"]["macro_auroc"]
            mb = cross_dataset["eicu"]["macro_auroc"]
            pa = cross_dataset["mimic"]["macro_auprc"]
            pb = cross_dataset["eicu"]["macro_auprc"]
            cross_dataset["transfer_drop_mimic_to_eicu"] = {
                "macro_auroc_drop": (ma - mb) if (ma is not None and mb is not None) else None,
                "macro_auprc_drop": (pa - pb) if (pa is not None and pb is not None) else None,
            }

    # cross-stage agreement: stage2 risk predictions vs stage4 interpretable state
    cross_stage_agreement = {}
    if not pred_df.empty and s4:
        s4_df = pd.DataFrame(s4)
        jcols = _join_cols(pred_df, s4_df)
        m = pred_df.merge(
            s4_df[jcols + ["individual_protocol_state_next"]].drop_duplicates(subset=jcols, keep="first"),
            on=jcols,
            how="inner",
        )
        mapping = {
            "vasopressor_signal": "hemodynamic_state",
            "resp_support_signal": "respiratory_state",
            "renal_support_signal": "renal_state",
            "any_deterioration": "metabolic_state",
        }
        for lab, st_key in mapping.items():
            y2 = (m[f"pred_{lab}"].to_numpy(dtype=float) >= 0.5).astype(int)
            y4 = np.array([
                int((x or {}).get(st_key, 0) >= 1) if isinstance(x, dict) else 0
                for x in m["individual_protocol_state_next"].tolist()
            ], dtype=int)
            cross_stage_agreement[lab] = {
                "state_key": st_key,
                "n": int(len(y2)),
                "agreement_rate": float(np.mean(y2 == y4)) if len(y2) else None,
                "cohen_kappa": _cohen_kappa_binary(y2, y4),
                "stage2_positive_rate": float(np.mean(y2)) if len(y2) else None,
                "stage4_positive_rate": float(np.mean(y4)) if len(y4) else None,
            }

    # protocol state coherence: stage4 state vs physiology evidence in facts
    protocol_state_coherence = {}
    if s4:
        dim_keys = [
            "hemodynamic_state",
            "respiratory_state",
            "renal_state",
            "metabolic_state",
            "systemic_inflammation_state",
            "global_deterioration_state",
        ]
        vals = {k: {"state_bin": [], "evidence_bin": []} for k in dim_keys}
        for o in s4:
            state_next = o.get("individual_protocol_state_next", {}) if isinstance(o.get("individual_protocol_state_next", {}), dict) else {}
            facts = o.get("packet", {}).get("facts", [])
            evidence = _state_signal_from_facts(facts)
            for k in dim_keys:
                sbin = int(state_next.get(k, 0) >= 1) if k in state_next else 0
                ebin = int(evidence.get(k, 0))
                vals[k]["state_bin"].append(sbin)
                vals[k]["evidence_bin"].append(ebin)
        for k in dim_keys:
            sbin = np.array(vals[k]["state_bin"], dtype=int)
            ebin = np.array(vals[k]["evidence_bin"], dtype=int)
            tp = int(np.sum((sbin == 1) & (ebin == 1)))
            fp = int(np.sum((sbin == 1) & (ebin == 0)))
            fn = int(np.sum((sbin == 0) & (ebin == 1)))
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
            f1 = float(2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
            protocol_state_coherence[k] = {
                "n": int(len(sbin)),
                "coherence_agreement_rate": float(np.mean(sbin == ebin)) if len(sbin) else None,
                "state_when_evidence_precision": prec,
                "evidence_when_state_recall": rec,
                "coherence_f1": f1,
                "state_positive_rate": float(np.mean(sbin)) if len(sbin) else None,
                "evidence_positive_rate": float(np.mean(ebin)) if len(ebin) else None,
            }

    # state stability + recovery dynamics
    state_stability_metrics, recovery_dynamics_metrics = _compute_stability_and_recovery(s4)

    # efficiency summary (from stage metrics files + lightweight payload proxies)
    m_router = _read_json_if_exists(os.path.join(out_dir, "metrics_router.json"))
    m_reasoner = _read_json_if_exists(os.path.join(out_dir, "metrics_reasoner.json"))
    m_auditor = _read_json_if_exists(os.path.join(out_dir, "metrics_auditor.json"))
    m_steward = _read_json_if_exists(os.path.join(out_dir, "metrics_steward.json"))
    by_stage = {
        "router": m_router,
        "reasoner": m_reasoner,
        "auditor": m_auditor,
        "steward": m_steward,
    }
    llm_calls_by_stage = {
        k: int(v.get("llm_calls", 0)) if isinstance(v, dict) else 0
        for k, v in by_stage.items()
    }
    llm_failures_by_stage = {
        k: int(v.get("llm_failures", 0)) if isinstance(v, dict) else 0
        for k, v in by_stage.items()
    }
    total_calls = int(sum(llm_calls_by_stage.values()))
    total_failures = int(sum(llm_failures_by_stage.values()))
    n_examples_eff = int(overall.get("n_examples", 0) or 0)
    # proxy payload sizes (characters); token accounting is model-tokenizer specific and not tracked here.
    avg_packet_chars = float(np.mean([len(json.dumps(o.get("packet", {}), default=str)) for o in s1])) if s1 else None
    avg_reasoner_pred_chars = float(np.mean([len(json.dumps(o.get("reasoner_prediction", {}), default=str)) for o in s2])) if s2 else None
    efficiency = {
        "n_examples": n_examples_eff,
        "llm_calls_by_stage": llm_calls_by_stage,
        "llm_failures_by_stage": llm_failures_by_stage,
        "llm_calls_total": total_calls,
        "llm_failures_total": total_failures,
        "llm_failure_rate_per_call": (float(total_failures / total_calls) if total_calls > 0 else None),
        "llm_calls_per_example": (float(total_calls / n_examples_eff) if n_examples_eff > 0 else None),
        "auditor_pass_rate": float(n_stage3_pass / len(s3)) if s3 else None,
        "auditor_fail_rate": float(n_stage3_fail / len(s3)) if s3 else None,
        "auditor_n_pass": int(n_stage3_pass),
        "auditor_n_fail": int(n_stage3_fail),
        "avg_packet_chars_stage1_proxy": avg_packet_chars,
        "avg_reasoner_prediction_chars_proxy": avg_reasoner_pred_chars,
    }

    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "metrics_overall.json"), overall)
    write_json(os.path.join(out_dir, "stagewise_metrics.json"), stagewise)
    write_json(os.path.join(out_dir, "temporal_metrics.json"), temporal)
    write_json(os.path.join(out_dir, "event_level_metrics.json"), event_level)
    write_json(os.path.join(out_dir, "risk_state_metrics.json"), risk_state)
    write_json(os.path.join(out_dir, "protocol_consistency_metrics.json"), protocol_consistency)
    write_json(os.path.join(out_dir, "cross_dataset_metrics.json"), cross_dataset)
    write_json(os.path.join(out_dir, "cross_stage_agreement.json"), cross_stage_agreement)
    write_json(os.path.join(out_dir, "protocol_state_coherence.json"), protocol_state_coherence)
    write_json(os.path.join(out_dir, "state_stability_metrics.json"), state_stability_metrics)
    write_json(os.path.join(out_dir, "recovery_dynamics_metrics.json"), recovery_dynamics_metrics)
    write_json(os.path.join(out_dir, "efficiency_metrics.json"), efficiency)
    write_json(os.path.join(out_dir, "calibrated_metrics.json"), calibrated_metrics)
    write_json(os.path.join(out_dir, "calibrated_thresholds.json"), {"thresholds": calibrated_metrics.get("thresholds", {}), "n_calib_rows": calibrated_metrics.get("n_calib_rows", 0)})
    pd.DataFrame(per_label).to_csv(os.path.join(out_dir, "per_label_metrics.csv"), index=False)
    pd.DataFrame(per_label_cal).to_csv(os.path.join(out_dir, "per_label_metrics_calibrated.csv"), index=False)
    pred_df.to_csv(os.path.join(out_dir, "predictions_with_gt.csv"), index=False)
    pred_df.to_parquet(os.path.join(out_dir, "predictions_with_gt.parquet"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol-json", default=None)
    ap.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = ap.parse_args()
    run(args.out_dir, args.protocol_json, args.bootstrap_replicates)

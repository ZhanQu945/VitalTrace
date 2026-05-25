import argparse
import json
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RESULTS_ROOT_DEFAULT = "./runs"
MIMIC_ROOT_DEFAULT = "./data/mimic"
EICU_ROOT_DEFAULT = "./data/eicu"


CANONICAL_COLUMNS = [
    "patient_id",
    "encounter_id",
    "source_dataset",
    "event_time",
    "event_type",
    "event_name",
    "value_num",
    "value_text",
    "unit",
    "abnormal_flag",
    "payload_json",
]

VITAL_RANGES = {
    "heart rate": (50.0, 110.0),
    "hr": (50.0, 110.0),
    "map": (65.0, 110.0),
    "mean blood pressure": (65.0, 110.0),
    "systolic blood pressure": (90.0, 160.0),
    "sbp": (90.0, 160.0),
    "diastolic blood pressure": (50.0, 100.0),
    "dbp": (50.0, 100.0),
    "respiratory rate": (10.0, 24.0),
    "rr": (10.0, 24.0),
    "spo2": (92.0, 100.0),
    "o2 saturation": (92.0, 100.0),
    "temperature": (36.0, 38.0),
    "temp": (36.0, 38.0),
}

FEATURE_ALIASES = {
    "lactate": ["lactate"],
    "hr": ["heart rate", "hr", "heartrate", "pulse rate"],
    "sbp": [
        "systolic blood pressure",
        "blood pressure systolic",
        "non invasive blood pressure systolic",
        "non-invasive blood pressure systolic",
        "nbp systolic",
        "nbp [systolic]",
        "abp systolic",
        "sbp",
        "systemicsystolic",
        "noninvasivesystolic",
        "arterial blood pressure systolic",
    ],
    "dbp": [
        "diastolic blood pressure",
        "blood pressure diastolic",
        "non invasive blood pressure diastolic",
        "non-invasive blood pressure diastolic",
        "nbp diastolic",
        "nbp [diastolic]",
        "abp diastolic",
        "dbp",
        "systemicdiastolic",
        "noninvasivediastolic",
        "arterial blood pressure diastolic",
    ],
    "rr": ["respiratory rate", "rr", "respiration", "resp rate"],
    "spo2": ["spo2", "o2 saturation", "sao2", "o2 sat", "pulseox", "pulse ox"],
    "temp": ["temperature", "temp", "temperature c", "temperature f"],
    "creatinine": ["creatinine"],
    "wbc": ["wbc", "white blood cell", "wbc count", "wbc x 1000"],
    "platelets": ["platelet", "platelets x 1000", "platelet count"],
    "hemoglobin": ["hemoglobin", "hgb"],
    "bun": ["bun", "urea nitrogen"],
    "pao2": ["pao2", "po2", "pao2/fio2", "p/f ratio"],
    "map": ["map", "mean blood pressure", "systemicmean", "noninvasivemean", "arterial blood pressure mean"],
}

CORE_PROTOCOL_FEATURES = {
    "map",
    "hr",
    "sbp",
    "dbp",
    "rr",
    "spo2",
    "temp",
    "lactate",
    "creatinine",
    "bun",
    "wbc",
    "platelets",
    "hemoglobin",
    "sodium",
    "potassium",
    "bicarbonate",
    "glucose",
    "bilirubin",
    "pao2",
}

LABEL_PATTERNS = {
    "legacy": {
        "vaso": re.compile(r"(?:norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine)", re.IGNORECASE),
        "resp": re.compile(r"(?:intub|ventilat)", re.IGNORECASE),
        "renal": re.compile(r"(?:dialysis)", re.IGNORECASE),
    },
    "expanded": {
        "vaso": re.compile(
            r"(?:norepinephrine|noradrenaline|levophed|vasopressin|epinephrine|adrenaline|phenylephrine|neosynephrine|dopamine|dobutamine|milrinone)",
            re.IGNORECASE,
        ),
        "resp": re.compile(
            r"(?:intub|endotracheal|ett|ventilat|mechanical vent|bipap|bi[- ]?pap|cpap|hfnc|high flow nasal)",
            re.IGNORECASE,
        ),
        "renal": re.compile(
            r"(?:dialysis|crrt|cvvh|cvvhd|cvvhdf|hemodialysis|hemofiltration|renal replacement|rrt)",
            re.IGNORECASE,
        ),
    },
}


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, msg: str):
        line = f"[{now_ts()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")


@dataclass
class RunPaths:
    run_id: str
    run_root: str
    artifacts_dir: str
    cache_dir: str
    logs_dir: str
    eval_dir: str
    samples_dir: str



def prepare_run_dirs(results_root: str, run_id: Optional[str]) -> RunPaths:
    rid = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(results_root, "runs", rid)
    artifacts = os.path.join(run_root, "artifacts")
    cache = os.path.join(run_root, "cache")
    logs = os.path.join(run_root, "logs")
    ev = os.path.join(run_root, "eval")
    samples = os.path.join(logs, "samples")
    for p in [run_root, artifacts, cache, logs, ev, samples]:
        os.makedirs(p, exist_ok=True)
    return RunPaths(rid, run_root, artifacts, cache, logs, ev, samples)


def set_cache_env(cache_dir: str):
    hf = os.path.join(cache_dir, "huggingface")
    torch_home = os.path.join(cache_dir, "torch")
    xdg = os.path.join(cache_dir, "xdg")
    tf_cache = os.path.join(hf, "transformers")
    hub = os.path.join(hf, "hub")
    for p in [hf, torch_home, xdg, tf_cache, hub]:
        os.makedirs(p, exist_ok=True)
    os.environ["HF_HOME"] = hf
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["TRANSFORMERS_CACHE"] = tf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    os.environ["TORCH_HOME"] = torch_home
    os.environ["XDG_CACHE_HOME"] = xdg


def write_manifest(paths: RunPaths, args: argparse.Namespace, inputs: Dict, outputs: Dict, stats: Dict):
    manifest_path = os.path.join(paths.logs_dir, "run_manifest.json")
    obj = {
        "run_id": paths.run_id,
        "created_at": now_ts(),
        "inputs": inputs,
        "outputs": outputs,
        "args": vars(args),
        "stats": stats,
    }
    with open(manifest_path, "w") as f:
        json.dump(obj, f, indent=2)


def _safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def _as_payload(d: Dict) -> str:
    return json.dumps(d, ensure_ascii=True)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
    out = df[keep].copy()
    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out = out[CANONICAL_COLUMNS]
    out = out.dropna(subset=["event_time", "patient_id", "encounter_id"])
    out["event_time"] = _safe_to_datetime(out["event_time"])
    out = out.dropna(subset=["event_time"])
    return out


def _sample_and_write(df: pd.DataFrame, path: str, n: int = 20):
    if df.empty:
        with open(path, "w") as f:
            json.dump([], f)
        return
    sample = df.sample(min(n, len(df)), random_state=42).to_dict("records")
    with open(path, "w") as f:
        json.dump(sample, f, indent=2, default=str)


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _normalize_flag(value: float, low: Optional[float], high: Optional[float]) -> str:
    if pd.isna(value):
        return "unknown"
    if low is not None and not pd.isna(low) and value < low:
        return "low"
    if high is not None and not pd.isna(high) and value > high:
        return "high"
    return "normal"


def _vital_range_for_name(name: str) -> Tuple[Optional[float], Optional[float]]:
    n = str(name).lower()
    for k, v in VITAL_RANGES.items():
        if k in n:
            return v
    return None, None


def _aggregate_vitals_hourly(vit: pd.DataFrame, group_cols: List[str], max_points_per_hour: int) -> pd.DataFrame:
    if vit.empty:
        return vit
    vit = vit.copy()
    vit["event_time"] = _safe_to_datetime(vit["event_time"])
    vit = vit.dropna(subset=["event_time"])
    vit["hour_bin"] = vit["event_time"].dt.floor("h")
    vit = vit.sort_values(group_cols + ["event_time"])
    agg = (
        vit.groupby(group_cols + ["hour_bin"], as_index=False)
        .agg(
            value_num_last=("value_num", "last"),
            value_num_min=("value_num", "min"),
            value_num_max=("value_num", "max"),
            value_num_median=("value_num", "median"),
            n_obs=("value_num", "count"),
        )
    )
    if max_points_per_hour > 0:
        agg["n_obs"] = agg["n_obs"].clip(upper=max_points_per_hour)
    agg["event_time"] = agg["hour_bin"]
    agg["value_num"] = agg["value_num_last"]
    return agg


def _thin_labs(labs: pd.DataFrame, keep_every_hours: int, max_normal_per_feature_per_encounter: int = 24) -> pd.DataFrame:
    if labs.empty:
        return labs
    labs = labs.copy().sort_values(["patient_id", "encounter_id", "event_name", "event_time"])
    if keep_every_hours <= 0:
        return labs
    labs["hour_bin"] = pd.to_datetime(labs["event_time"]).dt.floor(f"{keep_every_hours}h")
    abnormal = labs[labs["abnormal_flag"].isin(["low", "high"])]
    normal_like = labs[~labs["abnormal_flag"].isin(["low", "high"])]
    normal_like = normal_like.groupby(
        ["patient_id", "encounter_id", "event_name", "hour_bin"], as_index=False
    ).tail(1)
    if max_normal_per_feature_per_encounter > 0 and not normal_like.empty:
        normal_like = normal_like.sort_values(["patient_id", "encounter_id", "event_name", "event_time"])
        normal_like["_rank"] = normal_like.groupby(["patient_id", "encounter_id", "event_name"]).cumcount() + 1
        normal_like = normal_like[normal_like["_rank"] <= max_normal_per_feature_per_encounter]
        normal_like = normal_like.drop(columns=["_rank"], errors="ignore")
    out = pd.concat([abnormal, normal_like], ignore_index=True)
    return out.drop(columns=["hour_bin"], errors="ignore").sort_values(
        ["source_dataset", "patient_id", "encounter_id", "event_time"]
    )


def _thin_interventions(events: pd.DataFrame, min_gap_minutes: int = 60) -> pd.DataFrame:
    if events.empty:
        return events
    events = events.copy().sort_values(["patient_id", "encounter_id", "event_name", "event_time"])
    events["event_time"] = _safe_to_datetime(events["event_time"])
    last_time = events.groupby(["patient_id", "encounter_id", "event_name"])["event_time"].shift(1)
    gap_min = (events["event_time"] - last_time).dt.total_seconds().div(60)
    keep = gap_min.isna() | (gap_min >= min_gap_minutes)
    return events[keep]


def _canonical_feature_name(event_name: str) -> str:
    n = str(event_name).lower()
    for canonical, aliases in FEATURE_ALIASES.items():
        if any(a in n for a in aliases):
            return canonical
    return n[:80]


def _extract_symbolic_facts(ctx: pd.DataFrame, anchor_time: pd.Timestamp, lookback_hours: int = 12) -> List[Dict]:
    if ctx.empty:
        return []
    c = ctx.copy()
    c["event_time"] = pd.to_datetime(c["event_time"], errors="coerce")
    c = c[c["event_time"].notna()]
    c = c[c["event_time"] >= (anchor_time - pd.Timedelta(hours=lookback_hours))]
    if c.empty:
        return []

    facts = []
    grouped = c.groupby(["event_type", "event_name"], as_index=False)
    for _, g in grouped:
        g = g.sort_values("event_time")
        last = g.iloc[-1]
        f_name = _canonical_feature_name(last["event_name"])
        if f_name not in CORE_PROTOCOL_FEATURES:
            continue
        value_last = None if pd.isna(last.get("value_num")) else float(last["value_num"])
        abnormal_last = str(last.get("abnormal_flag", ""))
        trend = "unknown"
        vals = pd.to_numeric(g["value_num"], errors="coerce").dropna()
        if len(vals) >= 2:
            delta = float(vals.iloc[-1] - vals.iloc[0])
            if delta > 0:
                trend = "rising"
            elif delta < 0:
                trend = "falling"
            else:
                trend = "stable"
        facts.append({
            "feature": f_name,
            "event_type": str(last["event_type"]),
            "event_name": str(last["event_name"]),
            "value_last": value_last,
            "abnormal_flag_last": abnormal_last,
            "trend": trend,
            "count": int(len(g)),
            "time_last": str(last["event_time"]),
        })
    return facts


def _build_latent_state(facts: List[Dict], ctx: pd.DataFrame) -> Dict:
    abnormal = [f for f in facts if str(f.get("abnormal_flag_last", "")).lower() in {"high", "low"}]
    map_low = any(f["feature"] == "map" and f.get("value_last") is not None and f["value_last"] < 65 for f in facts)
    lactate_rising = any(f["feature"] == "lactate" and f.get("trend") == "rising" for f in facts)
    vasopressor_active = ctx["event_name"].astype(str).str.lower().str.contains(
        "norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine"
    ).any() if not ctx.empty else False
    return {
        "hemodynamic_instability_score": int(len(abnormal) >= 3) + int(map_low) + int(lactate_rising),
        "respiratory_instability_score": int(any(f["feature"] in {"rr", "spo2"} and str(f.get("abnormal_flag_last", "")).lower() in {"high", "low"} for f in facts)),
        "renal_instability_score": int(any(f["feature"] == "creatinine" and (f.get("trend") == "rising" or str(f.get("abnormal_flag_last", "")).lower() == "high") for f in facts)),
        "vasopressor_active": int(vasopressor_active),
        "abnormal_feature_count": int(len(abnormal)),
    }


def _counterfactual_candidates(latent_state: Dict, facts: List[Dict]) -> List[Dict]:
    cands = []
    map_low = any(f["feature"] == "map" and f.get("value_last") is not None and f["value_last"] < 65 for f in facts)
    lactate_rising = any(f["feature"] == "lactate" and f.get("trend") == "rising" for f in facts)
    if map_low or lactate_rising:
        cands.append({
            "action": "start_or_titrate_vasopressor",
            "target_risk": "shock_progression",
            "expected_state_delta": {"hemodynamic_instability_score": -1},
        })
        cands.append({
            "action": "fluid_resuscitation_bolus",
            "target_risk": "hypoperfusion",
            "expected_state_delta": {"hemodynamic_instability_score": -1},
        })
    if any(f["feature"] == "spo2" and str(f.get("abnormal_flag_last", "")).lower() == "low" for f in facts):
        cands.append({
            "action": "escalate_oxygen_support",
            "target_risk": "respiratory_failure",
            "expected_state_delta": {"respiratory_instability_score": -1},
        })
    if any(f["feature"] == "creatinine" and (f.get("trend") == "rising" or str(f.get("abnormal_flag_last", "")).lower() == "high") for f in facts):
        cands.append({
            "action": "renal_dose_adjustment_and_fluid_review",
            "target_risk": "aki_progression",
            "expected_state_delta": {"renal_instability_score": -1},
        })
    return cands


def _derive_eicu_lab_flags(lab_df: pd.DataFrame, min_group_n: int = 200) -> pd.Series:
    if lab_df.empty:
        return pd.Series(dtype="object")
    x = lab_df.copy()
    x["value_num"] = pd.to_numeric(x["value_num"], errors="coerce")
    q = (
        x.groupby("event_name")["value_num"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "q10", 0.9: "q90"})
    )
    n = x.groupby("event_name")["value_num"].count().rename("n")
    q = q.join(n, how="left")

    flags = []
    for name, val in zip(x["event_name"], x["value_num"]):
        if pd.isna(val):
            flags.append("unknown")
            continue
        if name not in q.index or q.loc[name, "n"] < min_group_n:
            flags.append("unknown")
            continue
        lo, hi = q.loc[name, "q10"], q.loc[name, "q90"]
        flags.append(_normalize_flag(val, lo, hi))
    return pd.Series(flags, index=x.index)


def mimic_admissions(mimic_root: str) -> pd.DataFrame:
    adm = pd.read_csv(os.path.join(mimic_root, "admissions.csv.gz"))
    for c in ["admittime", "dischtime", "deathtime"]:
        if c in adm.columns:
            adm[c] = _safe_to_datetime(adm[c])
    adm["in_hospital_mortality"] = adm["deathtime"].notna().astype(int)
    adm = adm.sort_values(["subject_id", "admittime"])
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["readmit_30d"] = (
        (adm["next_admittime"].notna())
        & ((adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() <= 30 * 24 * 3600)
        & ((adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() >= 0)
    ).astype(int)
    return adm


def extract_mimic_diagnoses(mimic_root: str, include: bool) -> pd.DataFrame:
    if not include:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    diag = pd.read_csv(os.path.join(mimic_root, "diagnoses_icd.csv.gz"))
    d_diag = pd.read_csv(os.path.join(mimic_root, "d_icd_diagnoses.csv.gz"))
    adm = pd.read_csv(os.path.join(mimic_root, "admissions.csv.gz"), usecols=["subject_id", "hadm_id", "admittime"])
    adm["admittime"] = _safe_to_datetime(adm["admittime"])
    title_cols = [c for c in ["long_title", "short_title"] if c in d_diag.columns]
    d_keep = ["icd_code", "icd_version"] + title_cols
    diag = diag.merge(d_diag[d_keep], on=["icd_code", "icd_version"], how="left")
    diag = diag.merge(adm, on=["subject_id", "hadm_id"], how="left")
    if "long_title" in diag.columns and "short_title" in diag.columns:
        desc = diag["long_title"].fillna(diag["short_title"]).fillna("Unknown diagnosis")
    elif "long_title" in diag.columns:
        desc = diag["long_title"].fillna("Unknown diagnosis")
    elif "short_title" in diag.columns:
        desc = diag["short_title"].fillna("Unknown diagnosis")
    else:
        desc = pd.Series(["Unknown diagnosis"] * len(diag))
    out = pd.DataFrame({
        "patient_id": diag["subject_id"].astype("Int64"),
        "encounter_id": diag["hadm_id"].astype("Int64"),
        "source_dataset": "mimic",
        "event_time": diag["admittime"],
        "event_type": "diagnosis",
        "event_name": desc,
        "value_num": np.nan,
        "value_text": desc,
        "unit": "",
        "abnormal_flag": "",
        "payload_json": [
            _as_payload({"icd_code": str(c), "icd_version": int(v) if pd.notna(v) else None})
            for c, v in zip(diag["icd_code"], diag["icd_version"])
        ],
    })
    return _finalize(out)


def extract_mimic_labs(mimic_root: str, top_n_labs: int) -> pd.DataFrame:
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper"]
    labs = pd.read_csv(os.path.join(mimic_root, "labevents.csv.gz"), usecols=usecols)
    labs = labs[labs["hadm_id"].notna()]
    d_lab = pd.read_csv(os.path.join(mimic_root, "d_labitems.csv.gz"), usecols=["itemid", "label", "fluid", "category"])
    labs = labs.merge(d_lab, on="itemid", how="left")
    if top_n_labs > 0:
        top = labs["itemid"].value_counts().head(top_n_labs).index
        labs = labs[labs["itemid"].isin(top)]

    labs["event_time"] = _safe_to_datetime(labs["charttime"])
    labs["abnormal_flag"] = labs.apply(
        lambda r: _normalize_flag(r["valuenum"], r.get("ref_range_lower"), r.get("ref_range_upper")), axis=1
    )
    name = labs["label"].fillna("Unknown Lab")
    out = pd.DataFrame({
        "patient_id": labs["subject_id"].astype("Int64"),
        "encounter_id": labs["hadm_id"].astype("Int64"),
        "source_dataset": "mimic",
        "event_time": labs["event_time"],
        "event_type": "lab",
        "event_name": name,
        "value_num": labs["valuenum"],
        "value_text": name + "=" + labs["valuenum"].astype(str),
        "unit": labs["valueuom"].fillna(""),
        "abnormal_flag": labs["abnormal_flag"],
        "payload_json": [
            _as_payload(
                {
                    "itemid": int(i) if pd.notna(i) else None,
                    "fluid": f,
                    "category": c,
                    "ref_low": None if pd.isna(lo) else float(lo),
                    "ref_high": None if pd.isna(hi) else float(hi),
                }
            )
            for i, f, c, lo, hi in zip(
                labs["itemid"], labs.get("fluid"), labs.get("category"), labs["ref_range_lower"], labs["ref_range_upper"]
            )
        ],
    })
    return _finalize(out)


def extract_mimic_vitals(mimic_root: str, max_vitals_per_hour: int) -> pd.DataFrame:
    chart_file = os.path.join(mimic_root, "MIMIC-ICU", "icu", "chartevents.csv.gz")
    d_items_file = os.path.join(mimic_root, "MIMIC-ICU", "icu", "d_items.csv.gz")
    if not os.path.exists(chart_file) or not os.path.exists(d_items_file):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    itemids = [220045, 220179, 220180, 220052, 220210, 223761, 220277]
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    vit = pd.read_csv(chart_file, usecols=usecols, low_memory=False)
    vit = vit[vit["hadm_id"].notna() & vit["itemid"].isin(itemids)]
    d_items = pd.read_csv(d_items_file, usecols=["itemid", "label"])
    vit = vit.merge(d_items, on="itemid", how="left")
    vit["event_time"] = _safe_to_datetime(vit["charttime"])

    base = pd.DataFrame({
        "subject_id": vit["subject_id"],
        "hadm_id": vit["hadm_id"],
        "itemid": vit["itemid"],
        "event_name": vit["label"].fillna("Vital"),
        "event_time": vit["event_time"],
        "value_num": pd.to_numeric(vit["valuenum"], errors="coerce"),
    }).dropna(subset=["value_num"])
    agg = _aggregate_vitals_hourly(
        base,
        group_cols=["subject_id", "hadm_id", "itemid", "event_name"],
        max_points_per_hour=max_vitals_per_hour,
    )
    flag = []
    for n, v in zip(agg["event_name"], agg["value_num"]):
        lo, hi = _vital_range_for_name(n)
        flag.append(_normalize_flag(v, lo, hi))

    out = pd.DataFrame({
        "patient_id": agg["subject_id"].astype("Int64"),
        "encounter_id": agg["hadm_id"].astype("Int64"),
        "source_dataset": "mimic",
        "event_time": agg["event_time"],
        "event_type": "vital",
        "event_name": agg["event_name"],
        "value_num": agg["value_num"],
        "value_text": agg["event_name"].astype(str) + "=" + agg["value_num"].astype(str),
        "unit": "",
        "abnormal_flag": flag,
        "payload_json": [
            _as_payload(
                {
                    "itemid": int(i) if pd.notna(i) else None,
                    "agg_min": None if pd.isna(vmin) else float(vmin),
                    "agg_max": None if pd.isna(vmax) else float(vmax),
                    "agg_median": None if pd.isna(vmed) else float(vmed),
                    "n_obs": int(n),
                }
            )
            for i, vmin, vmax, vmed, n in zip(
                agg["itemid"], agg["value_num_min"], agg["value_num_max"], agg["value_num_median"], agg["n_obs"]
            )
        ],
    })
    return _finalize(out)


def extract_mimic_meds(mimic_root: str) -> pd.DataFrame:
    meds = pd.read_csv(os.path.join(mimic_root, "prescriptions.csv.gz"), low_memory=False)
    meds = meds[meds["hadm_id"].notna()]
    meds["starttime"] = _safe_to_datetime(meds.get("starttime"))
    meds["stoptime"] = _safe_to_datetime(meds.get("stoptime"))

    starts = pd.DataFrame({
        "patient_id": meds["subject_id"].astype("Int64"),
        "encounter_id": meds["hadm_id"].astype("Int64"),
        "source_dataset": "mimic",
        "event_time": meds["starttime"],
        "event_type": "medication",
        "event_name": meds["drug"].fillna("Unknown Med"),
        "value_num": pd.to_numeric(meds.get("dose_val_rx"), errors="coerce"),
        "value_text": meds["drug"].fillna("") + " start",
        "unit": meds.get("dose_unit_rx", "").fillna(""),
        "abnormal_flag": "",
        "payload_json": [
            _as_payload({"action": "start", "route": r})
            for r in meds.get("route", "").fillna("")
        ],
    })
    stops = starts.copy()
    stops["event_time"] = meds["stoptime"]
    stops["value_text"] = meds["drug"].fillna("") + " stop"
    stops["payload_json"] = [_as_payload({"action": "stop"}) for _ in range(len(stops))]
    return _finalize(pd.concat([starts, stops], ignore_index=True))


def extract_mimic_procs(mimic_root: str) -> pd.DataFrame:
    pro = pd.read_csv(os.path.join(mimic_root, "procedures_icd.csv.gz"), low_memory=False)
    d = pd.read_csv(os.path.join(mimic_root, "d_icd_procedures.csv.gz"), low_memory=False)
    title_cols = [c for c in ["long_title", "short_title"] if c in d.columns]
    d_keep = ["icd_code", "icd_version"] + title_cols
    pro = pro.merge(d[d_keep], on=["icd_code", "icd_version"], how="left")
    pro["chartdate"] = _safe_to_datetime(pro.get("chartdate"))
    if "long_title" in pro.columns and "short_title" in pro.columns:
        name = pro["long_title"].fillna(pro["short_title"]).fillna("Unknown Procedure")
    elif "long_title" in pro.columns:
        name = pro["long_title"].fillna("Unknown Procedure")
    elif "short_title" in pro.columns:
        name = pro["short_title"].fillna("Unknown Procedure")
    else:
        name = pd.Series(["Unknown Procedure"] * len(pro))
    out = pd.DataFrame({
        "patient_id": pro["subject_id"].astype("Int64"),
        "encounter_id": pro["hadm_id"].astype("Int64"),
        "source_dataset": "mimic",
        "event_time": pro["chartdate"],
        "event_type": "procedure",
        "event_name": name,
        "value_num": np.nan,
        "value_text": name,
        "unit": "",
        "abnormal_flag": "",
        "payload_json": [
            _as_payload({"icd_code": str(c), "icd_version": int(v) if pd.notna(v) else None})
            for c, v in zip(pro["icd_code"], pro["icd_version"])
        ],
    })
    return _finalize(out)


def extract_eicu_patient_labels(eicu_root: str) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), low_memory=False)
    p["unitdischargeoffset"] = pd.to_numeric(p.get("unitdischargeoffset"), errors="coerce")
    p["hospitaladmitoffset"] = pd.to_numeric(p.get("hospitaladmitoffset"), errors="coerce")
    p["hospitaldischargeoffset"] = pd.to_numeric(p.get("hospitaldischargeoffset"), errors="coerce")
    p["readmit_30d"] = np.nan
    mortality_cols = [c for c in p.columns if "dischargestatus" in c.lower()]
    mort = np.zeros(len(p), dtype=int)
    for c in mortality_cols:
        mort = np.where(p[c].astype(str).str.lower().str.contains("expired"), 1, mort)
    p["in_hospital_mortality"] = mort
    return p


def _eicu_event_time_from_offset(anchor: pd.Series, offset_min: pd.Series) -> pd.Series:
    offs = pd.to_numeric(offset_min, errors="coerce")
    return anchor + pd.to_timedelta(offs, unit="m")


def extract_eicu_labs(eicu_root: str) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
    p["anchor_time"] = pd.Timestamp("2000-01-01")
    lab = pd.read_csv(os.path.join(eicu_root, "lab.csv.gz"), low_memory=False)
    key = "patientunitstayid"
    lab = lab.merge(p[[key, "anchor_time"]], on=key, how="left")
    lab["event_time"] = _eicu_event_time_from_offset(lab["anchor_time"], lab.get("labresultoffset"))

    name_col = "labname" if "labname" in lab.columns else "labtypeid"
    val_col = "labresult" if "labresult" in lab.columns else None
    val = pd.to_numeric(lab[val_col], errors="coerce") if val_col else np.nan
    ref_low_col = "labresultrevisedoffset" if "labresultrevisedoffset" in lab.columns else None
    out = pd.DataFrame({
        "patient_id": lab[key].astype("Int64"),
        "encounter_id": lab[key].astype("Int64"),
        "source_dataset": "eicu",
        "event_time": lab["event_time"],
        "event_type": "lab",
        "event_name": lab[name_col].astype(str),
        "value_num": val,
        "value_text": lab[name_col].astype(str) + "=" + lab.get(val_col, "").astype(str),
        "unit": lab.get("labmeasurenamesystem", "").astype(str) if "labmeasurenamesystem" in lab.columns else "",
        "abnormal_flag": "unknown",
        "payload_json": [
            _as_payload(
                {
                    "source": "eicu_lab",
                    "raw_result": None if val_col is None else str(rv),
                    "ref_hint": None if ref_low_col is None else str(rh),
                }
            )
            for rv, rh in zip(lab.get(val_col, pd.Series([None] * len(lab))), lab.get(ref_low_col, pd.Series([None] * len(lab))))
        ],
    })
    out = _finalize(out)
    if not out.empty:
        out["abnormal_flag"] = _derive_eicu_lab_flags(out, min_group_n=200).fillna("unknown")
    return out


def extract_eicu_vitals(eicu_root: str, max_vitals_per_hour: int) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
    p["anchor_time"] = pd.Timestamp("2000-01-01")

    frames = []
    for fname, src in [("vitalPeriodic.csv.gz", "periodic"), ("vitalAperiodic.csv.gz", "aperiodic")]:
        fpath = os.path.join(eicu_root, fname)
        if not os.path.exists(fpath):
            continue
        if fname == "vitalPeriodic.csv.gz":
            keep_cols = [
                "patientunitstayid",
                "observationoffset",
                "temperature",
                "sao2",
                "heartrate",
                "respiration",
                "systemicsystolic",
                "systemicdiastolic",
                "systemicmean",
            ]
        else:
            keep_cols = [
                "patientunitstayid",
                "observationoffset",
                "noninvasivesystolic",
                "noninvasivediastolic",
                "noninvasivemean",
            ]
        v = pd.read_csv(fpath, usecols=lambda c: c in keep_cols, low_memory=False)
        if "patientunitstayid" not in v.columns:
            continue
        v = v.merge(p[["patientunitstayid", "anchor_time"]], on="patientunitstayid", how="left")
        off_col = "observationoffset" if "observationoffset" in v.columns else None
        if not off_col:
            continue
        v["event_time"] = _eicu_event_time_from_offset(v["anchor_time"], v[off_col])

        value_candidates = [c for c in v.columns if c not in {"patientunitstayid", "anchor_time", off_col, "event_time"}]
        id_vars = ["patientunitstayid", "event_time"]
        vm = v[id_vars + value_candidates].melt(id_vars=id_vars, var_name="event_name", value_name="value_num")
        vm["value_num"] = pd.to_numeric(vm["value_num"], errors="coerce")
        vm = vm[vm["value_num"].notna()]
        vm["source_kind"] = src
        frames.append(vm)

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    vit = pd.concat(frames, ignore_index=True)
    vit = _aggregate_vitals_hourly(
        vit.rename(columns={"patientunitstayid": "pid"}),
        group_cols=["pid", "event_name"],
        max_points_per_hour=max_vitals_per_hour,
    )
    flag = []
    for n, v in zip(vit["event_name"], vit["value_num"]):
        lo, hi = _vital_range_for_name(n)
        flag.append(_normalize_flag(v, lo, hi))

    out = pd.DataFrame({
        "patient_id": vit["pid"].astype("Int64"),
        "encounter_id": vit["pid"].astype("Int64"),
        "source_dataset": "eicu",
        "event_time": vit["event_time"],
        "event_type": "vital",
        "event_name": vit["event_name"].astype(str),
        "value_num": vit["value_num"],
        "value_text": vit["event_name"].astype(str) + "=" + vit["value_num"].astype(str),
        "unit": "",
        "abnormal_flag": flag,
        "payload_json": [
            _as_payload(
                {
                    "source": "eicu_vital",
                    "agg_min": None if pd.isna(vmin) else float(vmin),
                    "agg_max": None if pd.isna(vmax) else float(vmax),
                    "agg_median": None if pd.isna(vmed) else float(vmed),
                    "n_obs": int(n),
                }
            )
            for vmin, vmax, vmed, n in zip(vit["value_num_min"], vit["value_num_max"], vit["value_num_median"], vit["n_obs"])
        ],
    })
    return _finalize(out)


def extract_eicu_meds(eicu_root: str) -> pd.DataFrame:
    events = []
    for fname, etype in [("medication.csv.gz", "medication"), ("infusionDrug.csv.gz", "medication")]:
        fpath = os.path.join(eicu_root, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, low_memory=False)
        if "patientunitstayid" not in df.columns:
            continue
        p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
        p["anchor_time"] = pd.Timestamp("2000-01-01")
        df = df.merge(p, on="patientunitstayid", how="left")
        offset_col = None
        for c in ["drugstartoffset", "infusionoffset", "intakeoutputoffset", "treatmentoffset"]:
            if c in df.columns:
                offset_col = c
                break
        if not offset_col:
            continue
        name_col = "drugname" if "drugname" in df.columns else ("drug" if "drug" in df.columns else None)
        if not name_col:
            continue
        df["event_time"] = _eicu_event_time_from_offset(df["anchor_time"], df[offset_col])
        ev = pd.DataFrame({
            "patient_id": df["patientunitstayid"].astype("Int64"),
            "encounter_id": df["patientunitstayid"].astype("Int64"),
            "source_dataset": "eicu",
            "event_time": df["event_time"],
            "event_type": etype,
            "event_name": df[name_col].astype(str),
            "value_num": np.nan,
            "value_text": df[name_col].astype(str),
            "unit": "",
            "abnormal_flag": "",
            "payload_json": [_as_payload({"source_table": fname}) for _ in range(len(df))],
        })
        events.append(ev)
    if not events:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    out = _finalize(pd.concat(events, ignore_index=True))
    # Add treatment-derived intervention signals that often include ventilation/dialysis keywords.
    tpath = os.path.join(eicu_root, "treatment.csv.gz")
    if os.path.exists(tpath):
        t = pd.read_csv(tpath, low_memory=False)
        if "patientunitstayid" in t.columns:
            p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
            p["anchor_time"] = pd.Timestamp("2000-01-01")
            t = t.merge(p, on="patientunitstayid", how="left")
            off_col = "treatmentoffset" if "treatmentoffset" in t.columns else None
            name_col = "treatmentstring" if "treatmentstring" in t.columns else None
            if off_col and name_col:
                t["event_time"] = _eicu_event_time_from_offset(t["anchor_time"], t[off_col])
                add = pd.DataFrame({
                    "patient_id": t["patientunitstayid"].astype("Int64"),
                    "encounter_id": t["patientunitstayid"].astype("Int64"),
                    "source_dataset": "eicu",
                    "event_time": t["event_time"],
                    "event_type": "medication",
                    "event_name": t[name_col].astype(str),
                    "value_num": np.nan,
                    "value_text": t[name_col].astype(str),
                    "unit": "",
                    "abnormal_flag": "",
                    "payload_json": [_as_payload({"source_table": "treatment.csv.gz", "derived": True}) for _ in range(len(t))],
                })
                out = _finalize(pd.concat([out, add], ignore_index=True))
    return out


def extract_eicu_procs(eicu_root: str) -> pd.DataFrame:
    fpath = os.path.join(eicu_root, "treatment.csv.gz")
    if not os.path.exists(fpath):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(fpath, low_memory=False)
    if "patientunitstayid" not in df.columns:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
    p["anchor_time"] = pd.Timestamp("2000-01-01")
    df = df.merge(p, on="patientunitstayid", how="left")
    off_col = "treatmentoffset" if "treatmentoffset" in df.columns else None
    if not off_col:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    name_col = "treatmentstring" if "treatmentstring" in df.columns else "treatmentid"
    df["event_time"] = _eicu_event_time_from_offset(df["anchor_time"], df[off_col])
    out = pd.DataFrame({
        "patient_id": df["patientunitstayid"].astype("Int64"),
        "encounter_id": df["patientunitstayid"].astype("Int64"),
        "source_dataset": "eicu",
        "event_time": df["event_time"],
        "event_type": "procedure",
        "event_name": df[name_col].astype(str),
        "value_num": np.nan,
        "value_text": df[name_col].astype(str),
        "unit": "",
        "abnormal_flag": "",
        "payload_json": [_as_payload({"source": "eicu_treatment"}) for _ in range(len(df))],
    })
    return _finalize(out)


def extract_eicu_diagnoses(eicu_root: str, include: bool) -> pd.DataFrame:
    if not include:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    fpath = os.path.join(eicu_root, "diagnosis.csv.gz")
    if not os.path.exists(fpath):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(fpath, low_memory=False)
    if "patientunitstayid" not in df.columns:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    p = pd.read_csv(os.path.join(eicu_root, "patient.csv.gz"), usecols=["patientunitstayid"], low_memory=False)
    p["anchor_time"] = pd.Timestamp("2000-01-01")
    df = df.merge(p, on="patientunitstayid", how="left")
    col = "diagnosisstring" if "diagnosisstring" in df.columns else ("diagnosispriority" if "diagnosispriority" in df.columns else None)
    if not col:
        col = df.columns[0]
    out = pd.DataFrame({
        "patient_id": df["patientunitstayid"].astype("Int64"),
        "encounter_id": df["patientunitstayid"].astype("Int64"),
        "source_dataset": "eicu",
        "event_time": df["anchor_time"],
        "event_type": "diagnosis",
        "event_name": df[col].astype(str),
        "value_num": np.nan,
        "value_text": df[col].astype(str),
        "unit": "",
        "abnormal_flag": "",
        "payload_json": [_as_payload({"source": "eicu_diagnosis"}) for _ in range(len(df))],
    })
    return _finalize(out)


def _compile_label_patterns(mode: str) -> Dict[str, re.Pattern]:
    m = str(mode or "legacy").strip().lower()
    if m not in LABEL_PATTERNS:
        m = "legacy"
    return LABEL_PATTERNS[m]


def _series_contains(s: pd.Series, pattern) -> pd.Series:
    # Suppress noisy pandas regex-group warnings locally; keep all other warnings visible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This pattern is interpreted as a regular expression, and has match groups.*",
            category=UserWarning,
        )
        return s.str.contains(pattern, na=False)


def _load_include_patient_pairs(path: Optional[str]) -> List[Tuple[str, int, Optional[int]]]:
    if not path or not os.path.exists(path):
        return []
    pairs: List[Tuple[str, int, Optional[int]]] = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [x.strip() for x in re.split(r"[,\t]", ln) if x.strip()]
            if len(parts) == 1:
                try:
                    pairs.append(("any", int(parts[0]), None))
                except Exception:
                    continue
            else:
                try:
                    enc = int(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
                    pairs.append((str(parts[0]).lower(), int(parts[1]), enc))
                except Exception:
                    continue
    return pairs


def _cap_patients_per_source(
    events: pd.DataFrame,
    max_patients_per_source: int,
    include_pairs: List[Tuple[str, int, Optional[int]]],
    seed: int,
    logger=None,
) -> pd.DataFrame:
    if events.empty or max_patients_per_source <= 0:
        return events

    rng = np.random.default_rng(seed)
    include_any = set()
    include_by_src: Dict[str, set] = {}
    for src, pid, _enc in include_pairs:
        if src == "any":
            include_any.add(pid)
        else:
            include_by_src.setdefault(src, set()).add(pid)

    out = []
    for src, g in events.groupby("source_dataset", sort=False):
        src_l = str(src).lower()
        uniq = g["patient_id"].dropna().astype("Int64").astype(int).unique().tolist()
        uniq_set = set(uniq)
        req = [pid for pid in (include_any | include_by_src.get(src_l, set())) if pid in uniq_set]
        if len(uniq) <= max_patients_per_source:
            keep = set(uniq)
        else:
            if len(req) > max_patients_per_source:
                if logger is not None:
                    logger.log(
                        f"[WARN] include list for source={src} has {len(req)} patients > cap={max_patients_per_source}; truncating include list."
                    )
                req = sorted(req)[:max_patients_per_source]
            rem = [pid for pid in uniq if pid not in set(req)]
            n_extra = max(0, max_patients_per_source - len(req))
            extra = rng.choice(rem, size=n_extra, replace=False).tolist() if n_extra > 0 and rem else []
            keep = set(req) | set(extra)
        gg = g[g["patient_id"].astype("Int64").astype(int).isin(list(keep))]
        out.append(gg)
        if logger is not None:
            logger.log(f"Patient cap source={src}: kept {len(keep)}/{len(uniq)} patients, rows={len(gg)}")

    return _finalize(pd.concat(out, ignore_index=True) if out else events.iloc[:0].copy())


def _is_deterioration_in_future(fut: pd.DataFrame, label_patterns: Dict[str, re.Pattern]) -> Dict:
    if fut.empty:
        return {}
    names = fut["event_name"].astype(str).str.lower()
    flags = fut["abnormal_flag"].astype(str).str.lower()
    meds = fut[fut["event_type"] == "medication"]["event_name"].astype(str).str.lower()
    procs = fut[fut["event_type"] == "procedure"]["event_name"].astype(str).str.lower()
    proc_any_pat = re.compile(label_patterns["resp"].pattern + "|" + label_patterns["renal"].pattern + "|cpr", re.IGNORECASE)
    return {
        "any_deterioration": int(
            ((flags == "high") | (flags == "low")).any()
            or _series_contains(meds, label_patterns["vaso"]).any()
            or _series_contains(procs, proc_any_pat).any()
            or _series_contains(names, "lactate").any() and (flags == "high").any()
        ),
        "abnormal_lab_or_vital": int(((fut["event_type"].isin(["lab", "vital"])) & (flags.isin(["high", "low"]))).any()),
        "vasopressor_signal": int(_series_contains(meds, label_patterns["vaso"]).any()),
        "resp_support_signal": int(_series_contains(procs, label_patterns["resp"]).any()),
        "renal_support_signal": int(_series_contains(procs, label_patterns["renal"]).any()),
    }


def build_horizon_examples(
    events: pd.DataFrame,
    horizon_hours: int,
    max_context_events: int,
    require_protocol_observations: bool = False,
    label_patterns: Optional[Dict[str, re.Pattern]] = None,
) -> pd.DataFrame:
    label_patterns = label_patterns or _compile_label_patterns("legacy")
    rows = []
    events = events.sort_values(["source_dataset", "patient_id", "encounter_id", "event_time"])
    grp_cols = ["source_dataset", "patient_id", "encounter_id"]
    for key, g in events.groupby(grp_cols):
        g = g.reset_index(drop=True)
        times = pd.to_datetime(g["event_time"])
        if len(times) > 0:
            span_h = (times.max() - times.min()).total_seconds() / 3600.0
            # Drop obviously corrupted trajectories with extreme mixed-era jumps.
            if span_h > (24 * 60):
                continue
        for i in range(len(g)):
            t = times.iloc[i]
            t_end = t + pd.Timedelta(hours=horizon_hours)
            ctx = g.iloc[max(0, i - max_context_events + 1): i + 1]
            fut = g[(times > t) & (times <= t_end)]
            if fut.empty:
                continue
            symbolic_facts = _extract_symbolic_facts(ctx, anchor_time=t, lookback_hours=max(12, horizon_hours))
            if require_protocol_observations and len(symbolic_facts) == 0:
                continue
            latent_state = _build_latent_state(symbolic_facts, ctx)
            cf_candidates = _counterfactual_candidates(latent_state, symbolic_facts)
            current_flags = ctx["abnormal_flag"].astype(str).str.lower()
            current_state = {
                "ctx_event_count": int(len(ctx)),
                "ctx_abnormal_count": int(current_flags.isin(["high", "low"]).sum()),
                "ctx_event_type_counts": ctx["event_type"].value_counts().to_dict(),
            }
            targets = _is_deterioration_in_future(fut, label_patterns=label_patterns)
            step_id = int(i)
            example_id = f"{key[0]}_{key[1]}_{key[2]}_{step_id}_h{horizon_hours}"
            rows.append({
                "example_id": example_id,
                "source_dataset": key[0],
                "patient_id": int(key[1]),
                "encounter_id": int(key[2]),
                "anchor_time": str(t),
                "horizon_hours": horizon_hours,
                "step_id": step_id,
                "step_index": step_id,
                "context_text": "\n".join(ctx["value_text"].fillna(ctx["event_name"]).astype(str).tolist()),
                "protocol_observations": symbolic_facts,
                "latent_state_current": latent_state,
                "symbolic_state_current": current_state,
                "individual_protocol_trace": {
                    "anchor_time": str(t),
                    "active_abnormal_signals": [f for f in symbolic_facts if str(f.get("abnormal_flag_last", "")).lower() in {"high", "low"}],
                },
                "counterfactual_candidates": cf_candidates,
                "future_event_types": fut["event_type"].value_counts().to_dict(),
                "future_event_names": fut["event_name"].astype(str).head(20).tolist(),
                "targets": targets,
            })
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame, labels_mimic: pd.DataFrame, labels_eicu: pd.DataFrame) -> Dict:
    out = {
        "total_rows": int(len(events)),
        "by_source": events["source_dataset"].value_counts().to_dict() if not events.empty else {},
        "by_event_type": events["event_type"].value_counts().to_dict() if not events.empty else {},
        "unique_patients": int(events["patient_id"].nunique()) if not events.empty else 0,
        "unique_encounters": int(events["encounter_id"].nunique()) if not events.empty else 0,
        "time_min": str(events["event_time"].min()) if not events.empty else None,
        "time_max": str(events["event_time"].max()) if not events.empty else None,
    }
    if not labels_mimic.empty:
        out["mimic_mortality_rate"] = float(labels_mimic["in_hospital_mortality"].mean())
        out["mimic_readmit_30d_rate"] = float(labels_mimic["readmit_30d"].mean())
    if not labels_eicu.empty and "in_hospital_mortality" in labels_eicu:
        out["eicu_mortality_rate"] = float(pd.to_numeric(labels_eicu["in_hospital_mortality"], errors="coerce").mean())
    if not events.empty:
        out["abnormal_flag_counts"] = events["abnormal_flag"].fillna("").astype(str).value_counts().to_dict()
    return out


def build_protocol_feature_table(events: pd.DataFrame, core_only: bool = True) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "patient_id",
                "encounter_id",
                "event_time",
                "feature",
                "event_type",
                "value_num",
                "abnormal_flag",
                "trend_tag",
            ]
        )
    e = events.copy().sort_values(["source_dataset", "patient_id", "encounter_id", "event_name", "event_time"])
    e["feature"] = e["event_name"].astype(str).map(_canonical_feature_name)
    if core_only:
        e = e[e["feature"].isin(CORE_PROTOCOL_FEATURES)]
    e["value_num"] = pd.to_numeric(e["value_num"], errors="coerce")
    prev = e.groupby(["source_dataset", "patient_id", "encounter_id", "feature"])["value_num"].shift(1)
    delta = e["value_num"] - prev
    trend = np.where(delta > 0, "rising", np.where(delta < 0, "falling", "stable"))
    trend = np.where(prev.isna() | e["value_num"].isna(), "unknown", trend)
    return pd.DataFrame(
        {
            "source_dataset": e["source_dataset"],
            "patient_id": e["patient_id"],
            "encounter_id": e["encounter_id"],
            "event_time": e["event_time"],
            "feature": e["feature"],
            "event_type": e["event_type"],
            "value_num": e["value_num"],
            "abnormal_flag": e["abnormal_flag"].astype(str),
            "trend_tag": trend,
        }
    )


def build_qc_report(events: pd.DataFrame, protocol_features: pd.DataFrame, horizon_outputs: Dict) -> Dict:
    qc = {}
    qc["event_type_counts"] = events["event_type"].value_counts().to_dict() if not events.empty else {}
    if not protocol_features.empty:
        qc["protocol_feature_counts_top30"] = protocol_features["feature"].value_counts().head(30).to_dict()
        core_cov = {}
        total = len(protocol_features)
        for f in sorted(CORE_PROTOCOL_FEATURES):
            core_cov[f] = float((protocol_features["feature"] == f).mean()) if total else 0.0
        qc["core_feature_coverage_rate"] = core_cov
        qc["abnormal_rate_by_feature_top20"] = (
            protocol_features.assign(is_ab=protocol_features["abnormal_flag"].isin(["high", "low"]).astype(int))
            .groupby("feature")["is_ab"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .round(4)
            .to_dict()
        )
    label_prev = {}
    cf_rates = {}
    for h, meta in horizon_outputs.items():
        p = meta["path"]
        n = 0
        sums = {"any_deterioration": 0, "abnormal_lab_or_vital": 0, "vasopressor_signal": 0, "resp_support_signal": 0, "renal_support_signal": 0}
        cf_nonempty = 0
        with open(p, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = obj.get("targets", {})
                for k in sums:
                    sums[k] += int(t.get(k, 0))
                if obj.get("counterfactual_candidates"):
                    cf_nonempty += 1
                n += 1
        if n > 0:
            label_prev[h] = {k: round(v / n, 6) for k, v in sums.items()}
            cf_rates[h] = round(cf_nonempty / n, 6)
    qc["label_prevalence_by_horizon"] = label_prev
    qc["counterfactual_nonempty_rate_by_horizon"] = cf_rates
    # Lightweight gates to quickly decide if a run is usable for staged latent-agent inference.
    gates = {}
    if not protocol_features.empty:
        cov = qc.get("core_feature_coverage_rate", {})
        expected_features = ["hr", "rr", "spo2", "map", "sbp", "dbp", "creatinine", "lactate", "wbc", "potassium", "sodium", "bicarbonate", "glucose"]
        for f in expected_features:
            gates[f"coverage_{f}_ge_0.001"] = bool(cov.get(f, 0.0) >= 0.001)
    for h, r in cf_rates.items():
        gates[f"cf_nonempty_h{h}_ge_0.01"] = bool(r >= 0.01)
    qc["qc_gates"] = gates
    qc["qc_pass"] = bool(gates) and all(gates.values())
    return qc


def build_typed_time_aware_steps(
    events: pd.DataFrame,
    same_type_max_gap_hours: int = 2,
    force_new_step_gap_hours: int = 12,
    max_events_per_step: int = 20,
) -> List[Dict]:
    if events.empty:
        return []
    rows = []
    gcols = ["source_dataset", "patient_id", "encounter_id"]
    events = events.sort_values(gcols + ["event_time"])
    for (src, pid, eid), g in events.groupby(gcols):
        g = g.reset_index(drop=True)
        step_id = 0
        cur = []
        cur_type = None
        last_t = None
        for _, r in g.iterrows():
            et = str(r["event_type"])
            t = pd.to_datetime(r["event_time"])
            if not cur:
                cur = [r]
                cur_type = et
                last_t = t
                continue
            gap_h = (t - last_t).total_seconds() / 3600.0 if last_t is not None else 0.0
            need_new = (
                et != cur_type
                or gap_h > same_type_max_gap_hours
                or gap_h > force_new_step_gap_hours
                or len(cur) >= max_events_per_step
            )
            if need_new:
                step_rows = pd.DataFrame(cur)
                rows.append({
                    "source_dataset": src,
                    "patient_id": int(pid),
                    "encounter_id": int(eid),
                    "step_id": step_id,
                    "step_type": cur_type,
                    "t_start": str(pd.to_datetime(step_rows["event_time"]).min()),
                    "t_end": str(pd.to_datetime(step_rows["event_time"]).max()),
                    "n_events": int(len(step_rows)),
                    "entities": step_rows[["event_name", "value_num", "abnormal_flag", "unit", "event_time"]].to_dict("records"),
                })
                step_id += 1
                cur = [r]
                cur_type = et
            else:
                cur.append(r)
            last_t = t
        if cur:
            step_rows = pd.DataFrame(cur)
            rows.append({
                "source_dataset": src,
                "patient_id": int(pid),
                "encounter_id": int(eid),
                "step_id": step_id,
                "step_type": cur_type,
                "t_start": str(pd.to_datetime(step_rows["event_time"]).min()),
                "t_end": str(pd.to_datetime(step_rows["event_time"]).max()),
                "n_events": int(len(step_rows)),
                "entities": step_rows[["event_name", "value_num", "abnormal_flag", "unit", "event_time"]].to_dict("records"),
            })
    return rows


def build_typed_binned_steps(
    events: pd.DataFrame,
    bin_hours: int = 6,
    max_events_per_step: int = 200,
    max_steps_per_trajectory: int = 0,
) -> List[Dict]:
    """Build typed steps by fixed-width time bins.

    This avoids over-fragmentation from type-switching and produces more stable
    temporal cadence for longitudinal modeling.
    """
    if events.empty:
        return []
    if bin_hours <= 0:
        bin_hours = 6

    rows = []
    gcols = ["source_dataset", "patient_id", "encounter_id"]
    events = events.sort_values(gcols + ["event_time"])
    for (src, pid, eid), g in events.groupby(gcols):
        g = g.reset_index(drop=True).copy()
        g["event_time"] = pd.to_datetime(g["event_time"], errors="coerce")
        g = g[g["event_time"].notna()]
        if g.empty:
            continue

        t0 = g["event_time"].min()
        step_idx = np.floor((g["event_time"] - t0).dt.total_seconds() / 3600.0 / float(bin_hours)).astype(int)
        g["step_idx"] = step_idx

        by_step = list(g.groupby("step_idx", sort=True))
        if max_steps_per_trajectory and max_steps_per_trajectory > 0:
            by_step = by_step[:max_steps_per_trajectory]

        for step_id, (_, sg) in enumerate(by_step):
            sg = sg.sort_values("event_time")
            if max_events_per_step and max_events_per_step > 0 and len(sg) > max_events_per_step:
                # Keep latest events in bin to preserve recency under high event density.
                sg = sg.tail(max_events_per_step)
            type_counts = sg["event_type"].astype(str).value_counts().to_dict()
            dominant_type = max(type_counts.items(), key=lambda kv: kv[1])[0] if type_counts else "mixed"
            rows.append(
                {
                    "source_dataset": src,
                    "patient_id": int(pid),
                    "encounter_id": int(eid),
                    "step_id": int(step_id),
                    "step_type": str(dominant_type),
                    "t_start": str(pd.to_datetime(sg["event_time"]).min()),
                    "t_end": str(pd.to_datetime(sg["event_time"]).max()),
                    "n_events": int(len(sg)),
                    "step_type_counts": type_counts,
                    "entities": sg[["event_name", "value_num", "abnormal_flag", "unit", "event_time", "event_type"]].to_dict("records"),
                }
            )
    return rows


def build_horizon_examples_from_typed_steps(
    steps: List[Dict],
    horizon_hours: int,
    max_context_steps: int = 16,
    require_protocol_observations: bool = False,
    label_patterns: Optional[Dict[str, re.Pattern]] = None,
) -> List[Dict]:
    label_patterns = label_patterns or _compile_label_patterns("legacy")
    if not steps:
        return []
    by_traj = {}
    for s in steps:
        key = (s["source_dataset"], s["patient_id"], s["encounter_id"])
        by_traj.setdefault(key, []).append(s)
    out = []
    for key, seq in by_traj.items():
        seq = sorted(seq, key=lambda x: (x["t_start"], x["step_id"]))
        t_starts = [pd.to_datetime(x["t_start"]) for x in seq]
        if t_starts:
            span_h = (max(t_starts) - min(t_starts)).total_seconds() / 3600.0
            if span_h > (24 * 60):
                continue
        for i, s in enumerate(seq):
            t = t_starts[i]
            t_end = t + pd.Timedelta(hours=horizon_hours)
            ctx = seq[max(0, i - max_context_steps + 1): i + 1]
            fut = [x for j, x in enumerate(seq) if t_starts[j] > t and t_starts[j] <= t_end]
            if not fut:
                continue
            # Build compact protocol observations from current step entities only.
            facts = []
            for e in s["entities"]:
                f = _canonical_feature_name(e.get("event_name", ""))
                if f in CORE_PROTOCOL_FEATURES:
                    facts.append({
                        "feature": f,
                        "event_name": e.get("event_name"),
                        "value_last": e.get("value_num"),
                        "abnormal_flag_last": e.get("abnormal_flag", ""),
                        "trend": "unknown",
                    })
            if require_protocol_observations and len(facts) == 0:
                continue
            latent = _build_latent_state(facts, pd.DataFrame({"event_name": [e.get("event_name", "") for e in s["entities"]]}))
            med_names = []
            proc_names = []
            for fs in fut:
                if fs["step_type"] == "medication":
                    med_names.extend([str(e.get("event_name", "")).lower() for e in fs["entities"]])
                elif fs["step_type"] == "procedure":
                    proc_names.extend([str(e.get("event_name", "")).lower() for e in fs["entities"]])
            med_s = pd.Series(med_names, dtype="object") if med_names else pd.Series([], dtype="object")
            proc_s = pd.Series(proc_names, dtype="object") if proc_names else pd.Series([], dtype="object")
            targets = {
                "any_deterioration": int(any(any(str(e.get("abnormal_flag", "")).lower() in {"high", "low"} for e in fs["entities"]) for fs in fut)),
                "abnormal_lab_or_vital": int(any(fs["step_type"] in {"lab", "vital"} and any(str(e.get("abnormal_flag", "")).lower() in {"high", "low"} for e in fs["entities"]) for fs in fut)),
                "vasopressor_signal": int(_series_contains(med_s, label_patterns["vaso"]).any()) if not med_s.empty else 0,
                "resp_support_signal": int(_series_contains(proc_s, label_patterns["resp"]).any()) if not proc_s.empty else 0,
                "renal_support_signal": int(_series_contains(proc_s, label_patterns["renal"]).any()) if not proc_s.empty else 0,
            }
            example_id = f"{key[0]}_{key[1]}_{key[2]}_{s['step_id']}_h{horizon_hours}"
            out.append({
                "example_id": example_id,
                "source_dataset": key[0],
                "patient_id": key[1],
                "encounter_id": key[2],
                "anchor_time": str(t),
                "horizon_hours": horizon_hours,
                "step_type": s["step_type"],
                "step_id": s["step_id"],
                "step_index": s["step_id"],
                # Compatibility with downstream readers that expect entity payload at top level.
                "entities": s["entities"],
                "step_entities": s["entities"],
                "context_steps": [{"step_id": x["step_id"], "step_type": x["step_type"], "n_events": x["n_events"], "t_start": x["t_start"], "t_end": x["t_end"]} for x in ctx],
                "protocol_observations": facts,
                "latent_state_current": latent,
                "counterfactual_candidates": _counterfactual_candidates(latent, facts),
                "targets": targets,
            })
    return out


def build_global_protocol_seed_v1() -> Dict:
    return {
        "version": "v1",
        "description": "Manually curated universal protocol seed for MIMIC-IV + eICU deterioration monitoring.",
        "rules": [
            {
                "rule_id": "HEMO_SHOCK_001",
                "trigger": [
                    {"feature": "map", "op": "<", "value": 65, "window_hours": 1},
                    {"feature": "lactate", "op": "trend_is", "value": "rising", "window_hours": 6},
                ],
                "state_update": {"hemodynamic_instability_score": "+1"},
                "risk": "shock_progression",
                "severity": "high",
                "counterfactual_candidates": [
                    "start_or_titrate_vasopressor",
                    "fluid_resuscitation_bolus",
                ],
            },
            {
                "rule_id": "RESP_FAIL_001",
                "trigger": [
                    {"feature": "spo2", "op": "<", "value": 90, "window_hours": 1},
                    {"feature": "rr", "op": ">", "value": 28, "window_hours": 1},
                ],
                "state_update": {"respiratory_instability_score": "+1"},
                "risk": "respiratory_failure",
                "severity": "high",
                "counterfactual_candidates": [
                    "escalate_oxygen_support",
                    "consider_ventilation_pathway",
                ],
            },
            {
                "rule_id": "RENAL_AKI_001",
                "trigger": [
                    {"feature": "creatinine", "op": "trend_is", "value": "rising", "window_hours": 12},
                ],
                "state_update": {"renal_instability_score": "+1"},
                "risk": "aki_progression",
                "severity": "medium",
                "counterfactual_candidates": [
                    "renal_dose_adjustment_and_fluid_review",
                ],
            },
            {
                "rule_id": "SEPSIS_RISK_001",
                "trigger": [
                    {"feature": "temp", "op": ">", "value": 38.0, "window_hours": 6},
                    {"feature": "wbc", "op": "abnormal", "value": True, "window_hours": 6},
                    {"feature": "lactate", "op": "abnormal", "value": True, "window_hours": 6},
                ],
                "state_update": {"systemic_inflammation_score": "+1"},
                "risk": "sepsis_progression",
                "severity": "high",
                "counterfactual_candidates": [
                    "infection_workup_and_early_antimicrobials",
                    "fluid_resuscitation_bolus",
                ],
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", default=MIMIC_ROOT_DEFAULT)
    parser.add_argument("--eicu-root", default=EICU_ROOT_DEFAULT)
    parser.add_argument("--results-root", default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-mimic", action="store_true")
    parser.add_argument("--include-eicu", action="store_true")
    parser.add_argument("--include-diagnoses", action="store_true")
    parser.add_argument("--top-n-labs", type=int, default=150)
    parser.add_argument("--max-vitals-per-hour", type=int, default=1)
    parser.add_argument("--lab-normal-keep-hours", type=int, default=6)
    parser.add_argument("--lab-max-normal-per-feature-per-encounter", type=int, default=24)
    parser.add_argument("--intervention-min-gap-minutes", type=int, default=60)
    parser.add_argument("--max-context-events", type=int, default=64)
    parser.add_argument("--emit-typed-steps", action="store_true")
    parser.add_argument("--same-type-max-gap-hours", type=int, default=2)
    parser.add_argument("--force-new-step-gap-hours", type=int, default=12)
    parser.add_argument("--max-events-per-step", type=int, default=20)
    parser.add_argument("--typed-step-mode", default="type_aware", choices=["type_aware", "binned"])
    parser.add_argument("--typed-bin-hours", type=int, default=6)
    parser.add_argument("--typed-max-steps-per-trajectory", type=int, default=0)
    parser.add_argument("--max-context-steps", type=int, default=16)
    parser.add_argument("--drop-empty-protocol-steps", action="store_true")
    parser.add_argument("--horizons", default="6,12")
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-patients-per-source", type=int, default=0)
    parser.add_argument("--include-patient-ids-file", default="")
    parser.add_argument("--patient-sample-seed", type=int, default=42)
    parser.add_argument("--labeling-mode", default="legacy", choices=["legacy", "expanded"])
    args = parser.parse_args()

    if not args.include_mimic and not args.include_eicu:
        args.include_mimic = True
        args.include_eicu = True

    paths = prepare_run_dirs(args.results_root, args.run_id)
    set_cache_env(paths.cache_dir)
    logger = RunLogger(os.path.join(paths.logs_dir, "preprocess.log"))

    logger.log(f"Run ID: {paths.run_id}")
    logger.log(f"Run root: {paths.run_root}")

    all_events = []
    mimic_labels = pd.DataFrame()
    eicu_labels = pd.DataFrame()

    if args.include_mimic:
        logger.log("Extracting MIMIC events...")
        mimic_labels = mimic_admissions(args.mimic_root)
        parts = [
            extract_mimic_labs(args.mimic_root, args.top_n_labs),
            extract_mimic_vitals(args.mimic_root, args.max_vitals_per_hour),
            extract_mimic_meds(args.mimic_root),
            extract_mimic_procs(args.mimic_root),
        ]
        if args.include_diagnoses:
            parts.append(extract_mimic_diagnoses(args.mimic_root, include=True))
        mimic_events = _finalize(pd.concat(parts, ignore_index=True))
        if args.max_rows_per_source and len(mimic_events) > args.max_rows_per_source:
            mimic_events = mimic_events.sample(args.max_rows_per_source, random_state=42)
        all_events.append(mimic_events)
        _sample_and_write(mimic_events, os.path.join(paths.samples_dir, "mimic_events_sample.json"))
        logger.log(f"MIMIC events rows: {len(mimic_events)}")

    if args.include_eicu:
        logger.log("Extracting eICU events...")
        eicu_labels = extract_eicu_patient_labels(args.eicu_root)
        parts = [
            extract_eicu_labs(args.eicu_root),
            extract_eicu_vitals(args.eicu_root, args.max_vitals_per_hour),
            extract_eicu_meds(args.eicu_root),
            extract_eicu_procs(args.eicu_root),
        ]
        if args.include_diagnoses:
            parts.append(extract_eicu_diagnoses(args.eicu_root, include=True))
        eicu_events = _finalize(pd.concat(parts, ignore_index=True))
        if args.max_rows_per_source and len(eicu_events) > args.max_rows_per_source:
            eicu_events = eicu_events.sample(args.max_rows_per_source, random_state=42)
        all_events.append(eicu_events)
        _sample_and_write(eicu_events, os.path.join(paths.samples_dir, "eicu_events_sample.json"))
        logger.log(f"eICU events rows: {len(eicu_events)}")

    events = _finalize(pd.concat(all_events, ignore_index=True)) if all_events else pd.DataFrame(columns=CANONICAL_COLUMNS)
    events = events.sort_values(["source_dataset", "patient_id", "encounter_id", "event_time"])
    include_pairs = _load_include_patient_pairs(args.include_patient_ids_file)
    if args.max_patients_per_source > 0:
        events = _cap_patients_per_source(
            events,
            max_patients_per_source=args.max_patients_per_source,
            include_pairs=include_pairs,
            seed=args.patient_sample_seed,
            logger=logger,
        )
        sel = (
            events[["source_dataset", "patient_id"]]
            .drop_duplicates()
            .sort_values(["source_dataset", "patient_id"])
            .to_dict(orient="records")
        )
        with open(os.path.join(paths.artifacts_dir, "selected_patients.json"), "w") as f:
            json.dump(sel, f, indent=2)
        logger.log(f"Saved selected patients: n={len(sel)} path={os.path.join(paths.artifacts_dir, 'selected_patients.json')}")
        if include_pairs:
            required = []
            ev = events[["source_dataset", "patient_id", "encounter_id"]].dropna()
            ev["source_dataset"] = ev["source_dataset"].astype(str).str.lower()
            ev["patient_id"] = ev["patient_id"].astype("Int64").astype(int)
            ev["encounter_id"] = ev["encounter_id"].astype("Int64").astype(int)
            for src, pid, enc in include_pairs:
                src_match = ev if src == "any" else ev[ev["source_dataset"] == src]
                if enc is None:
                    ok = bool((src_match["patient_id"] == int(pid)).any())
                else:
                    ok = bool(((src_match["patient_id"] == int(pid)) & (src_match["encounter_id"] == int(enc))).any())
                required.append({"source_dataset": src, "patient_id": int(pid), "encounter_id": (int(enc) if enc is not None else None), "present_after_sampling": bool(ok)})
            req_path = os.path.join(paths.artifacts_dir, "required_patients_check.json")
            with open(req_path, "w") as f:
                json.dump(required, f, indent=2)
            logger.log(f"Saved required patient/encounter check: {req_path}")

    labs_before = int(len(events[events["event_type"] == "lab"])) if not events.empty else 0
    meds_before = int(len(events[events["event_type"] == "medication"])) if not events.empty else 0
    procs_before = int(len(events[events["event_type"] == "procedure"])) if not events.empty else 0

    labs = _thin_labs(
        events[events["event_type"] == "lab"],
        keep_every_hours=args.lab_normal_keep_hours,
        max_normal_per_feature_per_encounter=args.lab_max_normal_per_feature_per_encounter,
    )
    meds = _thin_interventions(
        events[events["event_type"] == "medication"],
        min_gap_minutes=args.intervention_min_gap_minutes,
    )
    procs = _thin_interventions(
        events[events["event_type"] == "procedure"],
        min_gap_minutes=args.intervention_min_gap_minutes,
    )
    rest = events[~events["event_type"].isin(["lab", "medication", "procedure"])]
    events = _finalize(pd.concat([labs, meds, procs, rest], ignore_index=True))
    events = events.sort_values(["source_dataset", "patient_id", "encounter_id", "event_time"])

    logger.log(
        "Filtering summary: "
        f"labs {labs_before}->{len(labs)}, meds {meds_before}->{len(meds)}, procs {procs_before}->{len(procs)}"
    )

    canonical_path = os.path.join(paths.artifacts_dir, "canonical_events.parquet")
    events.to_parquet(canonical_path, index=False)
    logger.log(f"Saved canonical events: {canonical_path}")

    protocol_feature_df = build_protocol_feature_table(events, core_only=True)
    protocol_feature_path = os.path.join(paths.artifacts_dir, "protocol_features.parquet")
    protocol_feature_df.to_parquet(protocol_feature_path, index=False)
    logger.log(f"Saved protocol feature table: {protocol_feature_path}")

    protocol_schema = {
        "rule_template": {
            "rule_id": "STRING_ID",
            "trigger": [
                {"feature": "map", "op": "<", "value": 65, "window_hours": 1},
                {"feature": "lactate", "op": "trend_is", "value": "rising", "window_hours": 6},
            ],
            "state_update": {"hemodynamic_instability_score": "+1"},
            "risk": "shock_progression",
            "severity": "high",
        },
        "compatible_protocol_features_columns": list(protocol_feature_df.columns),
        "notes": [
            "feature values come from canonical alias mapping",
            "trend_tag is computed per feature trajectory within encounter",
            "abnormal_flag supports low/normal/high/unknown states for symbolic triggers",
        ],
    }
    protocol_schema_path = os.path.join(paths.artifacts_dir, "protocol_schema.json")
    with open(protocol_schema_path, "w") as f:
        json.dump(protocol_schema, f, indent=2)
    logger.log(f"Saved protocol schema: {protocol_schema_path}")

    global_protocol_seed = build_global_protocol_seed_v1()
    global_protocol_seed_path = os.path.join(paths.artifacts_dir, "global_protocol_seed_v1.json")
    with open(global_protocol_seed_path, "w") as f:
        json.dump(global_protocol_seed, f, indent=2)
    logger.log(f"Saved global protocol seed: {global_protocol_seed_path}")

    mimic_labels_path = os.path.join(paths.artifacts_dir, "mimic_admission_labels.parquet")
    eicu_labels_path = os.path.join(paths.artifacts_dir, "eicu_patient_labels.parquet")
    if not mimic_labels.empty:
        mimic_labels.to_parquet(mimic_labels_path, index=False)
    if not eicu_labels.empty:
        eicu_labels.to_parquet(eicu_labels_path, index=False)

    label_patterns = _compile_label_patterns(args.labeling_mode)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    horizon_outputs = {}
    for h in horizons:
        logger.log(f"Building horizon examples: {h}h")
        ex = build_horizon_examples(
            events,
            horizon_hours=h,
            max_context_events=args.max_context_events,
            label_patterns=label_patterns,
        )
        if args.drop_empty_protocol_steps:
            ex = ex[ex["protocol_observations"].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
        out_path = os.path.join(paths.artifacts_dir, f"longitudinal_examples_h{h}.jsonl")
        with open(out_path, "w") as f:
            for _, r in ex.iterrows():
                rec = r.to_dict()
                rec["future_event_types"] = rec.get("future_event_types", {})
                rec["future_event_names"] = rec.get("future_event_names", [])
                f.write(json.dumps(rec) + "\n")
        horizon_outputs[str(h)] = {"path": out_path, "rows": int(len(ex))}
        _sample_and_write(ex, os.path.join(paths.samples_dir, f"examples_h{h}_sample.json"))
        logger.log(f"Saved examples h={h}: rows={len(ex)} path={out_path}")

    typed_outputs = {}
    if args.emit_typed_steps:
        logger.log(f"Building typed steps (mode={args.typed_step_mode})...")
        if args.typed_step_mode == "binned":
            typed_steps = build_typed_binned_steps(
                events,
                bin_hours=args.typed_bin_hours,
                max_events_per_step=args.max_events_per_step,
                max_steps_per_trajectory=args.typed_max_steps_per_trajectory,
            )
        else:
            typed_steps = build_typed_time_aware_steps(
                events,
                same_type_max_gap_hours=args.same_type_max_gap_hours,
                force_new_step_gap_hours=args.force_new_step_gap_hours,
                max_events_per_step=args.max_events_per_step,
            )
        typed_steps_path = os.path.join(paths.artifacts_dir, "typed_steps.jsonl")
        write_jsonl(typed_steps_path, typed_steps)
        logger.log(f"Saved typed steps: n={len(typed_steps)} path={typed_steps_path}")
        for h in horizons:
            tx = build_horizon_examples_from_typed_steps(
                typed_steps,
                horizon_hours=h,
                max_context_steps=args.max_context_steps,
                require_protocol_observations=args.drop_empty_protocol_steps,
                label_patterns=label_patterns,
            )
            tx_path = os.path.join(paths.artifacts_dir, f"typed_longitudinal_examples_h{h}.jsonl")
            write_jsonl(tx_path, tx)
            typed_outputs[str(h)] = {"path": tx_path, "rows": int(len(tx))}
            logger.log(f"Saved typed examples h={h}: rows={len(tx)} path={tx_path}")

    stats = summarize(events, mimic_labels, eicu_labels)
    stats["horizon_outputs"] = horizon_outputs
    if typed_outputs:
        stats["typed_horizon_outputs"] = typed_outputs
    stats["filtering"] = {
        "lab_normal_keep_hours": args.lab_normal_keep_hours,
        "lab_max_normal_per_feature_per_encounter": args.lab_max_normal_per_feature_per_encounter,
        "intervention_min_gap_minutes": args.intervention_min_gap_minutes,
        "labs_rows_before": labs_before,
        "labs_rows_after": int(len(labs)),
        "med_rows_before": meds_before,
        "med_rows_after": int(len(meds)),
        "proc_rows_before": procs_before,
        "proc_rows_after": int(len(procs)),
        "labeling_mode": args.labeling_mode,
        "typed_step_mode": args.typed_step_mode,
        "typed_bin_hours": int(args.typed_bin_hours),
        "typed_max_steps_per_trajectory": int(args.typed_max_steps_per_trajectory),
    }

    qc_report = build_qc_report(events, protocol_feature_df, horizon_outputs)
    qc_path = os.path.join(paths.eval_dir, "qc_report.json")
    with open(qc_path, "w") as f:
        json.dump(qc_report, f, indent=2)
    logger.log(f"Saved QC report: {qc_path}")

    stats_path = os.path.join(paths.eval_dir, "dataset_summary.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.log(f"Saved dataset summary: {stats_path}")

    outputs = {
        "canonical_events": canonical_path,
        "protocol_features": protocol_feature_path,
        "protocol_schema": protocol_schema_path,
        "global_protocol_seed_v1": global_protocol_seed_path,
        "mimic_labels": mimic_labels_path if os.path.exists(mimic_labels_path) else None,
        "eicu_labels": eicu_labels_path if os.path.exists(eicu_labels_path) else None,
        "horizon_examples": horizon_outputs,
        "typed_horizon_examples": typed_outputs if typed_outputs else None,
        "dataset_summary": stats_path,
        "qc_report": qc_path,
        "logs": os.path.join(paths.logs_dir, "preprocess.log"),
    }
    write_manifest(
        paths,
        args,
        inputs={"mimic_root": args.mimic_root, "eicu_root": args.eicu_root},
        outputs=outputs,
        stats=stats,
    )
    logger.log("Completed preprocessing run.")


if __name__ == "__main__":
    main()

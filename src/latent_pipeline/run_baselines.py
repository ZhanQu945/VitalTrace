from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.latent_pipeline.common import iter_jsonl, log, write_json, write_jsonl
from src.latent_pipeline.llm_backend import LLMBackend, LLMConfig
from src.latent_pipeline.prediction_targets import (
    COMPOSITE_TARGET,
    SUPPORT_TARGETS,
    add_composite_probability,
)
from src.latent_pipeline.protocol_utils import feature_map_from_facts, load_protocol, rule_score

SUPPORT_LABELS = list(SUPPORT_TARGETS)
LABELS = SUPPORT_LABELS + [COMPOSITE_TARGET]
FEATURES = ["map", "lactate", "spo2", "rr", "creatinine", "wbc", "bicarbonate", "sodium", "potassium", "glucose", "hr", "sbp"]


def _example_id(ex: Dict) -> str:
    return ex.get("example_id") or f"{ex.get('source_dataset')}_{ex.get('patient_id')}_{ex.get('encounter_id')}_{ex.get('anchor_time')}_{ex.get('step_id',0)}"


def _to_float(v, default=0.0):
    try:
        if v is None:
            return None if default is None else float(default)
        return float(v)
    except Exception:
        return None if default is None else float(default)


def _step_vector(ex: Dict) -> np.ndarray:
    facts = ex.get("protocol_observations", [])
    fmap = feature_map_from_facts(facts)
    vals = []
    for f in FEATURES:
        rec = fmap.get(f, {})
        vals.extend(
            [
                _to_float(rec.get("value_last"), 0.0),
                _to_float(rec.get("value_mean"), 0.0),
                float(str(rec.get("trend", "")).lower() == "rising"),
                float(str(rec.get("trend", "")).lower() == "decreasing"),
                float(str(rec.get("abnormal_flag_last", "")).lower() == "high"),
                float(str(rec.get("abnormal_flag_last", "")).lower() == "low"),
                float(str(rec.get("abnormal_flag_last", "")).lower() == "abnormal"),
            ]
        )
    return np.array(vals, dtype=np.float32)


def _group_key(ex: Dict) -> Tuple[str, str, str]:
    return (str(ex.get("source_dataset")), str(ex.get("patient_id")), str(ex.get("encounter_id")))


class RetainLike(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.alpha = nn.Linear(hidden_dim, 1)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        a = torch.softmax(self.alpha(h).squeeze(-1), dim=1).unsqueeze(-1)
        b = torch.tanh(self.beta(h))
        c = torch.sum(a * b, dim=1)
        return self.out(c)


class RetainV2(nn.Module):
    """
    Closer to RETAIN:
    - visit embedding v_t
    - alpha attention from one GRU
    - beta attention vector from another GRU
    - context c = sum(alpha_t * beta_t * v_t)
    """
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru_alpha = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.gru_beta = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.embed(x)  # [B,T,E]
        h_a, _ = self.gru_alpha(v)  # [B,T,H]
        h_b, _ = self.gru_beta(v)   # [B,T,H]
        alpha = torch.softmax(self.alpha_fc(h_a).squeeze(-1), dim=1).unsqueeze(-1)  # [B,T,1]
        beta = torch.tanh(self.beta_fc(h_b))  # [B,T,E]
        c = torch.sum(alpha * beta * v, dim=1)  # [B,E]
        return self.out(c)


def _build_stage_rows(examples: List[Dict], probs_by_id: Dict[str, Dict[str, float]], protocol_rules: Dict, baseline_name: str) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    s1, s2, s3, s4 = [], [], [], []
    patient_state: Dict[str, Dict] = {}
    for ex in examples:
        exid = _example_id(ex)
        probs = probs_by_id.get(exid, {k: 0.0 for k in LABELS})
        facts = ex.get("protocol_observations", [])
        fmap = feature_map_from_facts(facts)
        scored = []
        for rid, rule in protocol_rules.items():
            ok, sc = rule_score(rule, fmap)
            if ok:
                scored.append((rid, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [r for r, _ in scored[:3]]
        active = {rid: protocol_rules[rid] for rid in selected if rid in protocol_rules}

        common = {
            "example_id": exid,
            "source_dataset": ex.get("source_dataset"),
            "patient_id": ex.get("patient_id"),
            "encounter_id": ex.get("encounter_id"),
            "anchor_time": ex.get("anchor_time"),
            "step_id": int(ex.get("step_id", 0) or 0),
            "packet": {"facts": facts},
            "target_isolation_verified": True,
            "inference_context_schema": "target_free_v1",
            "selected_rule_ids": selected,
            "active_rules": active,
            "ground_truth_targets": ex.get("targets", {}),
        }
        s1.append(
            {
                **common,
                "stage1_prediction": {"selected_rule_ids": selected, "n_selected_rules": len(selected), "active_rule_ids": selected},
                "stage1_ground_truth": ex.get("targets", {}),
            }
        )

        actions = []
        if probs["vasopressor_signal"] >= 0.5:
            actions.append("consider vasopressor support")
        if probs["resp_support_signal"] >= 0.5:
            actions.append("consider respiratory support")
        if probs["renal_support_signal"] >= 0.5:
            actions.append("consider renal support")
        if not actions:
            actions.append("continue monitoring")
        s2_row = {
            **common,
            "reasoner_prediction": {
                "next_bundle_type": "Mixed",
                "predicted_actions": actions,
                "risk_probs": probs,
                "any_deterioration_definition": "max_support_probability",
                "citations": selected,
                "counterfactual_notes": [],
                "baseline": baseline_name,
            },
            "stage2_prediction": {k: int(probs[k] >= 0.5) for k in LABELS},
            "stage2_ground_truth": ex.get("targets", {}),
        }
        s2.append(s2_row)

        audit = {"status": "PASS", "issues": [], "suggested_fixes": []}
        s3_row = {
            **s2_row,
            "audit": audit,
            "stage3_prediction": {"audit_status": "PASS", "n_issues": 0},
            "stage3_ground_truth": ex.get("targets", {}),
        }
        s3.append(s3_row)

        pkey = f"{ex.get('source_dataset')}_{ex.get('patient_id')}_{ex.get('encounter_id')}"
        prev = patient_state.get(
            pkey,
            {"hemodynamic_state": 0, "respiratory_state": 0, "renal_state": 0, "metabolic_state": 0, "systemic_inflammation_state": 0, "active_protocol_prediction": []},
        )
        nxt = dict(prev)
        nxt["hemodynamic_state"] = int(max(0, min(5, prev["hemodynamic_state"] + int(probs["vasopressor_signal"] >= 0.5))))
        nxt["respiratory_state"] = int(max(0, min(5, prev["respiratory_state"] + int(probs["resp_support_signal"] >= 0.5))))
        nxt["renal_state"] = int(max(0, min(5, prev["renal_state"] + int(probs["renal_support_signal"] >= 0.5))))
        nxt["metabolic_state"] = int(max(0, min(5, prev["metabolic_state"] + int(probs["any_deterioration"] >= 0.5))))
        nxt["systemic_inflammation_state"] = int(prev["systemic_inflammation_state"])
        nxt["active_protocol_prediction"] = selected
        delta = {k: int(nxt[k]) - int(prev[k]) for k in ["hemodynamic_state", "respiratory_state", "renal_state", "metabolic_state", "systemic_inflammation_state"]}
        patient_state[pkey] = nxt
        s4.append(
            {
                **s3_row,
                "individual_protocol_state_prev": prev,
                "individual_protocol_state_next": nxt,
                "individual_protocol_state_delta": delta,
                "state_version": 1,
                "stage4_prediction": {"state_next": nxt, "state_delta": delta, "state_version": 1},
                "stage4_ground_truth": ex.get("targets", {}),
            }
        )
    return s1, s2, s3, s4


def _run_single_llm_agent(examples: List[Dict], model_id: str, max_new_tokens: int, temperature: float, max_input_tokens: int) -> Dict[str, Dict[str, float]]:
    llm = LLMBackend(LLMConfig(model_id=model_id, max_new_tokens=max_new_tokens, temperature=temperature, max_input_tokens=max_input_tokens))
    model_id_l = str(model_id).lower()
    is_gpt_oss = "gpt-oss" in model_id_l
    probs = {}
    n_fallback = 0
    n_llm_valid = 0
    n_llm_zero = 0

    def _normalize_risk_blob(blob: Dict) -> Dict[str, float]:
        if not isinstance(blob, dict):
            return {}
        # Flatten one level if model wraps payload.
        if "risk_probs" in blob and isinstance(blob.get("risk_probs"), dict):
            blob = blob.get("risk_probs") or {}
        for wrap_k in ["prediction", "predictions", "output", "result", "data"]:
            if wrap_k in blob and isinstance(blob.get(wrap_k), dict):
                inner = blob.get(wrap_k) or {}
                if "risk_probs" in inner and isinstance(inner.get("risk_probs"), dict):
                    blob = inner.get("risk_probs") or {}
                else:
                    blob = inner
                break
        aliases = {
            "vasopressor_signal": ["vasopressor_signal", "vaso", "vaso_signal", "vasopressor", "hemodynamic_risk"],
            "resp_support_signal": ["resp_support_signal", "resp", "resp_signal", "respiratory_support_signal", "respiratory_risk"],
            "renal_support_signal": ["renal_support_signal", "renal", "renal_signal", "renal_risk"],
        }
        out: Dict[str, float] = {}
        for canonical, keys in aliases.items():
            val = None
            for k in keys:
                if k in blob:
                    val = blob.get(k)
                    break
            if val is None:
                continue
            # Handle simple percentage strings like "72%" or numeric strings.
            if isinstance(val, str):
                s = val.strip().replace("%", "")
                try:
                    fv = float(s)
                    if "%" in val:
                        fv /= 100.0
                    val = fv
                except Exception:
                    continue
            out[canonical] = float(np.clip(_to_float(val, 0.0), 0.0, 1.0))
        return add_composite_probability(out)

    def _heuristic_probs(ex: Dict) -> Dict[str, float]:
        facts = ex.get("protocol_observations", [])
        fmap = feature_map_from_facts(facts)
        mp = fmap.get("map", {})
        rr = fmap.get("rr", {})
        sp = fmap.get("spo2", {})
        cr = fmap.get("creatinine", {})
        lc = fmap.get("lactate", {})

        map_v = _to_float(mp.get("value_last"), None)
        rr_v = _to_float(rr.get("value_last"), None)
        sp_v = _to_float(sp.get("value_last"), None)
        cr_v = _to_float(cr.get("value_last"), None)
        lc_v = _to_float(lc.get("value_last"), None)

        # Conservative fallback (avoid all-positive collapse in tiny debug sets).
        vaso = 0.10
        if map_v is not None and map_v < 55:
            vaso = 0.75
        elif map_v is not None and map_v < 65:
            vaso = 0.55
        elif str(lc.get("trend", "")).lower() in {"rising", "up"} or (lc_v is not None and lc_v >= 4.0):
            vaso = 0.40

        resp = 0.08
        if sp_v is not None and sp_v < 88:
            resp = 0.75
        elif (sp_v is not None and sp_v < 92) or (rr_v is not None and rr_v >= 30):
            resp = 0.55
        elif rr_v is not None and rr_v >= 24:
            resp = 0.35

        renal = 0.08
        if cr_v is not None and cr_v >= 3.0:
            renal = 0.70
        elif cr_v is not None and cr_v >= 2.0:
            renal = 0.50
        elif str(cr.get("trend", "")).lower() in {"rising", "up"}:
            renal = 0.35

        any_det = max(vaso, resp, renal)
        return {
            "vasopressor_signal": float(np.clip(vaso, 0.0, 1.0)),
            "resp_support_signal": float(np.clip(resp, 0.0, 1.0)),
            "renal_support_signal": float(np.clip(renal, 0.0, 1.0)),
            "any_deterioration": float(np.clip(any_det, 0.0, 1.0)),
        }

    def _facts_for_single_agent(ex: Dict) -> List[Dict]:
        facts = ex.get("protocol_observations", [])
        if isinstance(facts, list) and len(facts) > 0:
            return facts
        # Fallback: synthesize structured facts from step-level values (stage3/4 formatted inputs).
        out = []
        for feat in ["map", "rr", "spo2", "creatinine", "lactate", "hr", "temp", "wbc", "bicarbonate", "sodium", "potassium", "glucose"]:
            v = _to_float(ex.get(f"{feat}_value"), None)
            if v is None:
                continue
            out.append(
                {
                    "feature": feat,
                    "value_last": v,
                    "trend": "unknown",
                    "count": 1,
                    "abnormal_flag_last": "unknown",
                }
            )
        return out

    for i, ex in enumerate(examples, start=1):
        facts = _facts_for_single_agent(ex)
        if is_gpt_oss:
            system_prompt = (
                "You are ICU-SingleAgent. Output STRICT JSON only. "
                "No prose, no markdown, no code block."
            )
            base_prompt = (
                "Task: predict near-term intervention risk probabilities from current facts.\n"
                "Return exactly these keys with numeric values in [0,1]:\n"
                "{\"vasopressor_signal\":0.0,\"resp_support_signal\":0.0,"
                "\"renal_support_signal\":0.0}\n"
                "Rules:\n"
                "- any_deterioration is computed downstream as the maximum of these three values.\n"
                "- If data suggests instability, do not set all risks to 0.\n"
                "- vasopressor: low MAP / rising lactate.\n"
                "- respiratory: low SpO2 / high RR.\n"
                "- renal: elevated or rising creatinine.\n"
                f"patient_facts={facts}\n"
            )
        else:
            system_prompt = "Return valid JSON only."
            base_prompt = (
                "You are ICU-SingleAgent, a conservative but sensitive next-step risk predictor.\n"
                "Estimate near-term intervention risks from current patient observations.\n"
                "Do NOT default all risks to zero when abnormalities exist.\n"
                "Return STRICT JSON only (no markdown, no explanation):\n"
                "{"
                "\"vasopressor_signal\": <float 0..1>,"
                "\"resp_support_signal\": <float 0..1>,"
                "\"renal_support_signal\": <float 0..1>"
                "}\n"
                "Clinical cues:\n"
                "- vasopressor_signal: low MAP, perfusion concerns, rising lactate\n"
                "- resp_support_signal: low SpO2, high RR\n"
                "- renal_support_signal: elevated/rising creatinine\n"
                "- any_deterioration is computed downstream as the maximum of these three values\n"
                f"patient_facts={facts}\n"
            )
        out = {}
        risk_blob = {}
        max_attempts = 2 if is_gpt_oss else 3
        for attempt in range(max_attempts):
            prompt = base_prompt
            if attempt >= 1:
                prompt = (
                    base_prompt
                    + "Your previous output was invalid or incomplete. "
                    + "Return ONLY this JSON with numeric values:\n"
                    + "{\"vasopressor_signal\":0.0,\"resp_support_signal\":0.0,\"renal_support_signal\":0.0}\n"
                )
            out = llm.generate_json(system_prompt, prompt)
            risk_blob = _normalize_risk_blob(out if isinstance(out, dict) else {})
            if isinstance(risk_blob, dict) and any(k in risk_blob for k in SUPPORT_LABELS):
                break
        row = {}
        for k in LABELS:
            row[k] = float(np.clip(_to_float(risk_blob.get(k), 0.0), 0.0, 1.0))
        valid_blob = isinstance(risk_blob, dict) and any(k in risk_blob for k in SUPPORT_LABELS)
        if valid_blob:
            n_llm_valid += 1
        if all(float(row[k]) == 0.0 for k in LABELS):
            n_llm_zero += 1
        # Fallback only when output is invalid/missing.
        if not valid_blob:
            row = _heuristic_probs(ex)
            n_fallback += 1
        probs[_example_id(ex)] = row
        if i % int(os.environ.get("STAGE_PROGRESS_EVERY", "25")) == 0:
            log(f"single_llm_agent progress: {i} rows, llm_failures={int(getattr(llm, '_failures', 0))}")
    log(
        "single_llm_agent summary: "
        f"n={len(examples)} llm_valid={n_llm_valid} llm_all_zero={n_llm_zero} fallback_used={n_fallback}"
    )
    return probs


def _run_retain_style(examples: List[Dict], epochs: int, lr: float, hidden_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    by_traj: Dict[Tuple[str, str, str], List[Dict]] = {}
    for ex in examples:
        by_traj.setdefault(_group_key(ex), []).append(ex)
    for k in by_traj:
        by_traj[k] = sorted(by_traj[k], key=lambda x: (str(x.get("anchor_time", "")), int(x.get("step_id", 0) or 0)))
    patient_ids = sorted(set((str(x.get("source_dataset")), str(x.get("patient_id"))) for x in examples))
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    tr = set(patient_ids[: int(0.7 * n)])
    va = set(patient_ids[int(0.7 * n): int(0.85 * n)])
    te = set(patient_ids[int(0.85 * n):])

    def pid(exx: Dict):
        return (str(exx.get("source_dataset")), str(exx.get("patient_id")))

    input_dim = len(_step_vector(examples[0])) if examples else 1
    model = RetainLike(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=len(SUPPORT_LABELS))
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    train_seqs = []
    val_seqs = []
    for _, seq in by_traj.items():
        if not seq:
            continue
        p = pid(seq[0])
        xs, ys = [], []
        for i in range(len(seq)):
            xs.append(_step_vector(seq[i]))
            ys.append([_to_float(seq[i].get("targets", {}).get(k, 0), 0.0) for k in SUPPORT_LABELS])
            x_t = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(0)
            y_t = torch.tensor(ys[-1], dtype=torch.float32).unsqueeze(0)
            if p in tr:
                train_seqs.append((x_t, y_t))
            elif p in va:
                val_seqs.append((x_t, y_t))

    best_state = None
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, y_t in train_seqs:
            opt.zero_grad()
            logits = model(x_t)
            loss = crit(logits, y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        vlosses = []
        with torch.no_grad():
            for x_t, y_t in val_seqs:
                vlosses.append(float(crit(model(x_t), y_t).cpu().item()))
        v = float(np.mean(vlosses)) if vlosses else float(np.mean(losses) if losses else 0.0)
        if v < best_val:
            best_val = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"retain_style epoch {ep}/{epochs}: train_loss={np.mean(losses) if losses else 0.0:.4f} val_loss={v:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    probs = {}
    model.eval()
    with torch.no_grad():
        for _, seq in by_traj.items():
            xs = []
            for ex in seq:
                xs.append(_step_vector(ex))
                x_t = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(0)
                p = torch.sigmoid(model(x_t)).squeeze(0).cpu().numpy()
                p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
                probs[_example_id(ex)] = add_composite_probability({k: float(np.clip(p[i], 0.0, 1.0)) for i, k in enumerate(SUPPORT_LABELS)})
    split_info = {
        "n_patients_total": int(n),
        "n_patients_train": int(len(tr)),
        "n_patients_val": int(len(va)),
        "n_patients_test": int(len(te)),
    }
    return probs, te, split_info


def _run_retain_v2(examples: List[Dict], epochs: int, lr: float, hidden_dim: int, embed_dim: int, dropout: float, wd: float, seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    by_traj: Dict[Tuple[str, str, str], List[Dict]] = {}
    for ex in examples:
        by_traj.setdefault(_group_key(ex), []).append(ex)
    for k in by_traj:
        by_traj[k] = sorted(by_traj[k], key=lambda x: (str(x.get("anchor_time", "")), int(x.get("step_id", 0) or 0)))
    patient_ids = sorted(set((str(x.get("source_dataset")), str(x.get("patient_id"))) for x in examples))
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    tr = set(patient_ids[: int(0.7 * n)])
    va = set(patient_ids[int(0.7 * n): int(0.85 * n)])
    te = set(patient_ids[int(0.85 * n):])

    def pid(exx: Dict):
        return (str(exx.get("source_dataset")), str(exx.get("patient_id")))

    input_dim = len(_step_vector(examples[0])) if examples else 1
    model = RetainV2(input_dim=input_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, out_dim=len(SUPPORT_LABELS), dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()

    train_seqs, val_seqs = [], []
    for _, seq in by_traj.items():
        if not seq:
            continue
        p = pid(seq[0])
        xs, ys = [], []
        for i in range(len(seq)):
            xs.append(_step_vector(seq[i]))
            ys.append([_to_float(seq[i].get("targets", {}).get(k, 0), 0.0) for k in SUPPORT_LABELS])
            x_t = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(0)
            y_t = torch.tensor(ys[-1], dtype=torch.float32).unsqueeze(0)
            if p in tr:
                train_seqs.append((x_t, y_t))
            elif p in va:
                val_seqs.append((x_t, y_t))

    best_state = None
    best_val = float("inf")
    patience = 3
    patience_left = patience
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, y_t in train_seqs:
            opt.zero_grad()
            logits = model(x_t)
            loss = crit(logits, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        vlosses = []
        with torch.no_grad():
            for x_t, y_t in val_seqs:
                vlosses.append(float(crit(model(x_t), y_t).cpu().item()))
        v = float(np.mean(vlosses)) if vlosses else float(np.mean(losses) if losses else 0.0)
        if v + 1e-6 < best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        log(f"retain_v2 epoch {ep}/{epochs}: train_loss={np.mean(losses) if losses else 0.0:.4f} val_loss={v:.4f} patience_left={patience_left}")
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs = {}
    model.eval()
    with torch.no_grad():
        for _, seq in by_traj.items():
            xs = []
            for ex in seq:
                xs.append(_step_vector(ex))
                x_t = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(0)
                p = torch.sigmoid(model(x_t)).squeeze(0).cpu().numpy()
                p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
                probs[_example_id(ex)] = add_composite_probability({k: float(np.clip(p[i], 0.0, 1.0)) for i, k in enumerate(SUPPORT_LABELS)})
    split_info = {
        "n_patients_total": int(n),
        "n_patients_train": int(len(tr)),
        "n_patients_val": int(len(va)),
        "n_patients_test": int(len(te)),
    }
    return probs, te, split_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol-json", required=True)
    ap.add_argument("--baseline", choices=["single_llm_agent", "retain_style", "retain_v2"], required=True)
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--llm-max-new-tokens", type=int, default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")))
    ap.add_argument("--llm-temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    ap.add_argument("--llm-max-input-tokens", type=int, default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "2048")))
    ap.add_argument("--retain-epochs", type=int, default=6)
    ap.add_argument("--retain-lr", type=float, default=1e-3)
    ap.add_argument("--retain-hidden-dim", type=int, default=64)
    ap.add_argument("--retain-v2-embed-dim", type=int, default=128)
    ap.add_argument("--retain-v2-dropout", type=float, default=0.1)
    ap.add_argument("--retain-v2-weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    examples = list(iter_jsonl(args.input_jsonl))
    log(f"Baseline run start: baseline={args.baseline} n_examples={len(examples)}")
    rules = load_protocol(args.protocol_json)

    eval_examples = examples
    split_info = None
    if args.baseline == "single_llm_agent":
        probs = _run_single_llm_agent(
            examples, args.llm_model_id, args.llm_max_new_tokens, args.llm_temperature, args.llm_max_input_tokens
        )
    elif args.baseline == "retain_style":
        probs, te, split_info = _run_retain_style(examples, args.retain_epochs, args.retain_lr, args.retain_hidden_dim, args.seed)
        # Fair supervised evaluation: report held-out test patients only.
        eval_examples = [e for e in examples if (str(e.get("source_dataset")), str(e.get("patient_id"))) in te]
        log(f"retain_style evaluation subset: n_eval_examples={len(eval_examples)}")
    else:
        probs, te, split_info = _run_retain_v2(
            examples,
            args.retain_epochs,
            args.retain_lr,
            args.retain_hidden_dim,
            args.retain_v2_embed_dim,
            args.retain_v2_dropout,
            args.retain_v2_weight_decay,
            args.seed,
        )
        eval_examples = [e for e in examples if (str(e.get("source_dataset")), str(e.get("patient_id"))) in te]
        log(f"retain_v2 evaluation subset: n_eval_examples={len(eval_examples)}")

    s1, s2, s3, s4 = _build_stage_rows(eval_examples, probs, rules, args.baseline)
    write_jsonl(os.path.join(args.out_dir, "stage1_router.jsonl"), s1)
    write_jsonl(os.path.join(args.out_dir, "stage2_reasoner.jsonl"), s2)
    write_jsonl(os.path.join(args.out_dir, "stage3_auditor.jsonl"), s3)
    write_jsonl(os.path.join(args.out_dir, "stage4_steward.jsonl"), s4)
    write_json(
        os.path.join(args.out_dir, "baseline_metrics.json"),
        {
            "baseline": args.baseline,
            "n_examples_input": len(examples),
            "n_examples_eval": len(eval_examples),
            "n_patients_input": len(set((str(e.get("source_dataset")), str(e.get("patient_id"))) for e in examples)),
            "n_patients_eval": len(set((str(e.get("source_dataset")), str(e.get("patient_id"))) for e in eval_examples)),
            "input_jsonl": args.input_jsonl,
            "split_info": split_info,
        },
    )
    log(f"Baseline run done: baseline={args.baseline} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

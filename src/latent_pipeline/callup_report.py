import argparse
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from src.latent_pipeline.common import iter_jsonl, write_json


def _eid(o: Dict) -> str:
    sid = o.get('source_dataset')
    pid = o.get('patient_id')
    enc = o.get('encounter_id')
    t = o.get('anchor_time')
    step = o.get('step_id', '')
    return f"{sid}_{pid}_{enc}_{t}_{step}"


def _parse_time(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def run(stage1_jsonl: str, out_json: str):
    rows = [o for o in iter_jsonl(stage1_jsonl)]
    n = len(rows)

    # Clinical activity density
    total_entities = []
    step_types = defaultdict(list)
    called_flags = []
    active_rule_counts = []

    # per-trajectory for lead-time proxy
    traj = defaultdict(list)

    for o in rows:
        facts = o.get('packet', {}).get('facts', [])
        total_entities.append(len(facts))
        st = o.get('step_type', 'unknown')
        step_types[st].append(o)

        called = 1 if len(o.get('selected_rule_ids', [])) > 0 else 0
        called_flags.append(called)
        if called:
            active_rule_counts.append(len(o.get('selected_rule_ids', [])))

        key = (o.get('source_dataset'), o.get('patient_id'), o.get('encounter_id'))
        traj[key].append(o)

    # Target rates when called vs not called
    def _target_pos(o):
        t = o.get('ground_truth_targets', {})
        return int(
            t.get('vasopressor_signal', 0)
            or t.get('resp_support_signal', 0)
            or t.get('renal_support_signal', 0)
            or t.get('any_deterioration', 0)
        )

    called_pos = []
    not_called_pos = []
    for o in rows:
        called = len(o.get('selected_rule_ids', [])) > 0
        if called:
            called_pos.append(_target_pos(o))
        else:
            not_called_pos.append(_target_pos(o))

    # Lead time proxy: for called steps, hours until next target-positive step in same trajectory
    lead_hours = []
    for _, seq in traj.items():
        seq = sorted(seq, key=lambda x: (x.get('anchor_time', ''), x.get('step_id', -1)))
        for i, o in enumerate(seq):
            if len(o.get('selected_rule_ids', [])) == 0:
                continue
            t0 = _parse_time(o.get('anchor_time', ''))
            if t0 is None:
                continue
            found = False
            for j in range(i + 1, len(seq)):
                if _target_pos(seq[j]) > 0:
                    t1 = _parse_time(seq[j].get('anchor_time', ''))
                    if t1 is None:
                        break
                    dt_h = (t1 - t0).total_seconds() / 3600.0
                    if dt_h >= 0:
                        lead_hours.append(dt_h)
                    found = True
                    break
            if not found:
                pass

    # callup by step type
    by_type = {}
    for st, seq in step_types.items():
        m = len(seq)
        if m == 0:
            continue
        c = sum(1 for o in seq if len(o.get('selected_rule_ids', [])) > 0)
        by_type[st] = {
            'n_steps': m,
            'protocol_called_rate': c / m,
            'avg_entities_per_step': sum(len(o.get('packet', {}).get('facts', [])) for o in seq) / m,
        }

    rep = {
        'n_steps_total': n,
        'avg_total_entities_per_step': (sum(total_entities) / n) if n else 0.0,
        'protocol_called_rate': (sum(called_flags) / n) if n else 0.0,
        'avg_active_rules_when_called': (sum(active_rule_counts) / len(active_rule_counts)) if active_rule_counts else 0.0,
        'callup_by_step_type': by_type,
        'target_event_rate_when_called': (sum(called_pos) / len(called_pos)) if called_pos else 0.0,
        'target_event_rate_when_not_called': (sum(not_called_pos) / len(not_called_pos)) if not_called_pos else 0.0,
        'lead_time_hours_mean_when_called': (sum(lead_hours) / len(lead_hours)) if lead_hours else None,
        'lead_time_hours_median_when_called': sorted(lead_hours)[len(lead_hours)//2] if lead_hours else None,
        'lead_time_samples': len(lead_hours),
    }

    write_json(out_json, rep)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-jsonl', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    run(args.stage1_jsonl, args.out_json)

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _pkey(o: Dict) -> Tuple[str, str]:
    return str(o.get("source_dataset")), str(o.get("patient_id"))


def main(input_jsonl: str, output_jsonl: str, summary_json: str, n_patients: int, require_protocol_calls: bool = True) -> None:
    rows: List[Dict] = []
    by_patient: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    patient_has_proto: Dict[Tuple[str, str], bool] = defaultdict(bool)

    with open(input_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(o)
            k = _pkey(o)
            by_patient[k].append(o)
            if len(o.get("protocol_observations", [])) > 0:
                patient_has_proto[k] = True

    keys = sorted(by_patient.keys())
    if require_protocol_calls:
        keys = [k for k in keys if patient_has_proto.get(k, False)]

    sel_keys = keys[:n_patients]
    out_rows: List[Dict] = []
    for k in sel_keys:
        out_rows.extend(by_patient[k])

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for o in out_rows:
            f.write(json.dumps(o) + "\n")

    proto_nonempty = sum(1 for o in out_rows if len(o.get("protocol_observations", [])) > 0)
    summary = {
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "n_input_rows": len(rows),
        "n_input_patients": len(by_patient),
        "require_protocol_calls": require_protocol_calls,
        "n_selected_patients": len(sel_keys),
        "n_output_rows": len(out_rows),
        "protocol_nonempty_rows": proto_nonempty,
        "protocol_nonempty_rate": (proto_nonempty / len(out_rows)) if out_rows else 0.0,
        "selected_patients": [{"source_dataset": k[0], "patient_id": k[1]} for k in sel_keys],
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--n-patients", type=int, default=10)
    ap.add_argument("--no-require-protocol-calls", action="store_true")
    args = ap.parse_args()
    main(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        n_patients=args.n_patients,
        require_protocol_calls=not args.no_require_protocol_calls,
    )

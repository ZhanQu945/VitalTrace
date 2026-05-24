import json
import os
import time
from typing import Dict, Iterable, List


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now_ts()}] {msg}", flush=True)


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def packet_to_text(packet: Dict) -> str:
    facts = packet.get("facts", [])
    parts = []
    for f in facts:
        parts.append(f"{f.get('feature')}={f.get('value_last')} trend={f.get('trend')} flag={f.get('abnormal_flag_last')}")
    probs = packet.get("risk_probs", {})
    parts.append("risks: " + ", ".join([f"{k}:{round(v,3)}" for k, v in probs.items()]))
    return " | ".join(parts)

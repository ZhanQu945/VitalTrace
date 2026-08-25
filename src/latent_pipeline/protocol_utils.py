import json
from typing import Dict, List, Tuple


def load_protocol(path: str) -> Dict:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "rules" in obj:
        rules = {r["rule_id"]: r for r in obj["rules"]}
    elif isinstance(obj, dict):
        rules = obj
    else:
        rules = {r.get("id", f"rule_{i}"): r for i, r in enumerate(obj)}
    return rules


def feature_map_from_facts(facts: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for f in facts:
        out[str(f.get("feature", ""))] = f
    return out


def rule_score(rule: Dict, fmap: Dict[str, Dict]) -> Tuple[bool, float]:
    triggers = rule.get("trigger", [])
    ok = True
    score = 0.0
    for t in triggers:
        feat = str(t.get("feature", "")).lower()
        op = t.get("op")
        val = t.get("value")
        min_occ = int(t.get("min_occurrences", 1) or 1)
        fact = fmap.get(feat)
        if fact is None:
            ok = False
            score -= 1.0
            continue
        fv = fact.get("value_last")
        tr = fact.get("trend")
        cnt = int(fact.get("count", 1) or 1)
        pass_t = False
        if op in ["<", "<=", ">", ">="] and fv is not None:
            if op == "<":
                pass_t = fv < val
            elif op == "<=":
                pass_t = fv <= val
            elif op == ">":
                pass_t = fv > val
            elif op == ">=":
                pass_t = fv >= val
        elif op == "outside_range":
            if fv is not None and isinstance(val, list) and len(val) == 2:
                lo, hi = val
                pass_t = (fv < lo) or (fv > hi)
        elif op == "trend_is":
            pass_t = str(tr) == str(val)
        elif op == "abnormal":
            pass_t = str(fact.get("abnormal_flag_last", "")).lower() in {"high", "low"}
        if pass_t and min_occ > 1:
            pass_t = cnt >= min_occ
        ok = ok and pass_t
        score += 1.0 if pass_t else -1.0
    return ok, score

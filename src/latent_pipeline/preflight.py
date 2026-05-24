from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    print(f"[PREFLIGHT][FAIL] {msg}")
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[PREFLIGHT][OK] {msg}")


def check_common(protocol_json: str | None, out_dir: str | None) -> None:
    if protocol_json:
        if "REPLACE_" in protocol_json:
            _fail(f"protocol_json has placeholder: {protocol_json}")
        if not Path(protocol_json).exists():
            _fail(f"protocol_json missing: {protocol_json}")
        _ok(f"protocol_json exists: {protocol_json}")
    if out_dir:
        if "REPLACE_" in out_dir:
            _fail(f"out_dir has placeholder: {out_dir}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _ok(f"out_dir writable: {out_dir}")


def check_input(input_jsonl: str) -> None:
    if "REPLACE_" in input_jsonl:
        _fail(f"input_jsonl has placeholder: {input_jsonl}")
    p = Path(input_jsonl)
    if not p.exists():
        _fail(f"input_jsonl missing: {input_jsonl}")
    if p.stat().st_size == 0:
        _fail(f"input_jsonl empty: {input_jsonl}")
    _ok(f"input_jsonl exists: {input_jsonl}")


def check_python_deps() -> None:
    mods = ["torch", "transformers", "numpy", "pandas", "sklearn", "pyarrow"]
    for m in mods:
        __import__(m)
    _ok("python dependencies importable")


def check_hf_access(model_id: str, backend: str) -> None:
    if backend != "llm":
        _ok("backend is deterministic; skip HF model access check")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        _fail("HF token missing (HF_TOKEN/HUGGINGFACE_HUB_TOKEN)")
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(repo_id=model_id, token=token)
    except Exception as e:
        _fail(f"cannot access model '{model_id}': {e}")
    _ok(f"HF access ok for model: {model_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl")
    ap.add_argument("--protocol-json")
    ap.add_argument("--out-dir")
    ap.add_argument("--agent-backend", default="llm")
    ap.add_argument("--llm-model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    ns = ap.parse_args()

    check_python_deps()
    if ns.input_jsonl:
        check_input(ns.input_jsonl)
    check_common(ns.protocol_json, ns.out_dir)
    check_hf_access(ns.llm_model_id, ns.agent_backend)
    print("[PREFLIGHT][OK] all checks passed")


if __name__ == "__main__":
    main()

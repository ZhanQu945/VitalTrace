from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config.loader import load_config


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    script = cfg.get("script", "data_v2/preprocess_longitudinal.py")
    args = cfg.get("args", {})
    flags = cfg.get("flags", {})

    cmd = [sys.executable, script]
    for k, v in args.items():
        if v is None:
            continue
        cmd.extend([f"--{k.replace('_','-')}", str(v)])
    for k, enabled in flags.items():
        if enabled:
            cmd.append(f"--{k.replace('_','-')}")

    print("[preprocess] cmd:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ns = ap.parse_args()
    main(ns.config)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(config_path.read_text())
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text())
    raise ValueError(f"Unsupported config format: {path}")

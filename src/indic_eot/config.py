"""Load project settings from params.yaml."""

from pathlib import Path
from typing import Any

import yaml

# src/indic_eot/config.py -> repo root is two levels above the package folder.
REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = REPO_ROOT / "params.yaml"


def load_params(path: Path | str = PARAMS_FILE) -> dict[str, Any]:
    """Return the contents of params.yaml as a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def resource_path(relative: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / relative


def load_defaults() -> dict[str, Any]:
    with resource_path("config/defaults.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
PRICING_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "pricing.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pricing(pricing_path: Optional[Path] = None) -> Dict[str, Any]:
    path = pricing_path or PRICING_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

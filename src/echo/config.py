import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(os.getenv("ECHO_CONFIGS_DIR", str(Path(__file__).parent.parent.parent / "configs")))

_config_cache: dict[str, dict] = {}

def load_yaml(filename: str) -> dict:
    if filename in _config_cache:
        return _config_cache[filename]

    path = str(CONFIGS_DIR / filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    if not config:
        raise ValueError(f"Config file is empty: {path}")

    logger.info(f"Loaded config '{filename}' from {path}" )
    _config_cache[filename] = config
    return config
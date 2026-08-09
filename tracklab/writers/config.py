import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from ..utils import _atomic_write


class ConfigWriter:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "config.json"

    def save(self, config):
        if isinstance(config, DictConfig):
            payload = OmegaConf.to_container(config)
        else:
            payload = config
        _atomic_write(self.path, json.dumps(payload, indent=4))
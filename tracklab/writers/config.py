import json
from omegaconf import DictConfig, OmegaConf

class ConfigWriter:
    def __init__(self, run_dir):
        self.path = f"{run_dir}/config.json"

    def save(self, config):
        with open(self.path, "w") as f:
            if isinstance(config, DictConfig):
                json.dump(OmegaConf.to_container(config), f, indent=4)
            else:
                json.dump(config, f, indent=4)
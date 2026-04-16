import os
from .run import Run

class Experiment:
    def __init__(self, experiment_name, base_dir="./data"):
        self.experiment_name = experiment_name
        self.exp_dir = f"{base_dir}/{experiment_name}"

        # create experiment directory if it doesn't exist
        os.makedirs(self.exp_dir, exist_ok=True)

    def start_run(self, config, artifacts=False):
        return Run(config, self.exp_dir, artifacts)

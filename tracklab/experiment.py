import os
from pathlib import Path
from .run import Run

class ExperimentTracker:
    """Class to manage experiments and runs. 
    It provides methods to start new runs, 
    list existing runs, and manage configurations."""
    def __init__(self, experiment_name, base_dir="./data"):
        self.experiment_name = experiment_name
        self.exp_dir = Path(base_dir)/experiment_name
        os.makedirs(self.exp_dir, exist_ok=True)

    def start_run(self, config, artifacts=False,min_flush_interval = 0.0) -> Run:
        return Run(config, self.exp_dir, artifacts,min_flush_interval=min_flush_interval)

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class ExperimentReader:
    def __init__(self, experiment_name, base_dir="./data"):
        self.experiment_name = experiment_name
        self.exp_dir = Path(base_dir)/experiment_name
    
    def list_runs(self):
        full_list = os.listdir(self.exp_dir)
        runs = []
        for dir in full_list:
            if dir.startswith("run_") and Path.exists(self.exp_dir/dir/"metrics.csv"):
                runs.append(dir)
        return runs
    
    def load_metrics(self, run_id):
        return pd.read_csv(self.exp_dir/run_id/"metrics.csv")
    
    def load_config(self, run_id):
        with open(self.exp_dir/run_id/"config.json", 'r') as f:
            return json.load(f)
    
    def list_artifacts(self, run_id):
        return pd.read_csv(self.exp_dir/run_id/"artifacts"/"index.csv")
    
    def load_artifact(self, run_id, artifact_name):
        artifact_path = self.exp_dir/run_id/"artifacts"/artifact_name
        if artifact_path.suffix == ".npy":
            return np.load(artifact_path)
        elif artifact_path.suffix == ".pkl":
            with open(artifact_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported artifact format: {artifact_path.suffix}")
    

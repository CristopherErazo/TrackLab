import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from .utils import flatten_dict

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
        
    def sumarize_runs(self, depth_names = 1):
        """
        Summarizes the runs in the experiment by creating a dataframe where each row corresponds 
        to a run and each column corresponds to a parameter in the configuration. 
        Only parameters that vary across runs are kept as columns. 
        The names of the columns are shortened to keep only the last part of the hierarchy of parameters up to a certain depth.
        """

        rows = []
        runs = self.list_runs()
        for run_id in runs:
            cfg = self.load_config(run_id)
            row = flatten_dict(cfg)
            row["run_id"] = run_id
            rows.append(row)
        # create dataframe with run_id as index and all the parameters as columns 
        df = pd.DataFrame(rows)#.set_index("run_id")
        # keep only parameters that var
        summary = df.loc[ : , df.nunique(dropna=False) > 1]
        # Change names of columns to keep only a depth of the hierarchy of parameters
        summary.columns = [ ".".join(col.rsplit(".", depth_names)[-depth_names:]) for col in summary.columns ]
        # Make run_id the first column
        cols = summary.columns.tolist()
        cols.insert(0, cols.pop(cols.index("run_id")))
        summary = summary[cols]
        return summary


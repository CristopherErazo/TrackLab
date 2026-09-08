import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path

from .writers.artifacts import _SERIALIZERS
from .utils import read_jsonl
from .live import MetricsStream

_EXT_TO_READ_FN = {ext: read_fn for _, (_, read_fn, ext) in _SERIALIZERS.items()}
# _EXT_TO_TYPE = {ext: type_name for type_name, (_, _, ext) in _SERIALIZERS.items()}


from .utils import flatten_dict, next_run_id, list_run_dirs, _atomic_write

class ExperimentReader:
    def __init__(self, experiment_name, base_dir="./data"):
        self.experiment_name = experiment_name
        self.exp_dir = Path(base_dir)/experiment_name
        
    def next_run_id(self):
        return next_run_id(self.exp_dir)

    def list_runs(self):
        """Run ids that have started logging metrics, sorted by run index.
        Returns [] if the experiment directory does not exist yet."""
        return [d for d in list_run_dirs(self.exp_dir)
                if (self.exp_dir/d/"metrics.jsonl").exists()]

    def load_metrics(self, run_id):
        rows = read_jsonl(self.exp_dir / run_id / "metrics.jsonl")
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["step", "metric", "value"])

    def get_metrics_stream(self,run_id = None):
        if not run_id: 
            run_id = self.list_runs()[-1]
        return MetricsStream(self.exp_dir/run_id)

    
    def load_config(self, run_id):
        with open(self.exp_dir/run_id/"config.json", 'r') as f:
            return json.load(f)
    
    def update_config(self, run_id, new_config):
        """
        Update the configuration of a specific run with new values.
        """
        config_path = self.exp_dir/run_id/"config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Shallow update: a top-level key in new_config replaces the whole value.
        config.update(new_config)
        # Same temp-file + rename as ConfigWriter, so a concurrent reader never sees a partial file.
        _atomic_write(config_path, json.dumps(config, indent=4))


    def list_artifact_groups(self, run_id):
        d = self.exp_dir / run_id / "artifacts"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


    def list_artifacts(self, run_id, group=None):
        d = self.exp_dir / run_id / "artifacts"
        index_path = (d if group is None else d / group) / "index.jsonl"
        rows = read_jsonl(index_path)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["step", "name", "file", "type"])

    def load_artifact(self, run_id, artifact_name, group=None):
        """
        Load an artifact from a specific run and group. 
        The artifact is loaded based on its file extension.
        """
        d = self.exp_dir / run_id / "artifacts"
        path = (d if group is None else d / group) / artifact_name
        if not path.exists():
            raise FileNotFoundError(f"No artifact at {path}")
        read_fn = _EXT_TO_READ_FN.get(path.suffix)
        if read_fn is None:
            raise ValueError(f"Unsupported artifact format: {path.suffix}")
        return read_fn(path)
 
    def summarize_runs(self, depth_names = 1):
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
        df = pd.DataFrame(rows)

        if df.empty:
            return pd.DataFrame(columns=["run_id"])

        if len(df) == 1:
            return df[["run_id"]]

        # keep only parameters that vary
        summary = df.loc[:, df.nunique(dropna=False) > 1]

        # Change names of columns to keep only a depth of the hierarchy of parameters
        summary.columns = [
            ".".join(col.rsplit(".", depth_names)[-depth_names:])
            for col in summary.columns
        ]

        # Make run_id the first column
        cols = summary.columns.tolist()
        cols.insert(0, cols.pop(cols.index("run_id")))
        summary = summary[cols]

        return summary


import numpy as np
import pandas as pd
import os

class ArtifactWriter:
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "artifacts")
        self.index_path = os.path.join(self.path, "index.csv")
        self.index = []

    def save_tensor(self, step, tensor, name=None):
        fname = f"{name}_step_{step}.npy" if name else f"step_{step}.npy"
        path = os.path.join(self.path, fname)
        np.save(path, tensor)
        self.index.append({"step": step, "file": fname})


    def flush(self):
        if not self.index:
            return

        df = pd.DataFrame(self.index)

        try:
            existing = pd.read_csv(self.index_path)
            df = pd.concat([existing, df])
        except FileNotFoundError:
            pass

        df.to_csv(self.index_path, index=False)
        self.index = []

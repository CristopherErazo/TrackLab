import numpy as np
import pandas as pd
import pickle  
import os

class ArtifactWriter:
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "artifacts")
        self.index_path = os.path.join(self.path, "index.csv")
        self.index = []

    def save_tensor(self, tensor, step=None, name=None):
        if step is not None:
            fname = f"{name}_step_{step}.npy" if name else f"step_{step}.npy"
        else:
            fname = f"{name}.npy" if name else "data.npy"
        path = os.path.join(self.path, fname)
        np.save(path, tensor)
        self.index.append({"step": step, "file": fname})
    
    def save_pickle(self, data, step=None, name=None):
        if step is not None:
            fname = f"{name}_step_{step}.pkl" if name else f"step_{step}.pkl"
        else:
            fname = f"{name}.pkl" if name else "data.pkl"
        path = os.path.join(self.path, fname)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
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

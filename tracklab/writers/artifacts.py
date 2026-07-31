import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path

from ..utils import append_jsonl, read_jsonl

# name -> (write_fn, read_fn, extension). Extend this dict, never the call sites.
_SERIALIZERS = {
    "tensor": (lambda obj, p: np.save(p, obj),
               lambda p: np.load(p),
               ".npy"),
    "torch":  (lambda obj, p: torch.save(obj, p),
               lambda p: torch.load(p, map_location="cpu"),
               ".pt"),
    "pickle": (lambda obj, p: pickle.dump(obj, open(p, "wb")),
               lambda p: pickle.load(open(p, "rb")),
               ".pkl"),
}



class _ArtifactGroup:
    """One subfolder + one index.csv. Created lazily on first use."""
    def __init__(self, dir_path: Path):
        self.dir = dir_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"
        self.buffer = []

    def _fname(self, name, step, ext):
        stem = "_".join(filter(None, [name, f"step_{step}" if step is not None else None]))
        return f"{stem or 'data'}{ext}"

    def save(self, data, name, step, type):
        """
        Save an artifact to disk and add its metadata to the buffer.
        The actual flush to disk happens in flush().
        type must be one of the keys in _SERIALIZERS: 'tensor', 'torch', or 'pickle'.
        """
        write_fn, _, ext = _SERIALIZERS[type]
        fname = self._fname(name, step, ext)
        write_fn(data, self.dir / fname)
        self.buffer.append({"step": step, "name": name, "file": fname, "type": type})

    def load(self, name, step, type):
        _, read_fn, ext = _SERIALIZERS[type]
        return read_fn(self.dir / self._fname(name, step, ext))

    def flush(self):
        if not self.buffer:
            return
        append_jsonl(self.index_path, self.buffer)
        self.buffer = []

    def index_df(self) -> pd.DataFrame:
        rows = read_jsonl(self.index_path)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["step", "name", "file", "type"])




class ArtifactWriter:
    """Write artifacts to disk in a structured way, with an index.csv for each group. 
    Each group is a subfolder under the root artifacts folder."""
    def __init__(self, run_dir):
        self.root = Path(run_dir) / "artifacts"
        self._groups: dict[str, _ArtifactGroup] = {}

    def _group(self, group: str | None) -> _ArtifactGroup:
        if group not in self._groups:
            # None -> the artifacts root itself; anything else -> a real subfolder
            dir_path = self.root if group is None else self.root / group
            self._groups[group] = _ArtifactGroup(dir_path)
        return self._groups[group]

    def save(self, data, group=None, name="", step=None, type="tensor"):
        """
        Save an artifact to the specified group. If group is None, save to the root artifacts folder.
        The artifact is saved with the specified name, step, and type.
        type must be one of the keys in _SERIALIZERS: 'tensor', 'torch', or 'pickle'.
        """
        self._group(group).save(data, name, step, type)

    def load(self, group=None, name="", step=None, type="tensor"):
        return self._group(group).load(name, step, type)

    def flush(self):
        for g in self._groups.values():
            g.flush()


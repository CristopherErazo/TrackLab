import os
import json
from pathlib import Path


def append_jsonl(path: Path, rows: list[dict]):
    """Append rows to a .jsonl file, one JSON object per line. Line-buffered
    so each row is flushed to disk as a complete line -- a concurrent reader
    (e.g. the dashboard) never sees a half-written row, only some whole
    number of complete lines followed by EOF."""
    with open(path, "a", buffering=1) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all complete rows from a .jsonl file. Skips a line that fails to
    parse -- this only happens if the file is read at the exact moment a
    write is in progress on a filesystem that doesn't honor local append
    atomicity (irrelevant for a single local disk, cheap insurance
    otherwise)."""
    if not Path(path).exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def create_run_dir(exp_dir, run_id, artifacts):
    """Creates a directory for the run inside the experiment directory. The directory is named using the run ID."""
    # path = os.path.join(exp_dir, run_id)
    path = Path(exp_dir) / run_id
    os.makedirs(path, exist_ok=True)
    # create artifacts subdir if needed
    if artifacts:
        os.makedirs(path/"artifacts", exist_ok=True)
    return path

def next_run_id(exp_dir):
    """Scans the experiment directory for existing runs and returns the next run ID which is one more than the maximum existing run ID."""

    # If the experiment directory doesn't exist, we can start with run_001
    if not os.path.exists(exp_dir):
        n = 1
    else:
        existing = [
            int(d.split("_")[1])
            for d in os.listdir(exp_dir)
            if d.startswith("run_")
        ]

        n = max(existing, default=0) + 1
    return f"run_{n:03d}"

def flatten_dict(d : dict, parent="", sep="."):
    """Flattens a nested dictionary into a single level dictionary with keys representing the path to each value."""
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out

def _atomic_write(path: Path, data: str):
    """Write to a temp file then rename into place, so a reader (the
    dashboard, polling once a second) never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)
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
    """Creates the directory for a single run_id inside exp_dir.

    Raises FileExistsError if run_id is already taken, instead of silently
    succeeding (the previous exist_ok=True let two callers "create" the
    same directory, interleaving two runs' data into one folder). Callers
    that need a *guaranteed-unique* run_id -- which is every real caller --
    should go through claim_run_dir() below, not call this directly; this
    function only knows how to create one specific, already-decided id, not
    how to pick a safe one.
    """
    path = Path(exp_dir) / run_id
    os.makedirs(path, exist_ok=False)
    if artifacts:
        os.makedirs(path / "artifacts", exist_ok=True)
    return path


def next_run_id(exp_dir):
    """Scans exp_dir and returns "the next run id, as of this scan" -- a
    best-effort SUGGESTION, not a reservation. Two calls made close
    together (two processes, or two rapid calls in one process) can return
    the same value; nothing here claims it. Kept as a standalone function
    because ExperimentReader.next_run_id() legitimately wants this exact
    "best guess" behavior for display/prediction purposes where a race
    doesn't matter. For actually creating a new run, use claim_run_dir(),
    which retries this scan until a creation actually succeeds.
    """
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


def claim_run_dir(exp_dir, artifacts):
    """Atomically claim a fresh run_id: retries the next_run_id() + 
    create_run_dir() combination until a directory creation actually
    succeeds, instead of trusting a single scan-then-create as if it were a
    safe reservation. This is what closes the race next_run_id() alone
    cannot: if two callers scan at the same moment and both compute
    "run_004", only one of their create_run_dir(..., "run_004") calls can
    win (FileExistsError on the other) -- the loser just tries "run_005"
    next, rather than silently overwriting/merging into the winner's
    directory.

    Returns
    -------
    (run_id, run_dir) : tuple[str, Path]
    """
    os.makedirs(exp_dir, exist_ok=True)
    while True:
        run_id = next_run_id(exp_dir)
        try:
            path = create_run_dir(exp_dir, run_id, artifacts)
            return run_id, path
        except FileExistsError:
            continue  # someone else claimed this id between our scan and our create; try the next one


def flatten_dict(d: dict, parent="", sep="."):
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
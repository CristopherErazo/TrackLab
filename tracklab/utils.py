import os

def create_run_dir(exp_dir, run_id, artifacts):
    """Creates a directory for the run inside the experiment directory. The directory is named using the run ID."""
    path = os.path.join(exp_dir, run_id)
    os.makedirs(path, exist_ok=True)
    # create artifacts subdir if needed
    if artifacts:
        os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
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
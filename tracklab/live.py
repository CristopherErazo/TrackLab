# tracklab/live.py
import json
from pathlib import Path

import pandas as pd


class MetricsStream:
    """Incremental JSONL reader for a live dashboard: each poll() call only
    parses bytes appended since the previous call."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.status_path = self.run_dir / "status.json"

        self._offset = 0
        self._rows = []

    def poll(self) -> pd.DataFrame:
        """Return a DataFrame of all metrics logged so far, parsing only new lines since the last call."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                f.seek(self._offset)
                new_text = f.read()
            if new_text:
                complete, _, _partial = new_text.rpartition("\n")  # keep any trailing unfinished line for next poll
                for line in complete.splitlines():
                    try:
                        self._rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                self._offset += len(complete) + 1  # +1 for the newline we split on
        return pd.DataFrame(self._rows) if self._rows else pd.DataFrame(columns=["step", "metric", "value"])

    def get_status(self) -> dict:
        """Read the current status from the json file. Returns an empty dict if no status has been written yet."""
        if not self.status_path.exists():
            return {}
        try:
            return json.loads(self.status_path.read_text())
        except json.JSONDecodeError:
            return {}  # caught the file mid-write; try again next poll

    def done(self):
        if not self.status_path.exists():
            return False

        with open(self.status_path) as f:
            status = json.load(f)

        return status.get("done", False)
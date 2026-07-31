from pathlib import Path
import pandas as pd

from ..utils import append_jsonl 

import time

class MetricsWriter:
    """
    Writer for metrics to a JSONL file. 
    Each row is a JSON object with keys: step, metric, value, note (optional), 
    and any additional tags provided.
    """
    def __init__(self, run_dir, min_flush_interval=0.0):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.buffer = []
        self.min_flush_interval = min_flush_interval # minimum time in seconds between flushes
        self._last_flush = 0.0

    def track(self, step, metrics, note=None, **tags):
        """
        Track metrics at a given step. Metrics should be a dictionary of metric names to values.
        note is an optional string that can be used to annotate the metrics. 
        Additional tags can be provided as keyword arguments.
        """
        for k, v in metrics.items():
            row = {"step": step, "metric": k, "value": v}
            if note is not None:
                row["note"] = note
            row.update(tags)
            self.buffer.append(row)

    def flush(self, force=False):
        """Flush the buffered metrics to disk. If force is True, flush regardless of the min_flush_interval."""
        if not self.buffer:
            return
        if not force and time.monotonic() - self._last_flush < self.min_flush_interval:
            return
        append_jsonl(self.path, self.buffer)
        self._last_flush = time.monotonic()
        self.buffer = []


import pandas as pd


class MetricsWriter:
    def __init__(self, run_dir):
        self.path = f"{run_dir}/metrics.csv"
        self.buffer = []

    def track(self, step, metrics: dict, note=None, **tags):
        for k, v in metrics.items():
            row = {"step": step, "metric": k, "value": v}
            if note is not None:
                row["note"] = note
            row.update(tags)          # e.g. branch_id="root_fork@4137"
            self.buffer.append(row)

    def flush(self):
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        try:
            existing = pd.read_csv(self.path)
            df = pd.concat([existing, df])
        except FileNotFoundError:
            pass

        df.to_csv(self.path, index=False)
        self.buffer = []

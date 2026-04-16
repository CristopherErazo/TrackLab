import pandas as pd


class MetricsWriter:
    def __init__(self, run_dir):
        self.path = f"{run_dir}/metrics.csv"
        self.buffer = []

    def track(self, step, metrics: dict):
        for k, v in metrics.items():
            self.buffer.append({
                "step": step,
                "metric": k,
                "value": v
            })

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

import logging
from .utils import create_run_dir, next_run_id
from .writers.config import ConfigWriter
from .writers.metrics import MetricsWriter
from .writers.artifacts import ArtifactWriter
from .logger import create_run_logger



class Run:
    def __init__(self, 
                 config,    
                 exp_dir,
                 artifacts=False):
            
        self.run_id = next_run_id(exp_dir)
        self.run_dir = create_run_dir(exp_dir, self.run_id, artifacts)

        # writers
        self.metrics = MetricsWriter(self.run_dir)
        self.config = ConfigWriter(self.run_dir)
        if artifacts:
            self.artifacts = ArtifactWriter(self.run_dir)        

        # save config immediately
        self.config.save(config)

    def track_metric(self, step, note=None, tags=None, **metrics):
        self.metrics.track(step, metrics, note=note, **(tags or {}))

    def finalize(self):
        self.metrics.flush()
        if hasattr(self, "artifacts"):
            self.artifacts.flush()

    def track_artifact(self, data, step = None, name='', type='tensor'):
        if hasattr(self, "artifacts"):
            if type == 'tensor':
                self.artifacts.save_tensor(data, step, name)
            elif type == 'pickle':
                self.artifacts.save_pickle(data, step, name)
            else:
                raise ValueError(f"Unsupported artifact type: {type}")

   # to finish...
    def load_artifact(self, name='', step=None, type='tensor'):
        if not hasattr(self, "artifacts"):
            raise RuntimeError("Run was created with artifacts=False")
        if type == 'tensor':
            return self.artifacts.load_tensor(step, name)
        elif type == 'pickle':
            return self.artifacts.load_pickle(step, name)
        else:   
            raise ValueError(f"Unsupported artifact type: {type}")
    
    def get_logger(self, log_to_terminal=True, log_to_file=True, level=logging.INFO, log_format="%(asctime)s - %(levelname)s - %(message)s"):
        return create_run_logger(self.run_dir, self.run_id, log_to_terminal, log_to_file, level, log_format)

import json
from pathlib import Path
from omegaconf import OmegaConf
import logging
from .utils import claim_run_dir  # atomic: replaces the old next_run_id()+create_run_dir() pair
from .writers.config import ConfigWriter
from .writers.metrics import MetricsWriter
from .writers.artifacts import ArtifactWriter
from .logger import create_run_logger



class Run:
    def __init__(self, 
                 config : dict | OmegaConf,    
                 exp_dir : Path,
                 artifacts : bool =False,
                 min_flush_interval : float = 0.0):

        # claim_run_dir retries internally until it wins a genuinely unused
        # run_id -- see tracklab.utils for why this replaced the previous
        # next_run_id(exp_dir) + create_run_dir(exp_dir, run_id) pair, which
        # had a gap between "decide the id" and "create the directory" that
        # two near-simultaneous Run() constructions could both walk through.
        self.run_id, self.run_dir = claim_run_dir(exp_dir, artifacts)
        self._finalized = False
        # self.status_path = self.run_dir / "status.json"

        # writers
        self.metrics = MetricsWriter(self.run_dir,min_flush_interval=min_flush_interval)
        self.config = ConfigWriter(self.run_dir)
        if artifacts:
            self.artifacts = ArtifactWriter(self.run_dir)        

        # save config immediately
        self.config.save(config)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()
        return False  # never swallow the exception

    def track_metric(self, step, note=None, tags=None, **metrics):
        """
        Track metrics at a given step. Metrics should be provided as keyword arguments.
        note is an optional string that can be used to annotate the metrics.
        tags is an optional dictionary of additional tags to include with the metrics.
        """
        self.metrics.track(step, metrics, note=note, **(tags or {}))
        self.metrics.flush()

    def track_artifact(self, data, step=None, group=None, name='', type='tensor'):
        """
        Track an artifact by saving it to disk and adding its metadata to the index.
        Parameters:
            data: the artifact data to save
            step: the step at which the artifact is saved
            group: the group under which to save the artifact (subfolder)
            name: the name of the artifact
            type: the type of the artifact, must be one of the keys in _SERIALIZERS: 'tensor', 'torch', or 'pickle'.
        """
        if hasattr(self, "artifacts"):
            self.artifacts.save(data, group=group, name=name, step=step, type=type)
            self.artifacts.flush()

    def finalize(self):
        if self._finalized:
            return
        self.metrics.flush()
        if hasattr(self, "artifacts"):
            self.artifacts.flush()
        self._close_logger_handlers()   
        self._finalized = True

    def _close_logger_handlers(self):
        logger = logging.getLogger(self.run_id)
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def load_artifact(self, group=None, name='', step=None, type='tensor'):
        if not hasattr(self, "artifacts"):
            raise RuntimeError("Run was created with artifacts=False")
        return self.artifacts.load(group=group, name=name, step=step, type=type)
    
    def get_logger(self, log_to_terminal=True, log_to_file=True, 
                   level=logging.INFO, log_format="%(asctime)s - %(levelname)s - %(message)s"):
        return create_run_logger(self.run_dir, self.run_id, log_to_terminal, log_to_file, level, log_format)


    # def set_status(self, **kwargs):
    #     """Write a status update to a json file."""
    #     _atomic_write(self.status_path, json.dumps(kwargs))
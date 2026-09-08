# TrackLab

A small, file-based experiment tracker. Each run writes its config, metrics, artifacts and logs as plain files into a folder; a reader loads them back as pandas DataFrames. No server, no database.

## Installation

Requires Python 3.10+. The latest stable release is `v1.1.0`.

**Use it in your project, without cloning.** With [uv](https://docs.astral.sh/uv/):

```bash
uv add "tracklab @ git+https://github.com/CristopherErazo/TrackLab.git@v1.1.0"
```

With pip:

```bash
pip install "tracklab @ git+https://github.com/CristopherErazo/TrackLab.git@v1.1.0"
```

Add `[torch]` after `tracklab` in either command to also install torch, which is only needed for `type='torch'` artifacts.

**Clone it, to run the example or develop.**

```bash
git clone --branch v1.1.0 https://github.com/CristopherErazo/TrackLab.git
cd TrackLab
uv sync                 # creates .venv with numpy, omegaconf, pandas
uv sync --extra torch   # optional
```

Drop `--branch v1.1.0` to get the development version on `main`.

## Data layout

```
data/<experiment_name>/run_<NNN>/
├── config.json          # saved when the run starts
├── metrics.jsonl        # one line per metric per step: {"step": 3, "metric": "loss", "value": 0.42}
├── artifacts/           # only if start_run(artifacts=True)
│   ├── index.jsonl      # {"step", "name", "file", "type"} per artifact
│   └── weights_step_10.npy
└── logs/
    ├── info.log
    └── error.log
```

Run folders are numbered `run_001`, `run_002`, ... and are claimed safely when several processes start runs at once. Metrics are appended one whole line at a time and `config.json` is written atomically, so a live reader never sees a partial file.

## Writing a run

```python
from tracklab import ExperimentTracker

exp = ExperimentTracker("my_experiment")        # writes to ./data/my_experiment/

with exp.start_run({"lr": 0.1, "n_steps": 30}, artifacts=True) as run:
    log = run.get_logger()

    for step in range(30):
        loss, acc = ...                         # your training step
        run.track_metric(step, loss=loss, accuracy=acc)

        if step % 10 == 0:
            run.track_artifact(weights, step=step, name="weights")   # numpy array -> .npy

    log.info("done")
```

Leaving the `with` block flushes everything and closes the log files. The config can be a dict or an OmegaConf `DictConfig`. `track_metric` accepts an optional `note=` string and `tags=` dict that are stored with the rows.

## Reading a run back

```python
from tracklab import ExperimentReader

reader = ExperimentReader("my_experiment")

reader.list_runs()                                   # ['run_001', 'run_002', ...]
reader.load_config("run_001")                        # dict
df = reader.load_metrics("run_001")                  # columns: step, metric, value
wide = df.pivot(index="step", columns="metric", values="value")

reader.list_artifacts("run_001")                     # DataFrame: step, name, file, type
w = reader.load_artifact("run_001", "weights_step_10.npy")

reader.summarize_runs()                              # one row per run, only the config fields that differ
```

To follow a run while it is still training, poll it incrementally:

```python
stream = reader.get_metrics_stream("run_001")   # defaults to the latest run
df = stream.poll()                              # call repeatedly; only new lines are parsed
```

## Artifacts

| `type`     | Saved with    | File   |
| ---------- | ------------- | ------ |
| `'tensor'` | `numpy.save`  | `.npy` |
| `'torch'`  | `torch.save`  | `.pt`  |
| `'pickle'` | `pickle.dump` | `.pkl` |

Files are named `<name>_step_<n>.<ext>`, or `<name>.<ext>` without a step. Pass `group="..."` to put artifacts in a subfolder with its own index. To add a format, add one entry to `_SERIALIZERS` in `tracklab/writers/artifacts.py`.

## Example and tests

```bash
uv run scripts/example.py                       # writes a fake run, then reads it back
uv run scripts/example.py sleep=0 train.lr=0.05  # OmegaConf command-line overrides
uv run --group dev pytest
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

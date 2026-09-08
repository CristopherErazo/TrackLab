# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TrackLab is a small, dependency-light experiment-tracking library (a local, file-based stand-in for W&B/MLflow). It writes structured run data — config, metrics, artifacts, logs — into a plain directory tree under `./data/` and provides a reader side for analysis and live dashboards.

## Commands

The project is managed with [uv](https://docs.astral.sh/uv/); all metadata lives in `pyproject.toml` (no `requirements.txt`). Runtime deps: numpy, omegaconf, pandas. `torch` is an optional extra.

```bash
uv sync --group dev               # create .venv, install tracklab (editable) + pytest
uv sync --group dev --extra torch # ...plus torch, for type='torch' artifacts
uv run pytest                     # lightweight suite in tests/, ~2s, writes only to tmp_path
uv run scripts/example.py         # end-to-end demo: fake training run, then reads it back (writes ./data/my_experiment/run_XXX)
uv run scripts/example.py sleep=0 experiment_name=quick_check    # fast smoke test (~1s)
uv run scripts/example.py train.lr=0.05 train.n_steps=100        # OmegaConf CLI overrides
uv build                          # sdist + wheel into dist/ (uv_build backend, flat layout via module-root = "")
```

**Releasing**: bump `version` in `pyproject.toml`, run `uv lock`, commit, then `git tag -a vX.Y.Z -m "..."` and `git push origin main vX.Y.Z`. The README installation section pins the latest tag; update it in the same commit. Existing tags: `v0.1.0`, `v0.2.0` (on origin), `v1.0.0` (local only), `v1.1.0`.

Tests live in `tests/test_tracklab.py` (pytest config is in `pyproject.toml` under `[tool.pytest.ini_options]`). They are deliberately small: one test per guarantee (finalize flushes, per-experiment loggers, unique concurrent run ids, partial-line tolerance in `MetricsStream`, artifact round-trips, optional torch, `summarize_runs`). Add a test when fixing a bug; keep the suite fast. `scripts/example.py` remains the end-to-end smoke check: it writes a run (`train()`) and then reads it back with `ExperimentReader` (`analyze()`); pass `sleep=0` for a fast check.

`torch` is an **optional** dependency (`pip install -e .[torch]`): it is imported lazily inside the `torch` serializer in `tracklab/writers/artifacts.py`, so `import tracklab` and the `tensor`/`pickle` artifact types work without it. Only `type='torch'` artifacts require it and raise a clear `ImportError` otherwise.

## Architecture

Two entry points, exported from `tracklab/__init__.py`:

- **Write path**: `ExperimentTracker(experiment_name, base_dir="./data")` → `.start_run(config, artifacts=..., min_flush_interval=...)` → a `Run`.
- **Read path**: `ExperimentReader(experiment_name, base_dir="./data")` for post-hoc analysis, plus `MetricsStream` (`tracklab/live.py`) for tailing an in-progress run.

`Run` (`tracklab/run.py`) owns one run directory and delegates all I/O to three writers in `tracklab/writers/`:

| Writer | File(s) | Format |
| --- | --- | --- |
| `ConfigWriter` | `config.json` | atomic write; unwraps OmegaConf `DictConfig` |
| `MetricsWriter` | `metrics.jsonl` | append-only, one **long-format** row per metric: `{step, metric, value, note?, ...tags}` |
| `ArtifactWriter` | `artifacts/[group/]<name>_step_<n>.<ext>` + per-group `index.jsonl` | pluggable serializers |

`Run` is a context manager (`with exp.start_run(...) as run:`); `__exit__` calls `finalize()`, which flushes writers and closes the run's logging handlers. `finalize()` is idempotent, so the explicit call in `scripts/example.py` inside the `with` block is harmless.

`get_logger()` (`tracklab/logger.py`) returns a stdlib logger named after `run_id`, with `LevelFilter` splitting messages into `logs/info.log` and `logs/error.log` — a record lands in exactly one file by exact level match, so levels other than INFO/ERROR are silently dropped from files.

### Data layout

```
data/<experiment_name>/run_<NNN>/
├── config.json
├── metrics.jsonl
├── artifacts/            # only if start_run(artifacts=True)
│   ├── index.jsonl       # root group
│   ├── test_step_0.npy
│   └── <group>/{index.jsonl, ...}
└── logs/{info.log, error.log}
```

## Conventions that matter

**Concurrency-safety is a deliberate design theme.** Multiple runs (and a live dashboard reader) are expected to touch the same tree simultaneously, so:

- Run ids are claimed via `claim_run_dir()` in `tracklab/utils.py`, which loops `next_run_id()` + `create_run_dir(exist_ok=False)` until a creation wins. Never call `create_run_dir()` or use `next_run_id()` as a reservation in the write path — `next_run_id()` is a best-effort guess kept only for `ExperimentReader.next_run_id()`'s display use.
- Append-only data goes through `append_jsonl()` (line-buffered, so a concurrent reader sees only whole lines); whole-file rewrites go through `_atomic_write()` (temp file + `os.replace`). Prefer JSONL over CSV/pickle for anything a reader might poll.
- Readers tolerate partial lines: `read_jsonl()` skips unparseable lines and `MetricsStream.poll()` retains a trailing partial line for the next poll.

**Adding an artifact format**: add an entry to `_SERIALIZERS` in `tracklab/writers/artifacts.py` (`name -> (write_fn, read_fn, extension)`). Call sites and `reader._EXT_TO_READ_FN` derive from that dict — don't special-case types elsewhere.

**Metrics are long-format, not wide.** `ExperimentReader.load_metrics()` returns the raw rows; pivot in the consumer rather than changing the on-disk shape.

**`summarize_runs()`** builds a run-comparison table by flattening each `config.json` and keeping only columns that vary across runs, shortening dotted keys to the last `depth_names` segments. It has explicit early returns for 0 and 1 runs — preserve those when editing.

## Documentation

`README.md` is the user-facing reference and was rewritten against the current code (`ExperimentTracker`, `.jsonl` files, data-first `track_artifact(data, step=None, group=None, name='', type='tensor')`). When changing public API, update the README, `scripts/example.py`, and the tests together.

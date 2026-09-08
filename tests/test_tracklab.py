"""Lightweight regression tests for TrackLab.

Everything writes into pytest's tmp_path, never into ./data. Run with:
    python -m pytest
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tracklab import ExperimentReader, ExperimentTracker
from tracklab.live import MetricsStream
from tracklab.writers.artifacts import _SERIALIZERS


# --- metrics -----------------------------------------------------------------

def test_finalize_flushes_despite_min_flush_interval(tmp_path):
    """Regression: with a large min_flush_interval the throttled flush used to
    return early at finalize and the buffered tail of the run was lost."""
    exp = ExperimentTracker("e", base_dir=tmp_path)
    with exp.start_run({"lr": 0.1}, min_flush_interval=60.0) as run:
        for step in range(5):
            run.track_metric(step, loss=float(step), acc=1.0)

    df = ExperimentReader("e", base_dir=tmp_path).load_metrics(run.run_id)
    assert len(df) == 10  # 5 steps x 2 metrics, long format
    assert set(df["metric"]) == {"loss", "acc"}


def test_metrics_row_shape(tmp_path):
    exp = ExperimentTracker("e", base_dir=tmp_path)
    with exp.start_run({}) as run:
        run.track_metric(3, note="hi", tags={"id": "x"}, loss=0.5)

    rows = [json.loads(l) for l in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows == [{"step": 3, "metric": "loss", "value": 0.5, "note": "hi", "id": "x"}]


# --- logging -----------------------------------------------------------------

def test_loggers_are_isolated_per_experiment(tmp_path):
    """Regression: two experiments both at run_001 shared a process-global
    logger, so the second run's messages went into the first run's files."""
    a = ExperimentTracker("exp_a", base_dir=tmp_path).start_run({})
    b = ExperimentTracker("exp_b", base_dir=tmp_path).start_run({})
    assert a.run_id == b.run_id  # same run number, different experiments

    la = a.get_logger(log_to_terminal=False)
    lb = b.get_logger(log_to_terminal=False)
    assert la is not lb
    la.info("from A")
    lb.info("from B")
    lb.error("B error")
    a.finalize()
    b.finalize()

    assert "from A" in (a.run_dir / "logs/info.log").read_text()
    assert "from B" not in (a.run_dir / "logs/info.log").read_text()
    assert "from B" in (b.run_dir / "logs/info.log").read_text()
    assert "B error" in (b.run_dir / "logs/error.log").read_text()
    assert not la.handlers and not lb.handlers  # finalize closed them


# --- run ids -----------------------------------------------------------------

def test_concurrent_runs_get_unique_ids(tmp_path):
    exp = ExperimentTracker("e", base_dir=tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(lambda _: exp.start_run({}), range(16)))
    for r in runs:
        r.finalize()
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == 16
    assert sorted(ids) == [f"run_{i:03d}" for i in range(1, 17)]


def test_stray_dirs_in_experiment_folder_are_ignored(tmp_path):
    exp_dir = tmp_path / "e"
    exp_dir.mkdir()
    (exp_dir / "run_backup").mkdir()   # used to crash next_run_id with ValueError
    (exp_dir / "notes.txt").write_text("")
    (exp_dir / "run_002").mkdir()
    run = ExperimentTracker("e", base_dir=tmp_path).start_run({})
    run.finalize()
    assert run.run_id == "run_003"


# --- reader ------------------------------------------------------------------

def test_list_runs_sorted_and_tolerates_missing_dir(tmp_path):
    reader = ExperimentReader("e", base_dir=tmp_path)
    assert reader.list_runs() == []   # experiment dir does not exist yet
    exp = ExperimentTracker("e", base_dir=tmp_path)
    for _ in range(11):
        with exp.start_run({}) as run:
            run.track_metric(0, x=1.0)
    assert reader.list_runs() == [f"run_{i:03d}" for i in range(1, 12)]


def test_update_config_is_atomic_and_shallow(tmp_path):
    exp = ExperimentTracker("e", base_dir=tmp_path)
    with exp.start_run({"a": 1, "nested": {"x": 1, "y": 2}}) as run:
        pass
    reader = ExperimentReader("e", base_dir=tmp_path)
    reader.update_config(run.run_id, {"a": 2, "note": "hi"})
    assert reader.load_config(run.run_id) == {"a": 2, "nested": {"x": 1, "y": 2}, "note": "hi"}
    assert not (run.run_dir / "config.json.tmp").exists()


# --- live stream -------------------------------------------------------------

def test_metrics_stream_keeps_partial_line_for_next_poll(tmp_path):
    path = tmp_path / "metrics.jsonl"
    stream = MetricsStream(tmp_path)

    path.write_text('{"step": 0, "metric": "loss", "value": 1.0}\n{"step": 1, "metric": "lo')
    assert len(stream.poll()) == 1  # only the complete line

    with open(path, "a") as f:
        f.write('ss", "value": 0.5}\n')
    df = stream.poll()
    assert len(df) == 2
    assert df["value"].tolist() == [1.0, 0.5]


# --- artifacts ---------------------------------------------------------------

def test_artifact_roundtrip_numpy_and_pickle(tmp_path):
    exp = ExperimentTracker("e", base_dir=tmp_path)
    with exp.start_run({}, artifacts=True) as run:
        run.track_artifact(np.arange(3), step=0, name="arr")
        run.track_artifact({"k": 1}, step=2, name="d", group="g", type="pickle")

    reader = ExperimentReader("e", base_dir=tmp_path)
    assert reader.load_artifact(run.run_id, "arr_step_0.npy").tolist() == [0, 1, 2]
    assert reader.load_artifact(run.run_id, "d_step_2.pkl", group="g") == {"k": 1}
    assert reader.list_artifacts(run.run_id)["file"].tolist() == ["arr_step_0.npy"]
    assert reader.list_artifact_groups(run.run_id) == ["g"]


def test_torch_serializer_gives_clear_error_when_torch_missing(tmp_path, monkeypatch):
    """torch is optional: import tracklab must not need it, and asking for a
    torch artifact without it must fail with an actionable ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise
    save, load, _ = _SERIALIZERS["torch"]
    with pytest.raises(ImportError, match="requires torch"):
        save([1, 2], tmp_path / "x.pt")
    with pytest.raises(ImportError, match="requires torch"):
        load(tmp_path / "x.pt")


def test_summarize_runs_keeps_only_varying_params(tmp_path):
    exp = ExperimentTracker("e", base_dir=tmp_path)
    for lr in (0.1, 0.2):
        with exp.start_run({"model": {"lr": lr, "depth": 3}, "seed": 0}) as run:
            run.track_metric(0, loss=1.0)

    summary = ExperimentReader("e", base_dir=tmp_path).summarize_runs()
    assert summary.columns.tolist() == ["run_id", "lr"]  # depth and seed are constant
    assert sorted(summary["lr"]) == [0.1, 0.2]

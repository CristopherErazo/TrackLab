"""End-to-end TrackLab example: write a run, then read it back.

Run it as is, or override any config field from the command line (OmegaConf):

    python scripts/example.py
    python scripts/example.py train.lr=0.05 train.n_steps=100
    python scripts/example.py sleep=0 experiment_name=quick_check   # fast smoke test

Run it twice with different `train.lr` values and the summary table at the end
will show which parameters differ between the runs.

Everything is written under ./data/<experiment_name>/run_NNN/.
"""
from dataclasses import dataclass, field
import time

import numpy as np
from omegaconf import OmegaConf

from tracklab import ExperimentReader, ExperimentTracker


# --- 1. Config -----------------------------------------------------------------
# Structured dataclasses give OmegaConf a schema, so a typo in a CLI override
# (e.g. train.lrr=0.1) is rejected instead of silently creating a new key.

@dataclass
class TrainConfig:
    lr: float = 0.1
    n_steps: int = 30
    seed: int = 0


@dataclass
class ExperimentConfig:
    experiment_name: str = "my_experiment"
    train: TrainConfig = field(default_factory=TrainConfig)
    artifact_every: int = 10   # save weights every k steps
    sleep: float = 0.1         # seconds per step, to make a live dashboard interesting


# --- 2. Fake training ----------------------------------------------------------

def fake_training_step(rng, step, lr, n_steps):
    """A decaying loss with noise; a larger lr converges faster."""
    progress = step / n_steps
    loss = float(0.1 + 0.9 * np.exp(-0.8 * lr * step) + 0.02 * rng.standard_normal())
    acc = float(1 - np.exp(-3 * progress) + 0.02 * rng.standard_normal())
    return max(loss, 0.0), min(max(acc, 0.0), 1.0)


def train(cfg):
    exp = ExperimentTracker(cfg.experiment_name)
    rng = np.random.default_rng(cfg.train.seed)
    weights = rng.standard_normal((4, 4))

    # start_run claims a fresh run_NNN directory and writes config.json immediately.
    # Leaving the `with` block finalizes the run: flushes metrics/artifacts, closes log files.
    with exp.start_run(cfg, artifacts=True) as run:
        log = run.get_logger(log_to_terminal=True, log_to_file=True)
        log.info(f"Started {run.run_id} in {run.run_dir}")

        prev_loss = float("inf")
        for step in range(cfg.train.n_steps):
            loss, acc = fake_training_step(rng, step, cfg.train.lr, cfg.train.n_steps)
            weights -= cfg.train.lr * rng.standard_normal(weights.shape)

            # Metrics are stored long-format: one row per metric per step.
            # `note` and `tags` are optional extras attached to every row of the call.
            run.track_metric(step, train_loss=loss, accuracy=acc)
            if acc > 0.9:
                run.track_metric(step, note="above target", tags={"phase": "converged"}, accuracy=acc)

            # Artifacts: any numpy array (type='tensor'), or a python object (type='pickle').
            # `group` puts them in a subfolder with its own index.jsonl.
            if step % cfg.artifact_every == 0:
                run.track_artifact(weights, step=step, name="weights", group="checkpoints")

            if step % 10 == 0:
                log.info(f"step {step:3d}  loss={loss:.3f}  acc={acc:.3f}")
            if loss > prev_loss + 0.03:   # errors go to logs/error.log, info to logs/info.log
                log.error(f"step {step}: loss spiked {prev_loss:.3f} -> {loss:.3f}")
            prev_loss = loss

            time.sleep(cfg.sleep)

        # A final artifact at the run root, without a step.
        run.track_artifact({"weights": weights, "final_loss": loss}, name="final", type="pickle")
        log.info("Finished")

    return run.run_id


# --- 3. Read it back -----------------------------------------------------------

def analyze(cfg, run_id):
    reader = ExperimentReader(cfg.experiment_name)

    print(f"\nRuns in '{cfg.experiment_name}':", reader.list_runs())
    print("Config of", run_id, "->", reader.load_config(run_id)["train"])

    # load_metrics gives the raw long-format rows; pivot to wide for plotting/analysis.
    metrics = reader.load_metrics(run_id)
    wide = metrics.pivot_table(index="step", columns="metric", values="value")
    print("\nMetrics (last 5 steps):")
    print(wide.tail().round(3).to_string())

    print("\nArtifact groups:", reader.list_artifact_groups(run_id))
    print("Checkpoints index:")
    print(reader.list_artifacts(run_id, group="checkpoints").to_string(index=False))
    last = reader.list_artifacts(run_id, group="checkpoints").iloc[-1]["file"]
    w = reader.load_artifact(run_id, last, group="checkpoints")
    print(f"Loaded {last}: shape={w.shape}, mean={w.mean():.3f}")
    final = reader.load_artifact(run_id, "final.pkl")
    print(f"Loaded final.pkl: keys={list(final)}, final_loss={final['final_loss']:.3f}")

    # One row per run, only the config fields that differ between runs.
    print("\nRun comparison (parameters that vary across runs):")
    print(reader.summarize_runs().to_string(index=False))


if __name__ == "__main__":
    cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.from_cli())
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    run_id = train(cfg)
    analyze(cfg, run_id)

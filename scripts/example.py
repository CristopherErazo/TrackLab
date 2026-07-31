import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass 
import time
from tracklab.experiment import ExperimentTracker

@dataclass
class Config:
    lr: float = 0.1
    n: int = 10
    experiment_name: str = "test_experiment"

@dataclass
class NestedConfig:
    config: Config
    param1: int = 5
    param2: str = "hello"
            

if __name__ == "__main__":

    config = NestedConfig(
        param1=10,
        param2="world",
        config=Config(
            lr=0.01,
            n=20,
            experiment_name="my_experiment"
        )
    )

    cfg = OmegaConf.merge(OmegaConf.structured(config), OmegaConf.from_cli())
    
    print("Final config:")
    print(OmegaConf.to_yaml(cfg))
    experiment_name = cfg.config.experiment_name
    print(f"Experiment name: {experiment_name}")

    # Create an experiment and run
    exp = ExperimentTracker(experiment_name)

    with exp.start_run(cfg, artifacts=True) as run:
        # Initialize logger for the run
        logger = run.get_logger(log_to_file=True, log_to_terminal=True)

        logger.info("Starting training")

        for step in range(30):
            # Simulate training and track results
            loss = np.random.random()
            acc = np.random.random()
            results = {"train_loss": loss, "accuracy": acc, "something_else": acc * 2}
            run.track_metric(step, **results)
            run.track_metric(step, note='hello', tags={'id':'1001'}, **results)

            # Log results 
            logger.info(f"Step {step} - Results: {results}")
            if acc > 0.5:
                logger.error(f"Accuracy is above 0.5 at step {step}!")
            
            # Track an artifact every 2 steps
            if step % 2 == 0:
                run.track_artifact(np.random.rand(3, 3),step,name='test')
                run.track_artifact({'data':np.random.rand(3, 3)},step,name='data',group='dic',type='pickle')

            time.sleep(1)  # Simulate time taken for training

        # Finalize the run to flush tracked metrics
        run.finalize()
        logger.info("Experiment completed")
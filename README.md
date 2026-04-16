# TrackLab

TrackLab is a Python package designed to manage and track experiments efficiently. It provides tools for logging, tracking metrics, saving artifacts, and organizing experiment data in a structured way. This makes it easier to analyze and reproduce experiments.

## Features
- **Experiment Management**: Create and manage experiments with unique identifiers.
- **Logging**: Log messages to both terminal and files, with support for different log levels (INFO, ERROR, etc.).
- **Metrics Tracking**: Track metrics over steps and save them in a structured format.
- **Artifact Management**: Save and organize artifacts (e.g., numpy arrays) generated during experiments.
- **Configurable Data Structure**: Automatically organizes experiment data into a clear directory structure.

---

## Installation

To use TrackLab, clone the repository and install the required dependencies:

```bash
git clone https://github.com/CristopherErazo/TrackLab.git
cd TrackLab
pip install -r requirements.txt
pip install .
```

---

## Usage

### Example Script
Below is an example of how to use TrackLab to manage an experiment:

```python
from tracklab.experiment import Experiment
import numpy as np

# Define your experiment configuration
config = {
    "lr": 0.01,
    "n": 20,
    "experiment_name": "my_experiment"
}

# Create an experiment
exp = Experiment(config["experiment_name"])
run = exp.start_run(config, artifacts=True)

# Initialize logger
logger = run.get_logger(log_to_file=True, log_to_terminal=False)
logger.info("Starting training")

# Simulate training
for step in range(10):
    loss = np.random.random()
    acc = np.random.random()
    results = {"train_loss": loss, "accuracy": acc}

    # Track metrics
    run.track_metric(step, **results)

    # Log results
    logger.info(f"Step {step} - Results: {results}")

    # Save artifacts every 2 steps
    if step % 2 == 0:
        run.track_artifact(step, np.random.rand(3, 3), 'test')

# Finalize the run
run.finalize()
logger.info("Experiment completed")
```

---

## Data Folder Structure

When you run an experiment, TrackLab organizes the data into the following structure:

```
data/
└── <experiment_name>/
    └── run_<id>/
        ├── config.json          # Configuration file for the run
        ├── metrics.csv          # Metrics tracked during the run
        ├── artifacts/           # Folder for saved artifacts
        │   ├── index.csv        # Index of saved artifacts
        │   ├── test_step_0.npy  # Example artifact file
        │   ├── test_step_2.npy
        │   └── ...
        └── logs/                # Folder for logs
            ├── info.log         # Log file for INFO messages
            └── error.log        # Log file for ERROR messages
```

### Key Details
- **`config.json`**: Stores the configuration used for the run.
- **`metrics.csv`**: Contains the metrics tracked during the experiment, with columns for steps and metric values.
- **`artifacts/`**: Stores artifacts (e.g., numpy arrays) saved during the experiment. Each artifact is named based on the step and a custom name.
- **`logs/`**: Contains log files for the run. Separate files are created for different log levels (e.g., `info.log`, `error.log`).

---

## Important Notes
- Ensure that the `data/` directory is not tracked by Git (add it to `.gitignore`) to avoid committing large experiment data.
- Use the `get_logger` method to customize logging behavior (e.g., log to terminal, file, or both).
- Always call `run.finalize()` at the end of an experiment to ensure all metrics and artifacts are saved properly.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve TrackLab.
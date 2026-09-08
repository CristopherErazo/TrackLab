import logging
from pathlib import Path


class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level



def create_run_logger(run_dir, logger_name, log_to_terminal=True, log_to_file=True, level=logging.INFO, log_format="%(asctime)s - %(levelname)s - %(message)s"):
    """
    Create a logger for a specific run.

    Args:
        run_dir (str): Directory for the run logs.
        logger_name (str): Name for logging.getLogger(). Must be unique per run
            directory (Run passes "tracklab.<experiment>.<run_id>"), because
            loggers are process-global and a shared name would send two runs'
            messages into one run's files.
        log_to_terminal (bool): Whether to log to the terminal.
        log_to_file (bool): Whether to log to a file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): Format for log messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # This logger installs its own handlers; don't also hand records to the
    # root logger, or every line prints twice whenever root is configured.
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    if log_to_file:
        # Create separate log files for different levels
        info_log_file = log_dir / "info.log"
        error_log_file = log_dir / "error.log"

        info_handler = logging.FileHandler(info_log_file)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        info_handler.addFilter(LevelFilter(logging.INFO))
        logger.addHandler(info_handler)

        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(LevelFilter(logging.ERROR))
        logger.addHandler(error_handler)

    if log_to_terminal:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
import logging
from pathlib import Path

def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # clear previous handlers to avoid duplicates in notebooks
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

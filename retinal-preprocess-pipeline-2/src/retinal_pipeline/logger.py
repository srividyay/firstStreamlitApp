import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_dir: str, level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"pipeline_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers in interactive runs
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

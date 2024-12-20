import logging
import time


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with a cool format."""
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)

def time_it(func: callable, *args: tuple, **kwargs: dict) -> tuple:
    """Wrapper function to apply a function and measure its execution time."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

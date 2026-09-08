"""
Thread-safe logging bridge between worker threads and the Tk GUI.

Worker threads log through the standard `logging` module. A `QueueHandler`
pushes records into a queue that the GUI drains from the Tk main loop with
`after()`, which keeps all widget updates on the main thread.
"""

from __future__ import annotations

import logging
import queue
import sys
from logging.handlers import QueueHandler

LOGGER_NAME = "lyricchord"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(level: int = logging.INFO) -> "queue.Queue[logging.LogRecord]":
    """Configure the package logger and return the queue the GUI should drain."""
    q: "queue.Queue[logging.LogRecord]" = queue.Queue()
    logger = get_logger()
    logger.setLevel(level)
    # Avoid duplicate handlers if setup_logging is called twice (e.g. in tests).
    if not any(isinstance(h, QueueHandler) for h in logger.handlers):
        logger.addHandler(QueueHandler(q))
    # Under pythonw.exe there is no console: sys.stderr is None, so skip the stream handler.
    if sys.stderr is not None and not any(type(h) is logging.StreamHandler for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(stream)
    # Quieten noisy third-party loggers.
    for noisy in ("numba", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return q

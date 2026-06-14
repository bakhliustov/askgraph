"""Simple structured logging for askgraph (uses rich)."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str = "askgraph") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

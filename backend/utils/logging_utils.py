"""Logging helpers used across the backend and the CLI scripts.

The goal is a single configured root logger with:

* colourised console output (disabled when ``NO_COLOR`` is set or output is not a TTY),
* a rotating file handler under ``logs/``,
* a per-module :func:`get_logger` helper.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        text = super().format(record)
        if not self.use_color:
            return text
        color = _COLORS.get(record.levelname, "")
        return f"{color}{text}{_RESET}" if color else text


def setup_logging(
    level: str | int = "INFO",
    log_dir: Optional[str | Path] = None,
    log_file: str = "vestiai.log",
    quiet: bool = False,
) -> logging.Logger:
    """Configure the ``vestiai`` root logger exactly once."""
    global _CONFIGURED
    logger = logging.getLogger("vestiai")
    if _CONFIGURED:
        return logger

    numeric_level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    datefmt = "%H:%M:%S"

    if not quiet:
        use_color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(_ColorFormatter(fmt, datefmt=datefmt, use_color=use_color))
        logger.addHandler(console)

    if log_dir is not None:
        try:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                path / log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - read-only FS
            logger.warning("Could not create file log handler: %s", exc)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, configuring the root logger on first use."""
    root = logging.getLogger("vestiai")
    if not _CONFIGURED:
        setup_logging(os.environ.get("VESTIAI_LOG_LEVEL", "INFO"), Path(os.environ.get("VESTIAI_LOG_DIR", "logs")))
        root = logging.getLogger("vestiai")
    if name.startswith("vestiai"):
        return logging.getLogger(name)
    return root.getChild(name)

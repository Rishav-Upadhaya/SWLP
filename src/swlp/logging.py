from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_logs:
        try:
            from pythonjsonlogger.json import JsonFormatter

            formatter_class = JsonFormatter
            formatter_kwargs = {"fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"}
            formatter_name = "json"
        except Exception:  # pragma: no cover - fallback path
            formatter_class = logging.Formatter
            formatter_kwargs = {"fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"}
            formatter_name = "plain"
    else:
        formatter_class = logging.Formatter
        formatter_kwargs = {"fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"}
        formatter_name = "plain"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                formatter_name: {
                    "()": formatter_class,
                    **formatter_kwargs,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter_name,
                    "level": log_level,
                }
            },
            "root": {"handlers": ["console"], "level": log_level},
        }
    )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

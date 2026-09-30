from __future__ import annotations

import logging
from logging.config import dictConfig

# Attributes every LogRecord carries; anything else came in via ``extra=``.
_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class ExtrasFormatter(logging.Formatter):
    """Plain-text formatter that appends ``extra={...}`` fields as key=value.

    The stdlib Formatter silently drops extras, so events like
    ``swlp_residency_plan`` printed a bare name with none of their numbers.
    """

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = " ".join(
            f"{k}={v!r}" for k, v in vars(record).items() if k not in _RECORD_ATTRS
        )
        return f"{line} {extras}" if extras else line


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter_kwargs = {"fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"}

    if json_logs:
        try:
            from pythonjsonlogger.json import JsonFormatter

            formatter_class = JsonFormatter
            formatter_name = "json"
        except Exception:  # pragma: no cover - fallback path
            formatter_class = ExtrasFormatter
            formatter_name = "plain"
    else:
        formatter_class = ExtrasFormatter
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
    # HTTP clients flood DEBUG with per-header lines (100+ per HF config fetch);
    # the Hub's WARNINGs are server nags (e.g. "unauthenticated requests").
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing_extensions import override

import structlog

if TYPE_CHECKING:
    from structlog.typing import FilteringBoundLogger


def configure_logging(
    level: str | int = "INFO",
    verbose_exception: bool = True,
) -> None:
    log_level = (
        structlog.processors.NAME_TO_LEVEL.get(level.casefold(), logging.INFO)
        if isinstance(level, str)
        else level
    )
    wrapper_class = structlog.make_filtering_bound_logger(log_level)
    if not verbose_exception:

        class BoundLoggerWithoutException(wrapper_class):  # type: ignore[misc, valid-type]
            exception = wrapper_class.error
            aexception = wrapper_class.aerror

        wrapper_class = BoundLoggerWithoutException
    structlog.configure(wrapper_class=wrapper_class)


class StructLogHandler(logging.Handler):
    @override
    def emit(self, record: logging.LogRecord) -> None:
        structlog.get_logger(record.name).bind(exc_info=record.exc_info).log(
            record.levelno,
            record.getMessage(),
        )


logger: FilteringBoundLogger = structlog.get_logger("kafubot")

logging.getLogger().handlers.clear()
logging.getLogger().addHandler(StructLogHandler())

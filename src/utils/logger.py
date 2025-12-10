import logging
import sys

from src.config import LOG_LEVEL, LOGGER_NAME


def setup_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Setting logger with emoji"""

    logger = logging.getLogger(name)
    logger.setLevel(
        getattr(logging, LOG_LEVEL.upper())
    )  # DEBUG < INFO < WARNING < ERROR < CRITICAL

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Format with emoji
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class LogEmoji:
    """Log emoji constant"""

    START = "🚀"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    SEARCH = "🔍"
    DATABASE = "💾"
    AI = "🤖"
    DATA = "📊"
    DOWNLOAD = "⬇️"
    PROCESS = "⚙️"

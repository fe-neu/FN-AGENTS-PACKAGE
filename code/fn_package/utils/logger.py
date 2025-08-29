import logging
from fn_package.config import ENABLE_LOGGING, LOG_LEVEL

def get_logger(name: str):
    """
    Create and configure a logger instance.

    - Ensures no duplicate handlers are attached.
    - Applies a standard log format with timestamp, level, logger name, and message.
    - Respects the global ENABLE_LOGGING flag and LOG_LEVEL from config.
      If logging is disabled, suppresses all output.
    
    Args:
        name (str): Name of the logger (typically __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set log level based on configuration
    if ENABLE_LOGGING:
        logger.setLevel(LOG_LEVEL)
    else:
        logger.setLevel(logging.CRITICAL + 1)  # Suppress all logs

    return logger

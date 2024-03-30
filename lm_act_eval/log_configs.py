import logging
import os

from rich.logging import RichHandler

rich_handler = RichHandler(rich_tracebacks=True)
rich_handler.setFormatter(fmt=logging.Formatter(fmt="%(message)s", datefmt="[%H:%M:%S.%f]"))
file_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="[%Y-%m-%d %H:%M:%S]")


# Setup FileHandler for logging to a file
file_handler = logging.FileHandler('lm_act_eval.log')
file_handler.setFormatter(file_formatter)


# Retrieve log level from environment variable or default to 'debug'
log_level_str = os.getenv('LOG_LEVELS', 'debug').lower()

# Match the log level string to logging levels
log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


log_level = log_levels.get(log_level_str, logging.DEBUG)  # Default to DEBUG if no match

# Configure the root logger
logging.basicConfig(
    level=log_level,
    handlers=[
        # Console output with Rich formatting
        rich_handler,
        # Log to a file with custom formatting
        file_handler,
    ]
)
logging.getLogger(__name__).setLevel(log_level)
logging.getLogger(__name__).setLevel(log_levels.get('error'))

logger = logging.getLogger(__name__)
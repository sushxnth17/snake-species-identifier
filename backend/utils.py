import logging
from backend.logging_config import setup_structured_logging

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures a basic but professional logging format for stdout.
    Delegates to structured logging.
    
    Args:
        level: The logging severity level to capture (default: logging.INFO)
    """
    level_name = logging.getLevelName(level)
    setup_structured_logging(level_name)


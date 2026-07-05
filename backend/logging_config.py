import logging
import sys
import json
import contextvars
from datetime import datetime
from typing import Any, Dict

# Context variable to store request ID across async tasks
request_id_var = contextvars.ContextVar("request_id", default="")

class StructuredJSONFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log entries as JSON.
    """
    RESERVED_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process"
    }

    def format(self, record: logging.LogRecord) -> str:
        # Create standard structured log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject request_id if available in context
        req_id = request_id_var.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Merge extra attributes passed via extra={}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        # Handle exceptions and record stack trace / type
        if record.exc_info:
            log_entry["exception_type"] = record.exc_info[0].__name__
            log_entry["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_structured_logging(level: str = "INFO") -> None:
    """
    Configures the root logger with the StructuredJSONFormatter.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Configure StreamHandler writing to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(handler)
    
    # Silence third-party verbose logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

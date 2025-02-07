import logging
import json
import sys
import os
import threading
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum
import uuid

class LogFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"  # New format for structured text output

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    QUIET = "quiet"

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        try:
            return cls[level.upper()]
        except KeyError:
            valid_levels = ", ".join(l.value for l in cls)
            raise ValueError(f"Invalid log level: {level}. Valid levels are: {valid_levels}")

    def to_logging_level(self) -> int:
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.QUIET: logging.ERROR + 10  # Higher than ERROR
        }[self]

class StructuredFormatter(logging.Formatter):
    """Format log records in a structured text format."""
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation ID if not present
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = str(uuid.uuid4())

        # Build structured message
        parts = [
            f"[{self.formatTime(record)}]",
            f"[{record.levelname}]",
            f"[{record.name}]",
            f"[{record.correlation_id}]",
            f"[PID:{os.getpid()}]",
            f"[TID:{threading.get_ident()}]",
            f"[{record.filename}:{record.lineno}]"
        ]

        # Add structured context if available
        if hasattr(record, 'context'):
            context_str = ' '.join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"[{context_str}]")

        # Add message
        parts.append(record.getMessage())

        # Add exception info if present
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return ' '.join(parts)

class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation ID if not present
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = str(uuid.uuid4())

        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": record.correlation_id,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage()
        }

        # Add structured context if available
        if hasattr(record, 'context'):
            log_data["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add error details if available
        if hasattr(record, 'error_details'):
            log_data["error_details"] = record.error_details

        return json.dumps(log_data)

def setup_logging(
    name: str,
    level: Union[LogLevel, str] = LogLevel.INFO,
    format_type: LogFormat = LogFormat.TEXT
) -> logging.Logger:
    """
    Set up logging with the specified configuration.
    
    Args:
        name: Logger name
        level: Log level to use
        format_type: Output format (text, json, or structured)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    
    logger = logging.getLogger(name)
    logger.setLevel(level.to_logging_level())
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level.to_logging_level())
    
    # Set formatter based on format type
    if format_type == LogFormat.JSON:
        formatter = JSONFormatter()
    elif format_type == LogFormat.STRUCTURED:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger 
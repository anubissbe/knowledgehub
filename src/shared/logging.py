"""Shared logging configuration"""

import logging
import sys
from typing import Optional, Dict, Any, Tuple
import json
from datetime import datetime

from .config import Config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "msecs", "pathname", "process", 
                          "processName", "relativeCreated", "thread", 
                          "threadName", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logging(
    name: str,
    level: Optional[str] = None,
    use_json: Optional[bool] = None
) -> logging.Logger:
    """Setup logging for a module"""
    config = Config()
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if use_json is None:
        use_json = config.is_production
    
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(config.LOG_FORMAT)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to logs"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]) -> None:
        super().__init__(logger, extra)
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        # Add extra context to all log messages
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"].update(self.extra)
        return msg, kwargs


def create_logger_with_context(
    name: str,
    context: Dict[str, Any]
) -> LoggerAdapter:
    """Create a logger with additional context"""
    logger = get_logger(name)
    return LoggerAdapter(logger, context)
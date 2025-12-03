"""
Logging Configuration Module

Sets up structured JSON logging for the sentence analysis system.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'job_id'):
            log_data['job_id'] = record.job_id
        
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_data['stack_trace'] = record.stack_info
        
        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    json_format: bool = True
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        json_format: Use JSON formatting if True
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('sentence_analysis')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        logger.addHandler(file_handler)
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    request_id: str = None,
    job_id: str = None,
    context: Dict[str, Any] = None,
    metrics: Dict[str, Any] = None
):
    """
    Log message with additional context
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error)
        message: Log message
        request_id: Optional request ID
        job_id: Optional job ID
        context: Optional context dictionary
        metrics: Optional metrics dictionary
    """
    extra = {}
    
    if request_id:
        extra['request_id'] = request_id
    
    if job_id:
        extra['job_id'] = job_id
    
    if context:
        extra['context'] = context
    
    if metrics:
        extra['metrics'] = metrics
    
    log_method = getattr(logger, level.lower())
    log_method(message, extra=extra)


def log_error(
    logger: logging.Logger,
    message: str,
    error: Exception,
    request_id: str = None,
    job_id: str = None,
    context: Dict[str, Any] = None
):
    """
    Log error with exception details
    
    Args:
        logger: Logger instance
        message: Error message
        error: Exception object
        request_id: Optional request ID
        job_id: Optional job ID
        context: Optional context dictionary
    """
    extra = {}
    
    if request_id:
        extra['request_id'] = request_id
    
    if job_id:
        extra['job_id'] = job_id
    
    if context:
        extra['context'] = context
    
    logger.error(message, exc_info=True, extra=extra, stack_info=True)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration: float,
    request_id: str = None,
    job_id: str = None,
    additional_metrics: Dict[str, Any] = None
):
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        request_id: Optional request ID
        job_id: Optional job ID
        additional_metrics: Optional additional metrics
    """
    metrics = {
        'operation': operation,
        'duration_seconds': duration
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    log_with_context(
        logger,
        'info',
        f"Performance: {operation} completed in {duration:.3f}s",
        request_id=request_id,
        job_id=job_id,
        metrics=metrics
    )


# Create default logger
default_logger = setup_logging()

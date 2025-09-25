"""
Logging utilities for the Virtual Influencer Persona Agent
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "persona_agent",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file
        console: Enable console logging
        file_logging: Enable file logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        if not log_file:
            # Default log file location
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all debug logs
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        setup_logger(name)
        # Prevent propagation to avoid duplicate logging
        logger.propagate = False
    
    return logger


class LogContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator


class ComponentLogger:
    """
    Enhanced logger wrapper that adds component-specific prefixes to all log messages.
    
    Provides clear identification of which component is generating logs, especially
    useful for parallel RAG operations and server deployments.
    """
    
    def __init__(self, component_type: str, instance_id: Optional[str] = None, base_logger: Optional[logging.Logger] = None):
        """
        Initialize component logger with automatic prefixing.
        
        Args:
            component_type: Short identifier for component type (e.g., 'ModelMgr', 'Reranker')
            instance_id: Optional instance identifier (e.g., persona_id, model_name)
            base_logger: Base logger to wrap (if None, creates default logger)
        """
        self.component_type = component_type
        self.instance_id = instance_id
        
        # Create component prefix
        if instance_id:
            self.prefix = f"[{component_type}:{instance_id}]"
        else:
            self.prefix = f"[{component_type}]"
        
        # Setup base logger
        if base_logger:
            self.logger = base_logger
        else:
            # Create logger with component-specific name for better filtering
            logger_name = f"component.{component_type.lower()}"
            if instance_id:
                logger_name += f".{instance_id}"
            self.logger = get_logger(logger_name)
    
    def _format_message(self, message: str) -> str:
        """Add component prefix to message."""
        return f"{self.prefix} {message}"
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with component prefix."""
        self.logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with component prefix."""
        self.logger.info(self._format_message(message), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with component prefix."""
        self.logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with component prefix."""
        self.logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with component prefix."""
        self.logger.critical(self._format_message(message), *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with component prefix."""
        self.logger.exception(self._format_message(message), *args, **kwargs)
    
    def log(self, level: int, message: str, *args, **kwargs):
        """Log at specific level with component prefix."""
        self.logger.log(level, self._format_message(message), *args, **kwargs)
    
    def setLevel(self, level):
        """Set logging level on underlying logger."""
        self.logger.setLevel(level)
    
    def isEnabledFor(self, level):
        """Check if logging is enabled for level."""
        return self.logger.isEnabledFor(level)
    
    @property
    def level(self):
        """Get current logging level."""
        return self.logger.level
    
    def get_base_logger(self) -> logging.Logger:
        """Get the underlying base logger."""
        return self.logger


def get_component_logger(component_type: str, instance_id: Optional[str] = None) -> ComponentLogger:
    """
    Create a component-specific logger with automatic prefixing.
    
    Args:
        component_type: Short identifier for component type
        instance_id: Optional instance identifier
        
    Returns:
        ComponentLogger instance with prefixed logging
        
    Examples:
        # Singleton components
        logger = get_component_logger("ModelMgr")
        logger.info("Loading model")  # -> [ModelMgr] Loading model
        
        # Instance-specific components  
        logger = get_component_logger("MMStore", "greg_startup")
        logger.info("Indexing documents")  # -> [MMStore:greg_startup] Indexing documents
    """
    return ComponentLogger(component_type, instance_id)
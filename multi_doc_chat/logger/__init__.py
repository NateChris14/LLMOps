from .custom_logger import CustomLogger as _CustomLogger #backwards compatibility
try:
    from .custom_logger import CustomLogger
except ImportError:
    CustomLogger = _CustomLogger

# Expose a global structlog-style logger used across the codebase
GLOBAL_LOGGER = CustomLogger().get_logger(__name__)
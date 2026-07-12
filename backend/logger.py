"""
Centralized logging configuration for BookAligner.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Console handler with colors
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# File handler for persistent logs
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "bookaligner_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

# Export
__all__ = ["logger"]
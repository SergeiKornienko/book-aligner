"""
Shared dependencies for API endpoints.
"""

from pathlib import Path


def get_upload_dir() -> Path:
    """Get the upload directory path."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir
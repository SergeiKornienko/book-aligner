"""
Shared data models for BookAligner services.
"""

from dataclasses import dataclass


@dataclass
class TextFragment:
    """
    Represents a text fragment with its properties and position.
    """
    text: str
    x: float
    y: float
    width: float
    height: float
    font_name: str = "Times-Roman"
    font_size: float = 12.0
    page_num: int = 0  # Page number (0-based)
    
    def __post_init__(self):
        """Validate fragment data"""
        if not isinstance(self.text, str):
            raise ValueError("Text must be a string")
        if self.width < 0 or self.height < 0:
            raise ValueError("Width and height must be non-negative")
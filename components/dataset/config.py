import enum
from dataclasses import dataclass


@dataclass
class ImageInfo:
    """Config of input images."""

    WIDTH: int = 1280
    HEIGHT: int = 1024


class Quality(enum.Enum):
    """Enumerate of BAD or GOOD."""

    BAD = 0
    GOOD = 1
    BOTH = 3

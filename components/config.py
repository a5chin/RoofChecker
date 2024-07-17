import enum
from dataclasses import dataclass


@dataclass
class MEAN:
    """Mean of ImageNet."""

    R = 0.485
    G = 0.456
    B = 0.406


@dataclass
class Model:
    """The Model config."""

    NAME = "tf_efficientnet_b6_ns"


@dataclass
class NPY:
    "The path of mean and cov_i."

    MEAN = "ckpt/mean.npy"
    COV_I = "ckpt/cov_i.npy"


@dataclass
class ImageInfo:
    """Config of input images."""

    WIDTH: int = 1280
    HEIGHT: int = 1024


@dataclass
class Parameters:
    """Config of Hyper Parameters."""

    ALPHA = 1e-4
    LAMDA = 1


class Quality(enum.Enum):
    """Enumerate of BAD or GOOD."""

    BAD = 0
    GOOD = 1
    BOTH = 3


@dataclass
class STD:
    """Standard Deviation of ImageNet."""

    R = 0.229
    G = 0.224
    B = 0.225

import numpy as np
import torch
from torch import nn

from components.config import NPY


class MahalanobisLoss(nn.Module):
    """MahalanobisLoss."""

    def __init__(
        self: "MahalanobisLoss",
        mean: np.ndarray | None = None,
        cov_i: np.ndarray | None = None,
    ) -> None:
        """Mahalanobis Loss Class.

        Args:
        ----
            mean (np.ndarray | None, optional):
                Mean of target features. Defaults to None.
            cov_i (np.ndarray | None, optional):
                Inverse covariance matrix of target features. Defaults to None.

        """
        super().__init__()

        if mean is None or cov_i is None:
            self.load()
        else:
            self.mean = mean
            self.cov_i = cov_i

    def load(self: "MahalanobisLoss") -> None:
        """Load mean and inverse covariance matrix for extracted features."""
        self.mean = np.load(NPY.MEAN)
        self.cov_i = np.load(NPY.COV_I)

    def forward(self: "MahalanobisLoss", features: torch.Tensor) -> torch.Tensor:
        """Calculate MahalanobisLoss.

        Args:
        ----
            features (torch.Tensor): Extracted features from the Model

        Returns:
        -------
            torch.Tensor: Calculated loss

        """
        diff = features - torch.from_numpy(self.mean).float()

        dist = torch.sqrt(
            torch.sum(
                torch.matmul(diff, torch.from_numpy(self.cov_i).float()) * diff,
                dim=1,
            )
        )

        return dist.mean()

import numpy as np
from torch import nn
from torch.utils.data import DataLoader


def get_features(dataloader: DataLoader, model: nn.Module) -> np.ndarray:
    """Return features.

    Args:
    ----
        dataloader (DataLoader): _description_
        model (nn.Module): _description_

    Returns:
    -------
        np.ndarray: features of dataset. (batch_size, 2304) shapes.

    """
    return np.vstack([model(inputs).detach().numpy() for inputs, _ in dataloader])

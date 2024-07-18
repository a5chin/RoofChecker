from pathlib import Path

import numpy as np

from components.config import NPY, Quality
from components.dataset import RoofDataLoader
from components.model import get_model
from components.share import get_features


def train() -> None:
    """Training function aimed at preserving the mean and inverse covariance matrix."""
    train_dataloader = RoofDataLoader(quality=Quality.BOTH, batch_size=1, shuffle=False)
    model = get_model()

    features = get_features(dataloader=train_dataloader, model=model)
    cov = np.cov(features.T)

    mean, cov_i = np.mean(features, axis=0), np.linalg.pinv(cov)

    np.save(file=Path(NPY.MEAN).with_suffix(""), arr=mean)
    np.save(file=Path(NPY.COV_I).with_suffix(""), arr=cov_i)

import cv2
import numpy as np


def get_heatmap(target: np.ndarray, feature: np.ndarray) -> np.ndarray:
    """Return annomaly heatmap.

    Args:
    ----
        target (np.ndarray): Target image
        feature (np.ndarray): Feature of target image

    Returns:
    -------
        np.ndarray: Anomaly heatmap

    """
    norm = np.linalg.norm(target - feature, axis=-1, ord=1)
    heatmap = (norm - np.min(norm)) / (np.max(norm) - np.min(norm))

    jet = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return cv2.cvtColor(jet, cv2.COLOR_BGR2RGB).astype(np.uint8)

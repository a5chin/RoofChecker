from torchvision import transforms

from components.config import MEAN, STD, ImageInfo


def get_transform() -> transforms.Compose:
    """Get Transforms function.

    Returns
    -------
        transforms.Compose: Target transforms

    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(ImageInfo.WIDTH // 10),
            transforms.Normalize(
                [getattr(MEAN, color) for color in ["R", "G", "B"]],
                [getattr(STD, color) for color in ["R", "G", "B"]],
            ),
        ]
    )

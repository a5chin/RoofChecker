from pathlib import Path

import cv2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from components.dataset.config import Image, Quality


class RoofDataLoader(DataLoader):
    """RoofChecker DataLoader class."""

    def __init__(
        self: "RoofDataLoader",
        transform: transforms.Compose | None = None,
        quality: Quality = Quality.GOOD,
        extensions: tuple[str] = ("png",),
        dataset: Dataset | None = None,
        batch_size: int = 4,
    ) -> None:
        """Initialize DataLoader.

        Args:
        ----
            transform (transforms.Compose | None, optional):
                The class to transform image. Defaults to None.
            quality (Quality, optional):
                Quality.BAD or Quality.GOOD. Defaults to Quality.GOOD.
            extensions (tuple[str], optional): Image extensions. Defaults to ("png",).
            dataset (Dataset | None, optional): The target Dataset Defaults to None.
            batch_size (int, optional): The batch size of target Dataset. Defaults to 4.

        """
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(size=Image.WIDTH // 5),
                    transforms.Resize(size=Image.WIDTH // 10),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        if dataset is None:
            dataset = RoofDataset(
                transform=transform, quality=quality, extensions=extensions
            )

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=True)


class RoofDataset(Dataset):
    """The Dataset of RoofChecker."""

    def __init__(
        self: "RoofDataset",
        transform: transforms.Compose,
        quality: Quality = Quality.GOOD,
        extensions: tuple[str] = ("png",),
    ) -> None:
        """To Collect images of roof.

        Args:
        ----
            transform (transforms.Compose | None, optional):
                The class to transform image. Defaults to None.
            quality (Quality, optional):
                Quality.BAD or Quality.GOOD. Defaults to Quality.GOOD
            extensions (tuple[str], optional): Image extensions. Defaults to ("png",).

        """
        super().__init__()
        self.transform = transform
        self.quality = quality
        self.images_list = [
            file
            for extension in extensions
            for file in Path(f"data/{self.quality.name}").glob(f"*.{extension}")
        ]

    def __getitem__(self: "RoofDataset", index: int) -> tuple[Tensor, Tensor]:
        """Return Data of image and labels.

        Args:
        ----
            index (int): The order

        Returns:
        -------
            tuple[Tensor, Tensor]: Data of images and labels

        """
        p = self.images_list[index]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        return self.transform(img), self.quality.value

    def __len__(self: "RoofDataset") -> int:
        """Return length of the Dataset.

        Returns
        -------
            int: The length of the Dataset

        """
        return len(self.images_list)

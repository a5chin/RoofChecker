from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from components.config import Quality
from components.share.transform import get_transform


class RoofDataLoader(DataLoader):
    """RoofChecker DataLoader class."""

    def __init__(
        self: "RoofDataLoader",
        quality: Quality = Quality.GOOD,
        extensions: tuple[str] = ("png",),
        transform: transforms.Compose | None = None,
        dataset: Dataset | None = None,
        batch_size: int = 4,
        shuffle: bool = True,
    ) -> None:
        """Initialize DataLoader.

        Args:
        ----
            quality (Quality, optional):
                Quality.BAD or Quality.GOOD. Defaults to Quality.GOOD.
            extensions (tuple[str], optional): Image extensions. Defaults to ("png",).
            transform (transforms.Compose | None, optional):
                The class to transform image. Defaults to None.
            dataset (Dataset | None, optional): The target Dataset Defaults to None.
            batch_size (int, optional): The batch size of target Dataset. Defaults to 4.
            shuffle (bool, optional): It is shuffled or not. Defaults to True.

        """
        if transform is None:
            transform = get_transform()
        if dataset is None:
            dataset = RoofDataset(
                transform=transform, quality=quality, extensions=extensions
            )

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


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
            for file in (
                Path("data")
                if quality == Quality.BOTH
                else Path(f"data/{self.quality.name}")
            ).glob(f"**/*.{extension}" if quality == Quality.BOTH else f"*.{extension}")
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
        img = Image.open(p).convert("RGB")

        return self.transform(img), self.quality.value

    def __len__(self: "RoofDataset") -> int:
        """Return length of the Dataset.

        Returns
        -------
            int: The length of the Dataset

        """
        return len(self.images_list)

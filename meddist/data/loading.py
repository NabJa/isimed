import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import List

import monai.transforms as tfm
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

MonaiData = List[dict]


class DistanceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.images = list(self.path.glob("*.nii.gz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": str(self.images[idx])}


def get_transformations(
    num_samples: int, crop_size=128, add_intensity_augmentation=False
):
    load_and_crop = [
        tfm.LoadImaged(keys="image", ensure_channel_first=True),
        tfm.CropForegroundd(
            keys="image", source_key="image", select_fn=lambda x: x > 0
        ),
        tfm.ScaleIntensityRangePercentilesd(
            keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
        ),
    ]

    if add_intensity_augmentation:
        augment = [
            tfm.RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
            tfm.RandGaussianSmoothd(keys="image", prob=0.2),
        ]

    else:
        augment = [tfm.Lambdad(keys="image", func=lambda x: x)]

    crop = [
        tfm.RandSpatialCropSamplesd(
            keys="image", roi_size=crop_size, num_samples=num_samples, random_size=False
        )
    ]

    train_transforms = tfm.Compose(load_and_crop + augment + crop)
    valid_transforms = tfm.Compose(load_and_crop + crop)

    return train_transforms, valid_transforms


def read_data_split(path):
    with open(path, mode="rb") as file:
        split = pickle.load(file)

    return split["train"], split["validation"], split["test"]


def get_dataloaders(
    path,
    num_samples: int = 1,
    crop_size=128,
    batch_size=1,
    add_intensity_augmentation=False,
    train_transform=None,
    valid_transform=None,
):
    train_data, valid_data, _ = read_data_split(path)

    _train_transform, _valid_transform = get_transformations(
        num_samples, crop_size, add_intensity_augmentation
    )

    train_data = Dataset(
        train_data,
        transform=_train_transform if train_transform is None else train_transform,
    )
    valid_data = Dataset(
        valid_data,
        transform=_valid_transform if valid_transform is None else valid_transform,
    )

    return DataLoader(train_data, batch_size=batch_size), DataLoader(valid_data)

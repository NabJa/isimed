import pickle
from pathlib import Path
from typing import List

import monai.transforms as tfm
from monai.data import DataLoader, Dataset

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
    num_workers=8,
    add_intensity_augmentation=False,
    train_transform=None,
    valid_transform=None,
):
    train_data, valid_data, _ = read_data_split(path)

    # If batch size > 1, all keys in data are concatenated. If some have a "label" and some dont, this causes an error.
    # This is fixed by just added the "label" key with an empty string so they can be concatenated to a list.
    for x in train_data:
        if "label" not in x:
            x["label"] = ""

    _train_transform, _valid_transform = get_transformations(
        num_samples, crop_size, add_intensity_augmentation
    )

    train_dataset = Dataset(
        train_data,
        transform=_train_transform if train_transform is None else train_transform,
    )
    valid_dataset = Dataset(
        valid_data,
        transform=_valid_transform if valid_transform is None else valid_transform,
    )

    return DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers), DataLoader(valid_dataset, num_workers=num_workers)

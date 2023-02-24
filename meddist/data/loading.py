import pickle
from pathlib import Path
from typing import List

import monai.transforms as tfm
import numpy as np
from monai.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from meddist.transforms import GetClassesFromCropsd

MonaiData = List[dict]


class DistanceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.images = list(self.path.glob("*.nii.gz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": str(self.images[idx])}


def get_contrastive_transform(crops: int = 2, crop_size: int = 32):
    return tfm.Compose(
        [
            tfm.LoadImaged(keys="image", ensure_channel_first=True),
            tfm.CropForegroundd(
                keys="image",
                source_key="image",
                select_fn=lambda x: x > 0,
            ),
            tfm.SpatialPadd(keys="image", spatial_size=crop_size),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[crop_size] * 3,
                random_size=False,
                num_samples=crops,
            ),
            tfm.CopyItemsd(
                keys=["image"],
                times=1,
                names=["image_2"],
                allow_missing_keys=False,
            ),
            tfm.OneOf(
                transforms=[
                    tfm.RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.33),
                        dropout_holes=True,
                    ),
                    tfm.RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.66),
                        dropout_holes=False,
                    ),
                ]
            ),
            tfm.RandCoarseShuffled(keys=["image"], prob=0.8, holes=2, spatial_size=6),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            tfm.OneOf(
                transforms=[
                    tfm.RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.33),
                        dropout_holes=True,
                    ),
                    tfm.RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.66),
                        dropout_holes=False,
                    ),
                ]
            ),
            tfm.RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=2, spatial_size=6),
        ]
    )


def get_transformations(
    num_samples: int, crop_size=128, add_intensity_augmentation=False
):
    load_and_crop = [
        tfm.LoadImaged(keys="image", ensure_channel_first=True),
        tfm.CropForegroundd(
            keys="image", source_key="image", select_fn=lambda x: x > 0
        ),
        tfm.SpatialPadd(keys="image", spatial_size=crop_size),
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
    valid_batch_size=2,
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

    return DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    ), DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
    )


###########################
### Downstream analysis ###
###########################


def get_downstream_transormation(crop_size=32, pos_ratio=0.65):
    return tfm.Compose(
        [
            tfm.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            tfm.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > 0,
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                pos=pos_ratio,
                num_samples=32,
                spatial_size=crop_size,
            ),
            GetClassesFromCropsd(label_key="label"),
        ]
    )


def get_downstram_classification_data(path_to_data_split, crop_size=64, pos_ratio=0.65):
    """Deprecated but still in use for downstram_class.py"""

    with open(path_to_data_split, mode="rb") as file:
        split = pickle.load(file)

    transform = tfm.Compose(
        [
            tfm.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            tfm.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > 0,
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                pos=pos_ratio,
                num_samples=32,
                spatial_size=crop_size,
            ),
            GetClassesFromCropsd(label_key="label"),
        ]
    )

    dataset_valid = Dataset(split["validation"], transform=transform)
    loader_valid = DataLoader(dataset_valid, num_workers=8)

    dataset_train = Dataset(split["test"], transform=transform)
    loader_train = DataLoader(dataset_train, num_workers=8)

    return loader_train, loader_valid


def kfold_get_downstram_data(
    path_to_data_split, n_splits=5, crop_size=32, num_workers=8, task="classification"
):
    
    pos_ratio = 0.65 if task == "classification" else 100
    transform = get_downstream_transormation(crop_size, pos_ratio)

    kfold = KFold(n_splits=n_splits)

    with open(path_to_data_split, mode="rb") as file:
        data = pickle.load(file)["test"]

    # Make iit indexable with ints
    data = np.array(data)

    for train_idx, test_idx in kfold.split(data):
        train_loader = DataLoader(
            Dataset(data[train_idx], transform=transform), num_workers=num_workers
        )
        test_loader = DataLoader(
            Dataset(data[test_idx], transform=transform), num_workers=num_workers
        )
        yield train_loader, test_loader

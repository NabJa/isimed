import pickle
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import monai.transforms as tfm
import nibabel as nib
import numpy as np
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MonaiData = List[dict]


def has_segmentation(data, label_key="label"):
    data["has_segmentation"] = 0

    if label_key in data:
        label = nib.load(data[label_key]).get_fdata()
        if np.sum(label) > 0:
            data["has_segmentation"] = 1

    return data


def get_all_with_seg(dataset, label_key="label", processes=8):
    has_seg = partial(has_segmentation, label_key=label_key)
    with Pool(processes=processes) as p:
        return list(tqdm(p.imap(has_seg, list(dataset)), total=len(dataset)))


def get_patient_name(path):
    return Path(path).parents[1].name


def split_segmentation_data(data, valid_size=0.2, save=None):

    data = get_all_with_seg(data)

    patients = defaultdict(list)
    patient_dicts = defaultdict(list)

    for d in data:
        name = get_patient_name(d["image"])
        patients[name].append(d["has_segmentation"])
        patient_dicts[name].append(d)

    patients_with_label = [k for k, v in patients.items() if np.all(v)]
    patients_without_label = [k for k, v in patients.items() if not np.all(v)]

    valid_data = set(random.sample(patients_with_label, int(valid_size * len(data))))
    test_data = set(patients_with_label) - valid_data
    train_data = set(patients_with_label) - valid_data - test_data | set(
        patients_without_label
    )

    _train_data, _valid_data, _test_data = [], [], []

    for patient, series in patient_dicts.items():
        if patient in train_data:
            _train_data += series
        elif patient in valid_data:
            _valid_data += series
        elif patient in test_data:
            _test_data += series

    if save is not None:
        save_datasplit(_train_data, _valid_data, _test_data, save)

    return _train_data, _valid_data, _test_data


def split_dataset(
    dataset, test_size=0.2, val_size=0.2, seed=42, save: Optional[Path] = None
) -> Tuple[MonaiData, MonaiData, MonaiData]:
    """
    Split a monai style dataset into train validation and test parts.

    Args:
        dataset: A dataset where __getitem__ returns a monai like data dictionary.
        test_size: Defaults to 0.2.
        val_size: Defaults to 0.2.
        save: Path to folder to save spliting. If None, will not save.

    Returns:
        Tuple: Train, validation and test data
    """
    train_size = 1 - test_size - val_size

    assert (train_size + test_size + val_size) == 1

    data = [item for item in dataset]
    train_data = [item for item in data if "label" not in item]
    labeled_data = [item for item in data if "label" in item]

    # Proportion of labeled_data that can go to train_data
    remaining_train_size = (train_size * len(data) - len(train_data)) / len(
        labeled_data
    )

    # Complete train data. All unlabeled + remaining labeled to get train_size
    remaining_train_data, val_and_test_data = train_test_split(
        labeled_data, train_size=remaining_train_size, random_state=seed
    )
    train_data += remaining_train_data

    # Remaining labeled data can be split in val and test
    val_data, test_data = train_test_split(
        val_and_test_data,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
    )

    if save is not None:
        save_datasplit(train_data, val_data, test_data, save)

    return train_data, val_data, test_data


def save_datasplit(train_data, val_data, test_data, save_to_path) -> None:
    save_to_path = Path(save_to_path)

    if save_to_path.is_dir():
        save_to_path = save_to_path / "split.pkl"

    with open(save_to_path, mode="wb") as file:
        split = {"train": train_data, "validation": val_data, "test": test_data}
        pickle.dump(split, file)


def find_parent_dirs(root_dir: str, postfix: str) -> set:
    """
    Returns a set of unique parent directories that contain files with the given postfix, starting from the given root directory.

    Args:
        root_dir (str): the directory to start searching from.
        postfix (str): the file extension to search for, including the dot (e.g. ".txt" or ".nii.gz")

    Returns:
        set: A set of unique parent directories that contain files with the given postfix
    """
    parent_dirs = set()
    for file_path in Path(root_dir).rglob("*" + postfix):
        parent_dirs.add(file_path.parent)
    return parent_dirs


class DistDataset:
    """
    This dataset searches all nifti files in a given directory.
    The parent directory is assumed to be a series and every series is treatet as one item.
    """

    def __init__(
        self,
        path: Path,
        image_pattern: str = "CTres",
        label_pattern: Optional[str] = "SEG",
    ) -> None:
        """
        Args:
            path: Path to folder containing nifti series.
            image_pattern: Prefix for all images. Defaults to "CTres".
            label_pattern: Prefix for all labels. Optional. Defaults to "SEG".
        """
        self.path = Path(path)
        self.series = list(find_parent_dirs(self.path, ".nii.gz"))
        self.image_pattern = image_pattern
        self.label_pattern = label_pattern

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> dict:
        series = self.series[index]

        image = next(series.glob(self.image_pattern + "*.nii.gz"))

        sample = {"image": str(image)}

        if self.label_pattern is not None:
            try:
                label = next(series.glob(self.label_pattern + "*.nii.gz"))
                sample["label"] = str(label)

            except StopIteration:
                pass

        return sample


class DistanceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.images = list(self.path.glob("*.nii.gz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": str(self.images[idx])}


def get_transformations(num_samples: int, crop_size=128, add_intensity_augmentation=False):
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


def get_dataloaders(
    path, num_samples: int, valid_size=0.3, crop_size=128, batch_size=1, add_intensity_augmentation=False
):
    path = Path(path)

    if path.is_file():

        with open(path, mode="rb") as file:
            split = pickle.load(file)

        train_data = split["train"]
        valid_data = split["validation"]

    else:
        data = list(DistanceDataset(path))
        assert len(data) > 1, f"Dataset too small. Found size: {len(data)}"

        train_data, valid_data = train_test_split(data, test_size=valid_size)

    train_transforms, valid_transforms = get_transformations(
        num_samples, crop_size, add_intensity_augmentation
    )

    train_data = Dataset(train_data, transform=train_transforms)
    valid_data = Dataset(valid_data, transform=valid_transforms)

    return DataLoader(train_data, batch_size=batch_size), DataLoader(valid_data)

import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm


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


def get_patient_name(path):
    return Path(path).parents[1].name


def split_autopet_data(dataset_path, valid_size=0.2, save=None):

    data = get_all_with_seg(DistDataset(dataset_path))

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

    # if save is not None:
    #     save_datasplit(_train_data, _valid_data, _test_data, save)

    return _train_data, _valid_data, _test_data

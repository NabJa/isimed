from typing import Callable, List

import monai.transforms as tfm
import numpy as np
import torch
from meddist.data.loading import read_data_split
from meddist.nets import load_latest_densenet
from meddist.transforms import GetClassesFromCropsd
from monai.data import DataLoader, Dataset, PatchIterd
from tqdm import tqdm


def determine_max_foreground_crop_size(
    data: List[dict],
    image_key: str = "image",
    select_fn: Callable = lambda x: x > 0,
    num_workers: int = 8,
) -> List:
    """Determine the maximum shape of some data after cropping the foreground.

    Args:
        data: List of dicts of paths. Typical monai data.
        image_key: Defaults to "image".
        select_fn: Defaults to lambda x: x > 0.
        num_workers: Passed to DataLoader. Defaults to 8.
    """
    crop = tfm.Compose(
        [
            tfm.LoadImaged(keys=image_key, ensure_channel_first=True),
            tfm.CropForegroundd(
                keys=image_key,
                source_key=image_key,
                select_fn=select_fn,
            ),
        ]
    )
    loader = DataLoader(Dataset(data, transform=crop), num_workers=num_workers)
    shapes = [x[image_key].shape for x in tqdm(loader, total=len(data))]
    return list(np.max(shapes, axis=0))[-3:]


class PatchDataset:
    def __init__(
        self,
        data,
        patch_size,
        roi_size=None,
        keys=("image", "label"),
        image_key: str = "image",
        pad_mode="minimum",
    ):
        self.data = data
        self.keys = keys

        if roi_size is None:
            self.roi_size = determine_max_foreground_crop_size(
                data, image_key=image_key
            )
        else:
            self.roi_size = roi_size

        assert len(self.roi_size) == 3

        self.patch_transform = tfm.Compose(
            [
                tfm.LoadImaged(keys=self.keys, ensure_channel_first=True),
                tfm.CenterSpatialCropd(keys=self.keys, roi_size=self.roi_size),
                tfm.ScaleIntensityRangePercentilesd(
                    keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
                ),
                PatchIterd(keys=self.keys, patch_size=[patch_size] * 3, mode=pad_mode),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return [x[0] for x in self.patch_transform(sample)]


def generate_dataset_representations(
    path_to_data_split, path_to_model_dir, output_path, split="train"
):

    # Get data
    train_images, valid_images, test_images = read_data_split(path_to_data_split)

    correct_split = {"train": train_images, "valid": valid_images, "test": test_images}

    data_laoder = DataLoader(
        PatchDataset(correct_split[split]), num_workers=1, batch_size=1
    )

    # Get model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_latest_densenet(path_to_model_dir).to(device).eval()

    # Generate representations
    all_representations = {}
    for sample in tqdm(data_laoder, total=len(correct_split[split])):
        sample_path = sample["image"]["filename_or_obj"][0]

        with torch.no_grad():
            representations = model(sample["image"].to(device)).detach().numpy()

        all_representations[sample_path] = representations

    return all_representations

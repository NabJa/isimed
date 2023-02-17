from functools import partial
from multiprocessing import Pool
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

    @property
    def npatches(self) -> int:
        sample = self.data[0]
        return len([x[0] for x in self.patch_transform(sample)])

    @property
    def npatients(self) -> int:
        return len(self.data)

    def __len__(self):
        return self.npatients

    def __getitem__(self, idx):
        sample = self.data[idx]
        return [x[0] for x in self.patch_transform(sample)]


@torch.no_grad()
def get_representation(sample, model, device, image_key):
    return model(sample[image_key].to(device)).detach().numpy()

def generate_dataset_representations(
    path_to_data_split, path_to_model_dir, patch_size=32, split="train", roi_size=None, num_workers=8
):

    # Get data
    train_images, valid_images, test_images = read_data_split(path_to_data_split)

    correct_split = {"train": train_images, "valid": valid_images, "test": test_images}

    patch_dataset = PatchDataset(correct_split[split][:10], patch_size, roi_size)
    data_laoder = DataLoader(patch_dataset)

    # Get model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_latest_densenet(path_to_model_dir).to(device).eval()


    # Generate representations
    _get_representation = partial(get_representation, model=model, device=device, image_key="image")   
    with Pool(num_workers) as p:
        all_representations = list(tqdm(p.imap(_get_representation, data_laoder), total=len(patch_dataset)))

    return np.array(all_representations)

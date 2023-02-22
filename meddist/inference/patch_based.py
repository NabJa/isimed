import argparse
from collections import defaultdict
from functools import partial
from typing import Callable, List

import monai.transforms as tfm
import numpy as np
import torch
from monai.data import DataLoader, Dataset, PatchIterd
from tqdm import tqdm

from meddist.data.loading import read_data_split
from meddist.nets import load_latest_densenet
from meddist.transforms import GetClassesFromCropsd

torch.multiprocessing.set_sharing_strategy("file_system")


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
        label_key: str = "label",
        pad_mode="minimum",
    ):
        self.data = data
        self.keys = keys
        self.label_key = label_key

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

        self.label_patches = tfm.Compose(
            [
                tfm.ToTensord(keys=self.keys),
                GetClassesFromCropsd(label_key=self.label_key),
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
        patches = [x[0] for x in self.patch_transform(sample)]
        return self.label_patches(patches)


@torch.no_grad()
def get_representation(sample, model, device, image_key):
    return model(sample[image_key].to(device)).detach().numpy()


def generate_dataset_representations(
    path_to_data_split,
    path_to_model_dir,
    patch_size=32,
    batch_size=1,
    num_workers=8,
    split="valid",
    roi_size=None,
) -> dict:

    # Get data
    train_images, valid_images, test_images = read_data_split(path_to_data_split)

    correct_split = {"train": train_images, "valid": valid_images, "test": test_images}

    patch_dataset = PatchDataset(correct_split[split], patch_size, roi_size)
    data_loader = DataLoader(
        patch_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Get model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_latest_densenet(path_to_model_dir).to(device).eval()

    # Generate representations
    repres = defaultdict(list)
    for batch in tqdm(data_loader, total=len(data_loader)):

        with torch.no_grad():
            repres["embeddings"].append(model(batch["image"].to(device)).cpu().numpy())

        repres["paths"].append(batch["image_meta_dict"]["filename_or_obj"])
        repres["has_pos_voxels"].append(batch["has_pos_voxels"].tolist())
        repres["num_pos_voxels"].append(batch["num_pos_voxels"].tolist())
        repres["patch_coords"].append(batch["patch_coords"].numpy())

    return {key: np.array(value) for key, value in repres.items()}


def save_dataset_representations(
    path_to_data_split, path_to_model_dir, output_path, **kwargs
):

    representations = generate_dataset_representations(
        path_to_data_split, path_to_model_dir, **kwargs
    )

    np.savez(output_path, **representations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path_to_split",
        type=str,
        required=True,
        help="Path to data split file containing image paths.",
    )
    parser.add_argument(
        "-path_to_model",
        type=str,
        required=True,
        help="Path to data model directory containing trained densenet.",
    )
    parser.add_argument(
        "-output_file_name",
        type=str,
        required=True,
        help="Path to file to save representations. Should end with .npz!",
    )
    parser.add_argument(
        "-split",
        type=str,
        default="valid",
        help="Choose between train, valid and test.",
    )
    parser.add_argument("-patch_size", type=int, default=32)
    args = parser.parse_args()

    save_dataset_representations(
        args.path_to_split,
        args.path_to_model,
        args.output_file_name,
        patch_size=args.patch_size,
        split=args.split,
    )

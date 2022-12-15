from pathlib import Path
from typing import List, Optional, Tuple

import monai.transforms as tfm
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from tqdm import tqdm
from wbpetct.data import FDG_PET_CT_Dataset


def get_shapes_and_spacings(data, image_key="image") -> Tuple[List, List]:
    shapes, spacings = [], []

    for sample in tqdm(data, desc="get_shapes_and_spacings"):
        img = nib.load(sample[image_key])
        shapes.append(img.shape)
        spacings.append(img.affine.diagonal()[:3])

    return shapes, spacings


def get_mean_image(loader, total, image_key="image", output_dir: Optional[Path] = None):
    mean_image = None

    for sample in tqdm(loader, total=total, desc="get_mean_image"):
        if mean_image is None:
            mean_image = sample[image_key] * (1 / total)
        else:
            mean_image += sample[image_key] * (1 / total)

    mean_image = mean_image[0]  # Remove batch dimension

    if output_dir is not None:
        with open(output_dir / "mean_image.th", mode="wb") as file:
            torch.save(mean_image, file)

    return mean_image


def main(data, output_dir: Path, image_key="image", num_workers=0):

    # Prepare output dir
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Prepare all images to median shape
    shapes, _ = get_shapes_and_spacings(data, image_key)
    median_shape = list(np.median(shapes, axis=0).astype(int))

    load_to_median_size = tfm.Compose(
        [
            tfm.LoadImaged(keys=image_key),
            tfm.EnsureChannelFirstd(keys=image_key),
            tfm.ResizeWithPadOrCropd(keys=image_key, spatial_size=median_shape),
            tfm.SaveImaged(
                keys=image_key,
                output_dir=str(processed_dir),
                resample=False,
                print_log=False,
                separate_folder=False,
                output_postfix="",
            ),
        ]
    )

    # Generate mean image and save processed images
    loader = DataLoader(
        Dataset(data, transform=load_to_median_size),
        num_workers=num_workers,
        batch_size=1,
    )

    mean_image = get_mean_image(
        loader, total=len(data), image_key=image_key, output_dir=output_dir
    )

    # Rigid registration

    # Crop
    # Dataset Summary
    # Preproecssing


if __name__ == "__main__":
    data_path = "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions"
    output_dir = Path(
        "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions-data"
    )
    data = FDG_PET_CT_Dataset(data_path)

    main(data, output_dir, image_key="ct", num_workers=8)

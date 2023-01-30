import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import monai.transforms as tfm
import nibabel as nib
import numpy as np
import torch
from meddist.data import DistDataset, split_segmentation_data
from meddist.registration import DirectoryRegistration
from monai.data import DataLoader, Dataset
from tqdm import tqdm
from wbpetct.data import FDG_PET_CT_Dataset

torch.multiprocessing.set_sharing_strategy("file_system")


def all_same_values(values) -> bool:
    return len(set(values)) <= 1


def get_shapes_and_spacings(
    data, image_keys: List[str] = ["ct", "seg"]
) -> Tuple[np.ndarray, np.ndarray]:
    shapes, spacings = [], []

    for sample in tqdm(data, desc="get_shapes_and_spacings"):
        images = [nib.load(sample[key]) for key in image_keys]

        assert all_same_values(
            [x.shape for x in images]
        ), f"Not all images have the same shape! {sample}"
        assert (
            all_same_values([tuple(x.affine.diagonal()[:3]) for x in images]) <= 1
        ), f"Not all images have the same spacing! {sample}"

        shapes.append(images[0].shape)
        spacings.append(images[0].affine.diagonal()[:3])

    return np.array(shapes), np.array(spacings)


def get_mean_image(loader, total, source_key="ct", output_dir: Optional[Path] = None):

    # Load if mean image already exists
    if output_dir is not None:
        out_file_name = output_dir / "mean_image.th"
        if out_file_name.is_file():
            with open(out_file_name, mode="rb") as file:
                logging.info("Loading existing mean image.")
                return torch.load(file)

    # Generate mean image
    logging.info("Generating mean image")
    mean_image = None
    for sample in tqdm(loader, total=total, desc="get_mean_image"):
        if mean_image is None:
            mean_image = sample[source_key] * (1 / total)
        else:
            mean_image += sample[source_key] * (1 / total)

    mean_image = mean_image[0]  # Remove batch dimension

    # Save mean image
    if output_dir is not None:
        with open(output_dir / "mean_image.th", mode="wb") as file:
            torch.save(mean_image, file)

    return mean_image


def preprocess(
    data,
    output_dir: Path,
    data_root_dir: Path,
    image_keys=["ct", "seg"],
    source_key="ct",
    num_workers=0,
):

    # Prepare output dir
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(output_dir / "processing.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        force=True,
    )

    logging.info("Getting all shapes and spacings...")
    shapes, _ = get_shapes_and_spacings(data, image_keys)

    MAX_DEPTH = 400
    samples_with_accaptable_shape = shapes[:, 2] <= MAX_DEPTH
    data = data[samples_with_accaptable_shape]
    logging.info(
        f"Filter samples with unsuitable size. Reduced from {shapes.shape[0]} to {len(data)} samples."
    )

    median_shape = list(
        np.median(shapes[samples_with_accaptable_shape], axis=0).astype(int)
    )
    logging.info(f"Median shape is {median_shape}")

    load_to_median_size = tfm.Compose(
        [
            tfm.LoadImaged(keys=image_keys),
            tfm.EnsureChannelFirstd(keys=image_keys),
            tfm.ResizeWithPadOrCropd(keys=image_keys, spatial_size=median_shape),
            tfm.SaveImaged(
                keys=image_keys,
                output_dir=str(processed_dir),
                resample=False,
                print_log=False,
                separate_folder=False,
                output_postfix="",
                data_root_dir=str(data_root_dir),
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
        loader, total=len(data), source_key=source_key, output_dir=output_dir
    )

    return mean_image


def parse_args():
    data_path = "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions"
    output_dir = Path(
        "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions-trans"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=Path, default=data_path)
    parser.add_argument("-output", type=Path, default=output_dir)
    parser.add_argument("-disable_preprocess", action="store_true", default=False)
    parser.add_argument("-disable_register", action="store_true", default=False)
    parser.add_argument("-disable_split", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    data = FDG_PET_CT_Dataset(args.input)

    # # Preprocessing
    if not args.disable_preprocess:
        mean_image = preprocess(
            data,
            args.output,
            data_root_dir=args.input,
            image_keys=["ct", "seg"],
            num_workers=8,
        )

    # # Registration
    dirreg = DirectoryRegistration(args.output)
    if not args.disable_register:
        dirreg.run_registration()

    # # Data split into train, valid and test set
    if not args.disable_split:
        _ = split_segmentation_data(DistDataset(dirreg.output_dir), save=args.output)

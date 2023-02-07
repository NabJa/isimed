from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

BRATS_MODALITY_CODE = {
    "t1": "0000",
    "t1ce": "0001",
    "t2": "0002",
    "flair": "0003",
}


def extract_brats_modality(path) -> str:
    """Return MODALITY from images of format some/directory/IMAGEID_MODALITY.nii.gz"""
    return Path(path).name.split(".")[0].split("_")[1]


def extract_brats_image_id(path) -> str:
    """Return IMAGEID from images of format some/directory/IMAGEID_MODALITY.nii.gz"""
    return Path(path).name.split("_")[0]


def image_path_to_label_path(image_path, label_dir_name="labelsTr") -> Path:
    """Return the label path for a corresponding image.
    Labels schould be placed in a directory (label_dir_name) that have the same parent diectory as the images."""
    image_path = Path(image_path)
    image_id = extract_brats_image_id(image_path)
    return image_path.parents[1] / label_dir_name / f"{image_id}.nii.gz"


def get_sample_from_image_path(image_path, label_dir_name="labelsTr") -> dict:
    """Return image and label dictionary containing the image path and corresponding label path."""
    return {
        "image": str(image_path),
        "label": str(image_path_to_label_path(image_path, label_dir_name)),
    }


def get_brats_images(
    raw_data_path, modality="t1", label_dir_name="labelsTr"
) -> List[dict]:
    """Return a list with one dict for every patient containing an image path and a label path."""

    image_dir = Path(raw_data_path) / "imagesTr"

    return [
        get_sample_from_image_path(image_path, label_dir_name)
        for image_path in image_dir.glob("*.nii.gz")
        if extract_brats_modality(image_path) == BRATS_MODALITY_CODE[modality]
    ]


def split_brats_data(
    raw_data_path,
    label_dir_name="labelsTr",
    valid_size=0.2,
    test_size=0.2,
    modality="t1",
    seed=42,
):
    """Split BraTS data into training, validation and test data.
    Every dataset containg a list with dicts containing image and label path."""

    images = get_brats_images(raw_data_path, modality, label_dir_name)

    train_images, valid_test_images = train_test_split(
        images, test_size=valid_size + test_size, random_state=seed
    )

    test_images, valid_images = train_test_split(
        valid_test_images,
        test_size=valid_size / (valid_size + test_size),
        random_state=seed,
    )

    return train_images, valid_images, test_images

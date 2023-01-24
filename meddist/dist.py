from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from monai.data.meta_tensor import MetaTensor


def find_key(data: dict, key: str) -> any:
    """
    This function recursively searches a given dictionary for a specified key,
    and returns the value of the key if it is found, or None if the key is not found.

    Parameters:
        data: The dictionary to search
        key: The key to search for

    Returns:
        The value of the key if found, or None if not found
    """
    if key in data:
        return data[key]
    for v in data.values():
        if isinstance(v, dict):
            result = find_key(v, key)
            if result is not None:
                return result
    return None


def extract_operation_class(meta, operation_class_name="RandSpatialCropSamples"):
    operation_dicts = []
    for image_operations in meta.applied_operations:
        for operation in image_operations:
            if operation["class"] == operation_class_name:
                operation_dicts.append(operation)
    return operation_dicts


def transform_bbox_formats(crop_dicts: List[dict]) -> np.ndarray:
    """Take all crop operations and retuns them as array with the format (nsamples, 6). Where 6 is [height_start, heigth_end, ...]"""

    if len(crop_dicts) == 0:
        return None

    orig_sizes = np.array([find_key(crop, "orig_size") for crop in crop_dicts])
    croppings = np.array([find_key(crop, "cropped") for crop in crop_dicts])

    bboxes = np.zeros((len(crop_dicts), 6))
    bboxes[:, ::2] = croppings[:, ::2]
    bboxes[:, 1::2] = orig_sizes - croppings[:, 1::2]

    return bboxes.astype(int)


def get_cropped_bboxes(
    meta: MetaTensor, crop_class: str = "RandSpatialCrop"
) -> np.ndarray:
    """
    Given a MetaTensor with cropping operations applied, extract the bounding boxes in format:
        [hight_start, hight_end, width_start, width_end, depth_start, depth_end]
    """

    return transform_bbox_formats(extract_operation_class(meta, crop_class))


def get_bbox_centers(bbox: np.ndarray) -> np.ndarray:
    """Center is computed as starts + (ends - starts) / 2"""
    bbox = np.array(bbox)
    
    if len(bbox.shape) == 1:
        return bbox[::2] + (bbox[1::2] - bbox[::2]) / 2
    if len(bbox.shape) == 2:
        return bbox[:, ::2] + (bbox[:, 1::2] - bbox[:, ::2]) / 2
    raise ValueError("Bboxes must be 1 dimensional or 2 dimenstional.")


def pairwise_comparisons(nsamples: int) -> List:
    """
    Generates all possible pairwise combinations of integers from 0 to nsamples-1.

    Parameters:
        nsamples (int): The number of samples for which to generate pairwise combinations.
    Returns:
        List: A list of tuples, where each tuple contains a pair of integers from 0 to nsamples-1.
        E.g.: [(0, 1), (0, 2), ...]

    """
    return list(combinations(np.arange(nsamples), 2))


def euclidean_dist(
    coordinates: np.ndarray, return_indices: bool = False
) -> Tuple[List, Optional[List]]:
    """Takes coordinates of shape (N, 3) in 3D case and (N, 2) in 2D case"""

    def dist(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2))

    comparisions = pairwise_comparisons(coordinates.shape[0])

    # Get all distances. TODO allow different distance function.
    distances = [dist(coordinates[i], coordinates[j]) for i, j in comparisions]

    if return_indices:
        return distances, comparisions
    return distances

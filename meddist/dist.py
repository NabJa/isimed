from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from monai.data.meta_tensor import MetaTensor


def get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Center is computed as starts + (ends - starts) / 2"""
    bbox = np.array(bbox)
    return bbox[::2] + (bbox[1::2] - bbox[::2]) / 2


def euclidean_dist(coordinates: np.ndarray, return_indices: bool = False) -> Tuple[List, Optional[List]]:
    """Takes coordinates of shape (N, 3) in 3D case and (N, 2) in 2D case"""

    def dist(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2))

    # All pairwise combinations between samples. E.g.: [(0, 1), (0, 2), ...]
    comparisions = combinations(np.arange(coordinates.shape[0]), 2)
    comparisions = list(comparisions)

    # Get all distances. TODO allow different distance function.
    distances = [dist(coordinates[i], coordinates[j]) for i, j in comparisions]

    if return_indices:
        return distances, comparisions
    return distances


def get_cropped_bboxes(meta: MetaTensor, crop_class: str = "RandSpatialCrop") -> np.ndarray:
    """
    Given a MetaTensor with cropping operations applied, extract the bounding boxes in format:
        [hight_start, hight_end, width_start, width_end, depth_start, depth_end]
    """

    # Find meta information for all applied crop classes
    crop_meta = [
        x for x in meta.applied_operations if x["class"].startswith(crop_class)
    ]

    # Crop info is stored in the format: [height_crop_left, height_crop_right, ...]
    crop_info = [np.array(x["extra_info"]["extra_info"]["cropped"]) for x in crop_meta]

    # Size of the image before cropping
    size_info = [np.array(x["orig_size"]) for x in crop_meta]

    # Compute bboxes with format  [hight_start, hight_end, ...]
    bboxes = []
    for crop, size in zip(crop_info, size_info):
        bbox = np.zeros(6)  # Assumption: 3D. In 2D this would be 4.
        bbox[::2] = 2 * crop[::2]
        bbox[1::2] = size
        bbox -= crop
        bboxes.append(bbox.astype(int))

    return bboxes[0]  # Could not find a case where there are multiple bbox in one meta

from typing import Callable, List

import monai.transforms as tfm
import numpy as np
from meddist.data.loading import read_data_split
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
    return list(np.max(shapes, axis=0))

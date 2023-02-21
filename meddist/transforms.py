from copy import deepcopy

import torch
from monai.config import KeysCollection
from monai.transforms import (
    MapTransform,
    RandCropByPosNegLabel,
    Randomizable,
    RandSpatialCropSamples,
)
from monai.transforms.utils import map_binary_to_indices


class RandCropBlanacedd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding metadata.
    And will return a list of dictionaries for all the cropped images.
    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center
    to ensure the valid crop ROI.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.
    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key=None,
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.allow_smaller = allow_smaller

        self.seen_pos = 0
        self.seen_neg = 0

    def update_pos_neg(self):
        self.pos = (self.seen_neg + 1) / (self.seen_pos + 1)
        self.neg = (self.seen_pos + 1) / (self.seen_neg + 1)

    def get_cropper(self, label, image):

        self.update_pos_neg()

        fg_indices, _ = map_binary_to_indices(label, image, self.image_threshold)

        if len(fg_indices) == 0:
            return RandSpatialCropSamples(
                roi_size=self.spatial_size,
                num_samples=self.num_samples,
                random_size=False,
            )

        return RandCropByPosNegLabel(
            spatial_size=self.spatial_size,
            pos=self.pos,
            neg=self.neg,
            num_samples=self.num_samples,
            image_threshold=self.image_threshold,
            allow_smaller=self.allow_smaller,
        )

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None

        cropper = self.get_cropper(label, image)

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        for key in self.key_iterator(d):

            if isinstance(cropper, RandSpatialCropSamples):
                crops = cropper(d[key])
            else:
                crops = cropper(d[key], label=label, randomize=True)

            if key == self.label_key:
                for crop in crops:
                    if torch.sum(crop.flatten()) > 0:
                        self.seen_pos += 1
                    else:
                        self.seen_neg += 1

            for i, im in enumerate(crops):
                ret[i][key] = im
        return ret


class GetClassesFromCropsd(MapTransform):
    def __init__(
        self, label_key, class_key="has_pos_voxels", regression_key="num_pos_voxels"
    ):
        self.label_key = label_key
        self.class_key = class_key
        self.regression_key = regression_key

    def __call__(self, data):
        if isinstance(data, list):
            for sample in data:
                num_pos_voxels = torch.sum(sample[self.label_key].flatten()).int()
                sample[self.regression_key] = num_pos_voxels.item()
                sample[self.class_key] = (num_pos_voxels > 0).int().item()
        else:
            num_pos_voxels = torch.sum(data[self.label_key].flatten()).int()
            data[self.regression_key] = num_pos_voxels.item()
            data[self.class_key] = (num_pos_voxels > 0).int().item()
        return data

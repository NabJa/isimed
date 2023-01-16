from pathlib import Path

import monai.transforms as tfm
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

NUM_SAMPLES = 64


class DistanceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.images = list(self.path.glob("*.nii.gz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": str(self.images[idx])}


def get_dataloaders(path, valid_size=0.3):

    transforms = tfm.Compose(
        [
            tfm.LoadImaged(keys="image", ensure_channel_first=True),
            tfm.CropForegroundd(
                keys="image", source_key="image", select_fn=lambda x: x > -1000
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandSpatialCropSamplesd(
                keys="image", roi_size=128, num_samples=NUM_SAMPLES, random_size=False
            ),
        ]
    )

    train_data, valid_data = train_test_split(
        list(DistanceDataset(path)), test_size=valid_size
    )
    train_data, valid_data = Dataset(train_data, transform=transforms), Dataset(
        valid_data, transform=transforms
    )

    train_loader, valid_loader = DataLoader(train_data), DataLoader(valid_data)

    return train_loader, valid_loader

import monai.transforms as tfm
import torch
import wandb
from meddist.data.loading import get_dataloaders
from monai.losses import ContrastiveLoss


def get_contrastive_transform(crops: int = 2, crop_size: int = 32):
    return tfm.Compose(
        [
            tfm.LoadImaged(keys="image", ensure_channel_first=True),
            tfm.CropForegroundd(
                keys="image", source_key="image", select_fn=lambda x: x > 0
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[crop_size] * 3,
                random_size=False,
                num_samples=crops,
            ),
            tfm.CopyItemsd(
                keys=["image"],
                times=1,
                names=["image_2"],
                allow_missing_keys=False,
            ),
            tfm.OneOf(
                transforms=[
                    tfm.RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.33),
                        dropout_holes=True,
                    ),
                    tfm.RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.66),
                        dropout_holes=False,
                    ),
                ]
            ),
            tfm.RandCoarseShuffled(keys=["image"], prob=0.8, holes=2, spatial_size=6),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            tfm.OneOf(
                transforms=[
                    tfm.RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.33),
                        dropout_holes=True,
                    ),
                    tfm.RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=int(crop_size * 0.1),
                        max_spatial_size=int(crop_size * 0.66),
                        dropout_holes=False,
                    ),
                ]
            ),
            tfm.RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=2, spatial_size=6),
        ]
    )


def forward_simclr(model, batch, loss_fn, mode=None, device="cuda"):
    # Prepare data
    inputs = batch["image"].to(device)
    inputs_2 = batch["image_2"].to(device)

    # Forward pass
    with torch.autocast(device_type=device):
        emb1: torch.Tensor = model(inputs)
        emb2: torch.Tensor = model(inputs_2)

    # Get loss
    emb1, emb2 = emb1.to("cpu").float(), emb2.to("cpu").float()
    loss = loss_fn(emb1, emb2)

    return loss, {}


def prepare_simclr(path_to_data_split):
    loss_fn = ContrastiveLoss(temperature=wandb.config.temperature)

    train_loader, valid_loader = get_dataloaders(
        path_to_data_split,
        batch_size=wandb.config.batch_size,
        train_transform=get_contrastive_transform(
            crops=wandb.config.number_of_crops, crop_size=wandb.config.crop_size
        ),
        valid_transform=get_contrastive_transform(
            crops=wandb.config.number_of_crops, crop_size=wandb.config.crop_size
        ),
        num_workers=wandb.config.num_workers,
    )

    return loss_fn, train_loader, valid_loader

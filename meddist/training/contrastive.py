from pathlib import Path

import monai.transforms as tfm
import torch
import wandb
from meddist import downstram_class
from meddist.data.loading import get_dataloaders
from meddist.training.logs import CheckpointSaver, MetricTracker
from monai.losses import ContrastiveLoss
from monai.networks.nets import DenseNet
from torch.optim import Adam, lr_scheduler


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


def run_epoch(model, loss_fn, dataloader, optimizer=None):
    mode = "valid" if optimizer is None else "train"

    tracker = MetricTracker(f"{mode}/Loss")

    for batch in dataloader:

        # Prepare data
        inputs = batch["image"].to("cuda")
        inputs_2 = batch["image_2"].to("cuda")

        # Prepare forward pass
        if mode == "train":
            optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type="cuda"):
            emb1: torch.Tensor = model(inputs)
            emb2: torch.Tensor = model(inputs_2)

        # Get loss
        emb1, emb2 = emb1.to("cpu").float(), emb2.to("cpu").float()
        loss = loss_fn(emb1, emb2)

        # Backward pass and optimization
        if mode == "train":
            loss.backward()
            optimizer.step()

        if mode == "train":
            wandb.log(
                {
                    f"{mode}/Loss": loss.item(),
                }
            )
        else:
            tracker(loss.item())

    # Free up all memory
    torch.cuda.empty_cache()

    metrics = tracker.aggregate()

    if mode == "valid":
        wandb.log(metrics, commit=False)

    return metrics[f"{mode}/Loss"]


def train(path_to_data_split, model_log_path):

    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    ).to("cuda")

    optimizer = Adam(model.parameters(), lr=wandb.config.lr)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

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
    )

    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    # wandb.watch(model, log_freq=1000, log="all", log_graph=False)

    for epoch in range(wandb.config.epochs):

        wandb.log({"LR": optimizer.param_groups[0]["lr"]}, commit=False)

        _ = run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        if epoch > wandb.config.epochs * 0.3:
            saver(model, epoch, valid_loss)

        scheduler.step()

    if (epoch + 1) % wandb.config.downstream_every_n_epochs == 0:
        downstram_class.train(path_to_data_split, model_log_path)

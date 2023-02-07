from pathlib import Path

import monai.transforms as tfm
import torch
import wandb
from meddist.data.loading import get_dataloaders
from meddist.train import CheckpointSaver, MetricTracker
from monai.networks.nets import DenseNet
from torch import nn
from torch.optim import Adam, lr_scheduler


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, y1, y2, d=0):
        """
        d = 0 means y1 and y2 are supposed to be same
        d = 1 means y1 and y2 are supposed to be different
        """      

        euc_dist = nn.functional.pairwise_distance(y1, y2, p=2.0)
        if d == 1:
            euc_dist = self.margin - euc_dist  # sort of reverse distance
            euc_dist = torch.clamp(euc_dist, min=0.0, max=None)

        return torch.mean(torch.pow(euc_dist, 2))


def heavy_augment(num_samples: int, crop_size=128):
    return [
        tfm.LoadImaged(keys="image", ensure_channel_first=True),
        tfm.CropForegroundd(
            keys="image", source_key="image", select_fn=lambda x: x > 0
        ),
        tfm.ScaleIntensityRangePercentilesd(
            keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
        ),
        tfm.RandGaussianNoised(keys="image", prob=0.4, mean=0.0, std=0.1),
        tfm.RandGaussianSmoothd(keys="image", prob=0.4),
        tfm.RandCoarseShuffled(
            keys="image", holes=1, max_holes=10, prob=0.5, spatial_size=8
        ),
        tfm.RandSpatialCropSamplesd(
            keys="image", roi_size=crop_size, num_samples=num_samples, random_size=False
        ),
    ]


def run_epoch(model, loss_fn, dataloader, optimizer=None):
    mode = "valid" if optimizer is None else "train"

    tracker = MetricTracker(f"{mode}/Loss")

    for iteration, batch in enumerate(dataloader):

        # Prepare data
        image = batch["image"]
        image_v1 = heavy_augment(image).to("cuda")
        image_v2 = heavy_augment(image).to("cuda")

        # Prepare forward pass
        if mode == "train":
            optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type="cuda"):
            emb1: torch.Tensor = model(image_v1)
            emb2: torch.Tensor = model(image_v2)

        # Get loss
        emb1, emb2 = emb1.to("cpu"), emb2.to("cpu")
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


def train(out_channels: int = 1024):

    model = DenseNet(spatial_dims=3, in_channels=1, out_channels=out_channels)

    optimizer = Adam(model.parameters())

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    loss_fn = ContrastiveLoss(margin=2)

    train_loader, valid_loader = get_dataloaders(
        wandb.config.path_to_data_split,
        num_samples=wandb.config.number_of_crops,
        crop_size=wandb.config.crop_size,
        add_intensity_augmentation=wandb.config.augment,
        batch_size=wandb.config.batch_size,
    )

    model_log_path = Path(wandb.config.output_path) / f"{wandb.run.name}_{wandb.run.id}"
    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    wandb.watch(model, log_freq=1000, log="all", log_graph=True)

    for epoch in range(wandb.config.epochs):

        wandb.log({"LR": optimizer.param_groups[0]["lr"]}, commit=False)

        _ = run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        if epoch > wandb.config.epochs * 0.3:
            saver(model, epoch, valid_loss)

        scheduler.step()

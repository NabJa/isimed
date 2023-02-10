from typing import Tuple

import torch
import wandb
from meddist import downstram_class
from meddist.data.loading import get_dataloaders
from meddist.training.contrastive import get_contrastive_transform
from meddist.training.logs import CheckpointSaver, MetricTracker
from monai.data import MetaTensor
from monai.networks.nets import DenseNet
from torch import nn
from torch.optim import Adam, lr_scheduler


def off_diagonal(x) -> torch.Tensor:
    "A flattened view of the off-diagonal elements of a square matrix"
    n, m = x.shape
    assert n == m

    if isinstance(x, MetaTensor):
        x = x.as_tensor()

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTLoss(nn.Module):
    def __init__(self, embedding_size, lambd: float = 0.0051):
        super().__init__()
        self.lambd = lambd
        self.embedding_size = embedding_size

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(embedding_size, affine=False)

    def __call__(self, z1, z2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = z1.shape[0]

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        invariant_loss = torch.diagonal(c).add_(-1).pow_(2).sum() / self.embedding_size
        covariant_loss = off_diagonal(c).pow_(2).sum() / self.embedding_size
        loss = invariant_loss + self.lambd * covariant_loss
        return loss, invariant_loss, covariant_loss


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
        loss, invariant_loss, covariant_loss = loss_fn(emb1, emb2)

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

    loss_fn = BTLoss(embedding_size=wandb.config.embedding_size)

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

        if (epoch + 1) % wandb.config.downstream_every_n_epochs == 0:
            downstram_class.train(model_log_path)

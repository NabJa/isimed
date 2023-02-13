from typing import Tuple

import torch
import wandb
from meddist.data.loading import get_dataloaders
from meddist.training.contrastive import get_contrastive_transform
from monai.data import MetaTensor
from torch import nn


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


def forward_barlow(model, batch, loss_fn, mode, device="cuda"):
    # Prepare data
    inputs = batch["image"].to(device)
    inputs_2 = batch["image_2"].to(device)

    # Forward pass
    with torch.autocast(device_type=device):
        emb1: torch.Tensor = model(inputs)
        emb2: torch.Tensor = model(inputs_2)

    # Get loss
    emb1, emb2 = emb1.to("cpu").float(), emb2.to("cpu").float()
    loss, invariant_loss, covariant_loss = loss_fn(emb1, emb2)

    return loss, {f"{mode}/invariant_loss": invariant_loss.item(), f"{mode}/covariant_loss": covariant_loss.item()}



def prepare_barlow(path_to_data_split):
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
        num_workers=wandb.config.num_workers
    )

    return loss_fn, train_loader, valid_loader

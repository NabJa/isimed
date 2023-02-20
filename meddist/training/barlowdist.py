import torch
import wandb
from meddist.data.loading import get_contrastive_transform, get_dataloaders
from meddist.training.barlow import BTLoss
from meddist.training.phys import DistanceLoss
from torch import nn


class BarlowDist(nn.Module):
    def __init__(
        self, embedding_size: int = 1024, barlow_weight: float = 10_000
    ) -> None:
        super().__init__()
        self.distance_loss_fn = DistanceLoss()
        self.bt_loss_fn = BTLoss(embedding_size)
        self.barlow_weight = barlow_weight

    def forward(self, image, embeddings_1, embeddings_2):
        embeddings = (embeddings_1 + embeddings_2) / 2

        distance_loss = self.distance_loss_fn(image, embeddings)
        bt_loss, _, _ = self.bt_loss_fn(embeddings_1, embeddings_2)

        return distance_loss + self.barlow_weight * bt_loss, distance_loss, bt_loss


def forward_barlow_dist(model, batch, loss_fn, mode, device="cuda"):

    # Forward pass
    with torch.autocast(device_type=device):
        emb1: torch.Tensor = model(batch["image"].to(device))
        emb2: torch.Tensor = model(batch["image_2"].to(device))

    # Get loss
    emb1, emb2 = emb1.to("cpu").float(), emb2.to("cpu").float()

    loss, distance_loss, bt_loss = loss_fn(batch["image"].to("cpu").float(), emb1, emb2)

    return loss, {
        f"{mode}/Meddist_loss": distance_loss.item(),
        f"{mode}/Barlow_loss": bt_loss.item(),
    }


def prepare_barlow_dist(path_to_data_split):
    loss_fn = BarlowDist(
        embedding_size=wandb.config.embedding_size,
        barlow_weight=wandb.config.temperature,
    )

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

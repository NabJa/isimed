import torch
from torch import nn

import wandb
from meddist.data.loading import get_contrastive_transform, get_dataloaders
from meddist.training.barlow import BTLoss
from meddist.training.phys import DistanceLoss

BT_EMBEDDING_SIZE = 2048


class DistanceScaledBTHead(nn.Module):
    """Make two heads to optimize dist and bt seperatly."""

    def __init__(
        self,
        backbone,
        emb_size=1024,
        dist_embedding=512,
        bt_embedding=BT_EMBEDDING_SIZE,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.linear_dist = nn.Linear(emb_size, dist_embedding)
        self.linear_bt = nn.Linear(emb_size, bt_embedding)

    def forward(self, x):
        emb = self.backbone(x)
        return self.linear_bt(emb), self.linear_dist(emb), emb


class DistanceScaledHeadLoss(nn.Module):
    def __init__(self, embedding_size=1024, barlow_weight=10_000) -> None:
        super().__init__()
        self.barlow_weight = barlow_weight
        self.bt_loss_fn = BTLoss(embedding_size)
        self.dist_loss_fn = DistanceLoss()

    def forward(self, image, bt_emb1, bt_emb2, dist_emb1, dist_emb2):

        dist_loss_1 = self.dist_loss_fn(image, dist_emb1)
        dist_loss_2 = self.dist_loss_fn(image, dist_emb2)
        dist_loss = (dist_loss_1 + dist_loss_2) / 2

        bt_loss, _, _ = self.bt_loss_fn(bt_emb1, bt_emb2)

        return dist_loss + self.barlow_weight * bt_loss


def forward_hydra(model, batch, loss_fn, mode, device="cuda"):

    # Forward pass
    with torch.autocast(device_type=device):
        bt_emb1, dist_emb1, _ = model(batch["image"].to(device))
        bt_emb2, dist_emb2, _ = model(batch["image_2"].to(device))

    # Get loss
    bt_emb1, dist_emb1 = bt_emb1.to("cpu").float(), dist_emb1.to("cpu").float()
    bt_emb2, dist_emb2 = bt_emb2.to("cpu").float(), dist_emb2.to("cpu").float()

    image = batch["image"].to("cpu").float()
    loss = loss_fn(image, bt_emb1, bt_emb2, dist_emb1, dist_emb2)

    return loss, {}


def prepare_hydra(path_to_data_split):
    loss_fn = DistanceScaledHeadLoss(
        embedding_size=BT_EMBEDDING_SIZE,
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

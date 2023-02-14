import torch
import wandb
from meddist.data.loading import get_contrastive_transform, get_dataloaders
from monai.losses import ContrastiveLoss


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

import torch
import wandb
from meddist.data.loading import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from torch import nn


class DistanceKLMSELoss(nn.Module):
    def __init__(self, eppsilon=2.0, mu=0.0, sd=1.0):
        super().__init__()

        self.eppsilon = eppsilon
        self.target_dist = torch.distributions.Normal(mu, sd)

        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, pred_dist, target_dist, embedding):

        normal_dist = self.target_dist.sample(embedding.shape)

        kl_loss = torch.abs(self.kldiv(embedding, normal_dist))
        mse_loss = self.mse(pred_dist, target_dist)

        total_loss = mse_loss + self.eppsilon * kl_loss

        return total_loss, kl_loss, mse_loss


def get_max_distance(dist_mat1: torch.Tensor, dist_mat2: torch.Tensor) -> float:
    return torch.max(torch.abs(dist_mat1 - dist_mat2)).item()


def forward_meddist(model, batch, loss_fn, mode):
    # Prepare GT
    bboxes = get_cropped_bboxes(batch["image"], "RandSpatialCropSamples")
    centers = get_bbox_centers(bboxes)

    gt_dist_mat = torch.cdist(
        torch.tensor(centers), torch.tensor(centers), p=2.0
    ).float()

    image = batch["image"].to("cuda")

    # Forward pass
    with torch.autocast(device_type="cuda"):
        embeddings: torch.Tensor = model(image)

    # Get loss
    embeddings = embeddings.to("cpu", dtype=torch.float32)
    pred_dist_mat = torch.cdist(embeddings, embeddings, p=2)
    total_loss, kl_loss, mse_loss = loss_fn(pred_dist_mat, gt_dist_mat, embeddings)

    # Get metrics that could be logged
    max_distance = get_max_distance(pred_dist_mat, gt_dist_mat)
    metrics = {
        f"{mode}/KLDLoss": kl_loss.item(),
        f"{mode}/MSELoss": mse_loss.item(),
        f"{mode}/MaxDistance": max_distance,
    }

    return total_loss, metrics


def prepare_meddist(path_to_data_split):
    # Define the loss function
    loss_fn = DistanceKLMSELoss(eppsilon=wandb.config.temperature)

    # Define the DataLoader
    train_loader, valid_loader = get_dataloaders(
        path_to_data_split,
        num_samples=wandb.config.number_of_crops,
        crop_size=wandb.config.crop_size,
        add_intensity_augmentation=wandb.config.augment,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.num_workers
    )

    return loss_fn, train_loader, valid_loader

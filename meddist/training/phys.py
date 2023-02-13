import torch
import wandb
from meddist import downstram_class
from meddist.data.loading import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from meddist.training.logs import CheckpointSaver, MetricTracker
from monai.networks.nets import DenseNet
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


def run_epoch(model, loss_fn, dataloader, optimizer=None) -> None:

    mode = "valid" if optimizer is None else "train"

    tracker = MetricTracker(
        f"{mode}/Loss", f"{mode}/KLLoss", f"{mode}/MSELoss", f"{mode}/MaxDistance"
    )

    for batch in dataloader:

        # Prepare GT
        bboxes = get_cropped_bboxes(batch["image"], "RandSpatialCropSamples")
        centers = get_bbox_centers(bboxes)

        gt_dist_mat = torch.cdist(
            torch.tensor(centers), torch.tensor(centers), p=2.0
        ).float()

        # Prepare forward pass
        if mode == "train":
            optimizer.zero_grad()
        image = batch["image"].to("cuda")

        # Forward pass
        with torch.autocast(device_type="cuda"):
            embeddings: torch.Tensor = model(image)

        # Get loss
        embeddings = embeddings.to("cpu", dtype=torch.float32)
        pred_dist_mat = torch.cdist(embeddings, embeddings, p=2)
        total_loss, kl_loss, mse_loss = loss_fn(pred_dist_mat, gt_dist_mat, embeddings)

        # Backward pass and optimization
        if mode == "train":
            total_loss.backward()
            optimizer.step()

        # Log
        max_distance = get_max_distance(pred_dist_mat, gt_dist_mat)

        if mode == "train":
            wandb.log(
                {
                    f"{mode}/Loss": total_loss.item(),
                    f"{mode}/KLDLoss": kl_loss.item(),
                    f"{mode}/MSELoss": mse_loss.item(),
                    f"{mode}/MaxDistance": max_distance,
                }
            )

        tracker(total_loss.item(), kl_loss.item(), mse_loss.item(), max_distance)

    # Free up all memory
    torch.cuda.empty_cache()

    metrics = tracker.aggregate()
    if mode == "valid":
        wandb.log(metrics, commit=False)

    return metrics[f"{mode}/Loss"]


def train(path_to_data_split, model_log_path):

    # Define the model and optimizer
    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9, verbose=True
    )

    # Define the loss function
    loss_fn = DistanceKLMSELoss(eppsilon=wandb.config.temperature)

    # Define the DataLoader
    train_loader, valid_loader = get_dataloaders(
        path_to_data_split,
        num_samples=wandb.config.number_of_crops,
        crop_size=wandb.config.crop_size,
        add_intensity_augmentation=wandb.config.augment,
        batch_size=wandb.config.batch_size,
    )

    # Define checkpoint saver
    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    # Start training

    # wandb.watch(model, log_freq=1000, log="all", log_graph=False)

    for epoch in range(wandb.config.epochs):

        wandb.log({"LR": optimizer.param_groups[0]["lr"]}, commit=False)

        _ = run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        # if epoch > wandb.config.epochs * 0.3:
        saver(model, epoch, valid_loss)

        scheduler.step()

        if wandb.config.run_downsream_task:
            if (epoch + 1) % wandb.config.downstream_every_n_epochs == 0:
                downstram_class.train(path_to_data_split, model_log_path)

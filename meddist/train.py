import logging
from pathlib import Path

import numpy as np
import torch
import wandb
from meddist.config import init_wandb
from meddist.data import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from monai.networks.nets import DenseNet
from torch import nn


class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """

        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val):
        model_path = self.dirpath / (model.__class__.__name__ + f"_epoch{epoch:03}.pt")
        save = (
            metric_val < self.best_metric_val
            if self.decreasing
            else metric_val > self.best_metric_val
        )

        if save:
            logging.info(
                f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B."
            )
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f"model-ckpt-epoch-{epoch}.pt", model_path, metric_val)
            self.top_model_paths.append({"path": model_path, "score": metric_val})
            self.top_model_paths = sorted(
                self.top_model_paths,
                key=lambda o: o["score"],
                reverse=not self.decreasing,
            )

        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(
            filename, type="model", metadata={"Validation score": metric_val}
        )
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n :]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            Path(o["path"]).unlink()
        self.top_model_paths = self.top_model_paths[: self.top_n]


def get_max_distance(dist_mat1: torch.Tensor, dist_mat2: torch.Tensor) -> float:
    return torch.max(torch.abs(dist_mat1 - dist_mat2)).item()


def run_epoch(model, loss_fn, dataloader, optimizer=None) -> None:

    mode = "valid" if optimizer is None else "train"

    running_loss = 0.0
    running_max_dist = 0.0

    for iteration, batch in enumerate(dataloader):

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
        loss = loss_fn(pred_dist_mat, gt_dist_mat)

        # Backward pass and optimization
        if mode == "train":
            loss.backward()
            optimizer.step()

        # Log
        max_distance = get_max_distance(pred_dist_mat, gt_dist_mat)

        if mode == "train":
            wandb.log(
                {f"{mode}/Loss": loss.item(), f"{mode}/MaxDistance": max_distance}
            )

        running_loss += loss.item()
        running_max_dist = (
            running_max_dist if running_max_dist > max_distance else max_distance
        )

    # Free up all memory
    torch.cuda.empty_cache()

    epoch_loss = running_loss / iteration

    if mode == "valid":
        wandb.log(
            {f"{mode}/Loss": epoch_loss, f"{mode}/MaxDistance": running_max_dist},
            commit=False,
        )

    return epoch_loss


def run_training():

    # Define the model and optimizer
    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9, verbose=True
    )

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the DataLoader
    train_loader, valid_loader = get_dataloaders(
        wandb.config.path_to_images,
        num_samples=wandb.config.number_of_crops,
        add_intensity_augmentation=wandb.config.augment,
    )

    # Define checkpoint saver
    model_log_path = Path(wandb.config.output_path) / f"{wandb.run.name}_{wandb.run.id}"
    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    # Start training
    for epoch in range(wandb.config.epochs):

        wandb.log({"LR": optimizer.param_groups[0]["lr"]}, commit=False)

        _ = run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        if epoch > wandb.config.epochs * 0.3:
            saver(model, epoch, valid_loss)

        scheduler.step()


if __name__ == "__main__":

    # Read config file specifed in command line arguments.
    init_wandb()

    # New training run
    run_training()

import torch
import wandb
from meddist.config import init_wandb
from meddist.data import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from monai.networks.nets import DenseNet
from torch import nn


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

    if mode == "valid":
        wandb.log(
            {f"{mode}/Loss": loss.item(), f"{mode}/MaxDistance": running_max_dist}
        )


def run_training():

    # Define the model and optimizer
    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    ).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the DataLoader
    train_loader, valid_loader = get_dataloaders(
        wandb.config.path_to_images, num_samples=wandb.config.number_of_crops, add_intensity_augmentation=wandb.config.augment
    )

    # Start training
    for _ in range(wandb.config.epochs):
        run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            run_epoch(model, loss_fn, valid_loader)


if __name__ == "__main__":

    # Read config file specifed in command line arguments.
    init_wandb()

    # New training run
    run_training()

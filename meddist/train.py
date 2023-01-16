import torch
import wandb
from meddist.config import init_wandb
from meddist.data import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from monai.networks.nets import DenseNet
from torch import nn


def run_epoch(model, loss_fn, dataloader, optimizer=None):

    mode = "valid" if optimizer is None else "train"

    running_loss = 0.0

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

        if iteration == 10:
            break

        # Log
        running_loss += loss.item()
        if mode == "train":
            wandb.log({f"Loss/{mode}": loss.item()})

    # Free up all memory
    torch.cuda.empty_cache()

    return running_loss / iteration


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
        wandb.config.path_to_images, num_samples=wandb.config.number_of_crops
    )

    # Start training
    for _ in range(wandb.config.epochs):
        _ = run_epoch(model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, loss_fn, valid_loader)

        wandb.log({"Loss/valid": valid_loss})


if __name__ == "__main__":

    # Read config file specifed in command line arguments.
    init_wandb()

    # New training run
    run_training()

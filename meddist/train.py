import torch
import wandb
from meddist.config import init_wandb
from meddist.data import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from monai.networks.nets import DenseNet
from torch import nn


def train_distance_model(model, optimizer, loss_fn, dataloader, num_epochs):

    with wandb.init(project="Meddist"):
        for epoch in range(num_epochs):
            for iteration, batch in enumerate(dataloader):

                # Prepare GT
                bboxes = get_cropped_bboxes(batch["image"], "RandSpatialCropSamples")
                centers = get_bbox_centers(bboxes)

                gt_dist_mat = torch.cdist(
                    torch.tensor(centers), torch.tensor(centers), p=2.0
                ).float()

                # Prepare forward pass
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
                loss.backward()
                optimizer.step()

                # Log
                wandb.log({"Loss": loss.item()})


if __name__ == "__main__":

    # Read config file specifed in command line arguments.
    init_wandb()

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

    # Train the model
    train_distance_model(
        model, optimizer, loss_fn, train_loader, num_epochs=wandb.config.epochs
    )

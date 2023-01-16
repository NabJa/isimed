import torch
import wandb
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

                with torch.autocast(device_type="cuda"):
                    # Forward pass
                    embeddings = model(image).cpu()

                pred_dist_mat = torch.cdist(embeddings.float(), embeddings.float(), p=2)
                loss = loss_fn(pred_dist_mat, gt_dist_mat)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Log
                wandb.log({"Loss": loss.item()})


# Define the model and optimizer
model = DenseNet(spatial_dims=3, in_channels=1, out_channels=512).to("cuda")
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
loss_fn = nn.MSELoss()

# Define the DataLoader
data_path = "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions-data2/registered"
train_loader, valid_loader = get_dataloaders(data_path)

# Train the model
train_distance_model(model, optimizer, loss_fn, train_loader, num_epochs=10)

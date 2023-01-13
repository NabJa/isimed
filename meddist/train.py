import torch
from meddist.data import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes, pairwise_comparisons
from meddist.model import ContrastiveDistanceDenseNet
from monai.networks.nets import DenseNet
from torch import nn


def train_distance_model(model, optimizer, loss_fn, dataloader, num_epochs):

    for epoch in range(num_epochs):
        for iteration, batch in enumerate(dataloader):

            bboxes = get_cropped_bboxes(batch["image"], "RandSpatialCropSamples")
            centers = get_bbox_centers(bboxes)

            gt_dist_mat = torch.cdist(
                torch.tensor(centers), torch.tensor(centers), p=2.0
            ).float()

            # Forward pass
            image = batch["image"].to("cuda")
            embeddings = model(image).cpu()

            pred_dist_mat = torch.cdist(embeddings, embeddings, p=2)

            loss = loss_fn(pred_dist_mat, gt_dist_mat)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            print(f"Epoch/Iteration {epoch:03}/{iteration:03} Loss, {loss.item():.3f}")


# Define the model and optimizer
model = DenseNet(spatial_dims=3, in_channels=1, out_channels=512).to("cuda")
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
loss_fn = nn.MSELoss()

# Define the DataLoader
data_path = "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions-data2"
train_loader, valid_loader = get_dataloaders(data_path)

# Train the model
train_distance_model(model, optimizer, loss_fn, train_loader, num_epochs=10)

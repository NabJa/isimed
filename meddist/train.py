import torch
from meddist.data import get_dataloaders
from meddist.dist import pairwise_comparisons
from meddist.model import ContrastiveDistanceDenseNet


def train_distance_model(model, optimizer, loss_fn, dataloader, num_epochs):

    for epoch in range(num_epochs):
        for batch in dataloader:
            
            image = batch["image"]

            comparisons = pairwise_comparisons(image.shape[0])

            for index1, index2 in comparisons:
                # Forward pass
                pred = model.forward_distance(image[index1], image[index2])
                loss = loss_fn(output1, output2, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            print("Loss, ", loss.item())


# Define the model and optimizer
model = ContrastiveDistanceDenseNet()
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define the DataLoader
data_path = "/sc-scratch/sc-scratch-gbm-radiomics/tcia/manifest-1654187277763/nifti/FDG-PET-CT-Lesions-data2"
dataloader = get_dataloaders(path=data_path)

# Train the model
train_distance_model(model, optimizer, loss_fn, dataloader, num_epochs=10)

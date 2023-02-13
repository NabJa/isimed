import torch
import wandb
from meddist import downstram_class
from meddist.training.barlow import forward_barlow, prepare_barlow
from meddist.training.contrastive import forward_simclr, prepare_simclr
from meddist.training.logs import CheckpointSaver, MetricTracker
from meddist.training.phys import forward_meddist, prepare_meddist
from monai.networks.nets import DenseNet

MODEL_PREP = {
    "meddist": (forward_meddist, prepare_meddist),
    "barlow": (forward_barlow, prepare_barlow),
    "simclr": (forward_simclr, prepare_simclr),
}


def run_epoch(forward, model, loss_fn, dataloader, optimizer=None) -> None:

    mode = "valid" if optimizer is None else "train"

    tracker = MetricTracker(f"{mode}/Loss")

    for batch in dataloader:

        # Prepare forward pass
        if mode == "train":
            optimizer.zero_grad()

        loss, metrics = forward(model, batch, loss_fn, mode)

        # Backward pass and optimization
        if mode == "train":
            loss.backward()
            optimizer.step()
            wandb.log({f"{mode}/loss": loss.item()})

        # Log
        tracker(loss.item())

    # Free up all memory
    torch.cuda.empty_cache()

    aggregated = tracker.aggregate()
    if mode == "valid":
        wandb.log(aggregated, commit=False)

    return aggregated[f"{mode}/Loss"]


def train(model_type, path_to_data_split, model_log_path):

    # Define the model and optimizer
    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    ).to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9, verbose=True
    )

    # Define the loss function and loaders
    forward, prep = MODEL_PREP[model_type]

    loss_fn, train_loader, valid_loader = prep(path_to_data_split)

    # Define checkpoint saver
    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    # Start training
    for epoch in range(wandb.config.epochs):

        wandb.log({"LR": optimizer.param_groups[0]["lr"]}, commit=False)

        _ = run_epoch(forward, model, loss_fn, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(forward, model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        # if epoch > wandb.config.epochs * 0.3:
        saver(model, epoch, valid_loss)

        scheduler.step()

        if (
            wandb.config.run_downsream_task
            and (epoch + 1) % wandb.config.downstream_every_n_epochs == 0
        ):
            downstram_class.train(path_to_data_split, model_log_path)

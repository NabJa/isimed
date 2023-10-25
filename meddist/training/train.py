import numpy as np
import torch
import wandb
from meddist import downstram_class
from meddist.metrics import estimate_rank_based_on_singular_values
from meddist.training.barlow import forward_barlow, prepare_barlow
from meddist.training.barlowdist import forward_barlow_dist, prepare_barlow_dist
from meddist.training.logs import CheckpointSaver, MetricTracker
from meddist.training.phys import forward_meddist, prepare_meddist
from meddist.training.simclr import forward_simclr, prepare_simclr
from meddist.training.simple_regression import SRegHead, forward_sreg, prepare_sreg
from meddist.training.two_headed_BTdist import (
    DistanceScaledBTHead,
    forward_hydra,
    prepare_hydra,
)
from monai.networks.nets import DenseNet

MODEL_PREP = {
    "meddist": (forward_meddist, prepare_meddist),
    "barlow": (forward_barlow, prepare_barlow),
    "simclr": (forward_simclr, prepare_simclr),
    "barlowdist": (forward_barlow_dist, prepare_barlow_dist),
    "hydra": (forward_hydra, prepare_hydra),
    "sreg": (forward_sreg, prepare_sreg),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(
    forward, model, loss_fn, dataloader, optimizer=None, global_step=0
) -> None:
    mode = "train"
    if optimizer is None:
        mode = "valid"
        embeddings = []
        model.eval()

    tracker = MetricTracker(f"{mode}/loss")
    for i, batch in enumerate(dataloader):
        # Prepare forward pass
        if mode == "train":
            optimizer.zero_grad()

        loss, metrics = forward(model, batch, loss_fn, mode, DEVICE)

        # Backward pass and optimization
        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm(
                model.parameters(), max_norm=2.0, error_if_nonfinite=False
            )
            optimizer.step()
            wandb.log({f"{mode}/loss": loss.item()}, step=global_step)
            global_step += 1

        if mode == "valid":
            emb: torch.Tensor = model(batch["image"].to(DEVICE))
            embeddings.append(emb.cpu().numpy())

        # Log
        tracker(loss.item())

    # Free up all memory
    torch.cuda.empty_cache()

    aggregated = tracker.aggregate()
    if mode == "valid":
        embeddings = np.concatenate(embeddings, axis=0)
        aggregated["valid/rankme"] = estimate_rank_based_on_singular_values(embeddings)
        wandb.log(aggregated, commit=False)

    return aggregated[f"{mode}/loss"], global_step


def train(path_to_data_split, model_log_path):
    # Define the model and optimizer
    model = DenseNet(
        spatial_dims=3, in_channels=1, out_channels=wandb.config.embedding_size
    )

    if wandb.config.model == "hydra":
        model = DistanceScaledBTHead(model, emb_size=wandb.config.embedding_size)
        wandb.watch(model, log="all")
    elif wandb.config.model == "sreg":
        model = SRegHead(model, wandb.config.embedding_size, wandb.config.crop_size)
        wandb.watch(model, log="all")

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9, verbose=True
    )

    # Define the loss function and loaders
    forward, prep = MODEL_PREP[wandb.config.model]

    loss_fn, train_loader, valid_loader = prep(path_to_data_split)

    # Define checkpoint saver
    saver = CheckpointSaver(model_log_path, decreasing=True, top_n=3)

    # Start training
    global_step = 0
    for epoch in range(wandb.config.epochs):
        wandb.log(
            {"meta/LR": optimizer.param_groups[0]["lr"], "meta/Epoch": epoch},
            commit=False,
        )

        _, global_step = run_epoch(
            forward,
            model,
            loss_fn,
            train_loader,
            optimizer,
            global_step,
        )

        with torch.no_grad():
            valid_loss, _ = run_epoch(forward, model, loss_fn, valid_loader)

        # Save checkpoints only 30% into the training. This prevents saving to early models.
        # if epoch > wandb.config.epochs * 0.3:
        saver(model, epoch, valid_loss)

        scheduler.step()

        if (
            wandb.config.run_downsream_task
            and (epoch + 1) % wandb.config.downstream_every_n_epochs == 0
        ):
            downstram_class.train(path_to_data_split, model_log_path)

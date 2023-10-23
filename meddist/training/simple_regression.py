import torch
import wandb
from meddist.data.loading import get_dataloaders
from meddist.dist import get_bbox_centers, get_cropped_bboxes
from torch import nn


class SRegHead(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, crop_size=None) -> None:
        super().__init__()

        self.min_center = crop_size // 2 if crop_size else 0

        self.backbone = backbone
        self.reg = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            # nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.backbone(x)

    def forward_coord(self, x):
        x = self.forward(x)
        return self.reg(x) + self.min_center


def forward_sreg(model: SRegHead, batch, loss_fn, mode, device):
    bboxes = get_cropped_bboxes(batch["image"], "RandSpatialCropSamples")
    centers = torch.tensor(get_bbox_centers(bboxes), dtype=torch.float32)


    with torch.autocast(device_type=device):
        pred_centers: torch.Tensor = model.forward_coord(batch["image"].to(device))

    pred_centers = pred_centers.to("cpu", dtype=torch.float32)
    loss: torch.Tensor = loss_fn(pred_centers, centers)

    with torch.no_grad():
        mean_diff = (centers - pred_centers).abs().mean().item()

    return loss, {f"{mode}/MeanDiff": mean_diff}



def prepare_sreg(path_to_data_split):

    loss_fn = nn.MSELoss()

    train_loader, valid_loader = get_dataloaders(
        path_to_data_split,
        num_samples=wandb.config.number_of_crops,
        crop_size=wandb.config.crop_size,
        add_intensity_augmentation=wandb.config.augment,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.num_workers,
    )

    return loss_fn, train_loader, valid_loader

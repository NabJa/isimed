from pathlib import Path

import torch
from monai.networks.layers import get_act_layer
from monai.networks.nets import DenseNet
from torch import nn


def get_latest_model(path_to_model_directory):
    directory = Path(path_to_model_directory).iterdir()
    # Model name has the format ARCHITECTURE_EPOCH001.pt
    sorted_models = sorted(directory, key=lambda x: int(x.name.split(".")[0][-3:]))
    return sorted_models[-1]


def load_densenet(path, out_channels=1024, map_location="cpu"):
    densenet = DenseNet(spatial_dims=3, in_channels=1, out_channels=out_channels)
    with open(path, mode="rb") as file:
        state_dict = torch.load(file, map_location=map_location)
    densenet.load_state_dict(state_dict)
    return densenet


def load_latest_densenet(path):
    return load_densenet(get_latest_model(path))

class ClassificationHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_size: int = 1024,
        hidden_size: int = 512,
        out_size: int = 1,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        retrain_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone.eval()
        self.retrain_backbone = bool(retrain_backbone)

        self.classfier = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            get_act_layer(activation),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.retrain_backbone):
            x = self.backbone(x)

        return self.classfier(x)


class LinearHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_size=1024,
        out_classes=1,
        retrain_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.retrain_backbone = retrain_backbone
        self.linear = nn.Linear(embedding_size, out_classes)

    def forward(self, x):
        with torch.set_grad_enabled(self.retrain_backbone):
            x = self.backbone(x)
        return self.linear(x)

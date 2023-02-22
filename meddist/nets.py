from pathlib import Path
from typing import OrderedDict

import torch
from monai.networks.layers import get_act_layer
from monai.networks.nets import DenseNet
from torch import nn


def get_latest_model(path_to_model_directory):
    directory = Path(path_to_model_directory).glob("*.pt")
    # Model name has the format ARCHITECTURE_EPOCH001.pt
    sorted_models = sorted(directory, key=lambda x: int(x.name.split(".")[0][-3:]))
    model_path: Path = sorted_models[-1]
    model_name = model_path.name.split("_")[0]
    return model_name, model_path


def load_state_file(path, map_location="cpu") -> OrderedDict[str, torch.Tensor]:
    with open(path, mode="rb") as file:
        return torch.load(file, map_location=map_location)


def load_bthead_backbone_state(path, map_location="cpu"):
    state_dict = load_state_file(path, map_location)
    return {
        k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone" in k
    }


def load_latest_densenet(path, out_channels=1024):
    name, path = get_latest_model(path)

    if name.lower() == "densenet":
        state_dict = load_state_file(path)
    elif name.lower() == "distancescaledbthead":
        state_dict = load_bthead_backbone_state(path)
    else:
        raise NotImplementedError(f"Model {name} unknown.")

    densenet = DenseNet(spatial_dims=3, in_channels=1, out_channels=out_channels)
    densenet.load_state_dict(state_dict)

    return densenet


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

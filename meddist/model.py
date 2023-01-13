from typing import Tuple

import torch
from monai.networks.nets import DenseNet
from torch import nn


def get_distance(embedding1, embedding2, p_norm=2):
    return torch.dist(embedding1, embedding2, p=p_norm)


class ContrastiveDistanceDenseNet(nn.Module):
    def __init__(self, embedding_size: int = 512, p_norm=2, **kwargs):
        super(ContrastiveDistanceDenseNet, self).__init__()
        self.p_norm = p_norm
        self.encoder = DenseNet(
            spatial_dims=3, in_channels=1, out_channels=embedding_size, **kwargs
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two crops as input without batch dimension."""
        embedding1 = self.encoder(input1.unsqueeze(0))
        embedding2 = self.encoder(input2.unsqueeze(0))

        return embedding1, embedding2

    def forward_distance(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        embedding1, embedding2 = self.forward(input1, input2)
        return get_distance(embedding1, embedding2)

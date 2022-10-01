import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.encoders._encoder import Encoder
from src.testingnetworks.model.layers.basic_layers import Dense


class DenseNetwork(Encoder):
    """
    The simple DenseNet module. This module is basically a convolution between features and weights, without taking into account the adjacency matrix.
    """
    INPUTS = [DATA.NODE_FEATURES]

    def __init__(self, layers_dimensions: list, act=torch.nn.ReLU(), bias: bool = False, device: str = 'cpu'):
        super(DenseNetwork, self).__init__(input_list=DenseNetwork.INPUTS)

        self.layers = torch.nn.ModuleList()

        for i in range(1, len(layers_dimensions)):
            gcn_i = Dense(input_dim=layers_dimensions[i - 1], output_dim=layers_dimensions[i], act=act, bias=bias)
            self.layers.append(gcn_i.to(device))

    def forward(self, sample: dict) -> torch.Tensor:
        node_embs = sample[DATA.NODE_FEATURES]

        for layer in self.layers:
            node_embs = layer.forward(node_embs)

        return node_embs

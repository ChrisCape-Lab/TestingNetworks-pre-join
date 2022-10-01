import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.encoders._encoder import Encoder
from src.testingnetworks.model.layers.conv_layers import GraphConvolution


class GCN(Encoder):
    """The GCN module. GCN is a simple sequence of GraphConvolution layers, one after another"""
    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def __init__(self, layers_dimensions: list, act=torch.nn.ReLU(), bias: bool = False, device: str = 'cpu'):
        super(GCN, self).__init__(input_list=GCN.INPUTS)

        self.layers = torch.nn.ModuleList()
        self.device = device

        for i in range(1, len(layers_dimensions)):
            gcn_i = GraphConvolution(input_dim=layers_dimensions[i - 1], output_dim=layers_dimensions[i], act=act, bias=bias)
            self.layers.append(gcn_i.to(self.device))

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix = sample[DATA.ADJACENCY_MATRIX]
        node_embs = sample[DATA.NODE_FEATURES]
        for layer in self.layers:
            node_embs = layer(adj_matrix, node_embs)

        return node_embs

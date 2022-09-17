import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.encoders._encoder import Encoder
from src.testingnetworks.model.layers.complex_layers import GRCUvH, GRCUvO


class EvolveGCNvH(Encoder):

    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES, DATA.NODE_MASK]

    def __init__(self, layers_dims, act=torch.nn.RReLU, skipfeats=False, device='cpu'):
        super(EvolveGCNvH, self).__init__(input_list=EvolveGCNvH.INPUTS)

        self.act = act
        self.skipfeats = skipfeats
        self.device = device

        self.GRCU_layers = torch.nn.ModuleList()

        num_layers = len(layers_dims)
        for i in range(1, num_layers):
            grcu_i = GRCUvH(layers_dims[i - 1], layers_dims[i], act=act if i != num_layers else lambda x: x)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix_list = sample[DATA.ADJACENCY_MATRIX]
        node_embs_list = sample[DATA.NODE_FEATURES]
        nodes_mask_list = sample[DATA.NODE_MASK]

        node_feats = node_embs_list[-1]

        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_layers:
            node_embs_list = unit.forward(adj_matrix_list, node_embs_list, nodes_mask_list)

        # The real output is just the last one since the sample is an history
        out = node_embs_list[-1]

        # If skipfeats, the features are skip-connected to the output with a torch.cat
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)

        return out


class EvolveGCNvO(Encoder):

    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def __init__(self, layers_dims, act=torch.nn.RReLU, skipfeats=False, device='cpu'):
        super(EvolveGCNvO, self).__init__(input_list=EvolveGCNvO.INPUTS)

        self.act = act
        self.skipfeats = skipfeats
        self.device = device

        self.GRCU_layers = torch.nn.ModuleList()

        num_layers = len(layers_dims)
        for i in range(1, len(layers_dims)):
            grcu_i = GRCUvO(layers_dims[i - 1], layers_dims[i], act=act if i != num_layers else lambda x: x)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix_list = sample[DATA.ADJACENCY_MATRIX]
        node_embs_list = sample[DATA.NODE_FEATURES]

        nodes_embs = node_embs_list
        node_feats = node_embs_list[-1]

        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_layers:
            nodes_embs = unit.forward(adj_matrix_list, nodes_embs)

        # The real output is just the last one since the sample is an history
        out = nodes_embs[-1]

        # If skipfeats, the features are skip-connected to the output with a torch.cat
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)

        return out

import torch

from src.model.networks.network import Network
from src.testingnetworks.model.layers.complex_layers import GRCUvH
from src.testingnetworks.model.layers.recurrent_layers import GRUMirrored
from src.testingnetworks.model.layers.complex_layers import HCA


class Prj1(Network):
    def __init__(self, layers_dims, device='cpu', skipfeats='False'):
        super().__init__()

        self.device = device
        self.skipfeats = skipfeats

        self.submodules = torch.nn.ModuleList()
        self.GRCU_layers = []

        for i in range(1, len(layers_dims)):
            input_dim = layers_dims[i-1]
            if (skipfeats == 'Total') & i != 1:
                input_dim += layers_dims[0]
            grcu_i = GRCUvH(input_dim, layers_dims[i])
            self.GRCU_layers.append(grcu_i.to(self.device))
            self.submodules.append(grcu_i)

    def forward(self, inputs):
        adj_matrix_list = inputs[0]
        node_embs_list = inputs[1]
        nodes_mask_list = inputs[2]

        nodes_embs = node_embs_list
        node_feats = node_embs_list[-1]

        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_layers:
            nodes_embs = unit.forward(adj_matrix_list, nodes_embs, nodes_mask_list)

        # The real output is just the last one since the sample is an history
        out = nodes_embs[-1]

        # If skipfeats, the features are skip-connected to the output with a torch.cat
        if self.skipfeats == 'Partial':
            out = torch.cat((out, node_feats), dim=1)

        return out


class Prj2(Network):
    def __init__(self, layers_dims, init_hiddens, nb_window, act=torch.nn.ReLU(), dropout=0., bias_rows=None, device='cpu', skipfeats=False):
        super().__init__()

        self.nb_window = nb_window
        self.act = act
        self.device = device
        self.skipfeats = skipfeats

        self.hiddens = []
        for i in range(0, self.nb_window):
            self.hiddens.append(init_hiddens)

        self.submodules = torch.nn.ModuleList()
        self.GRCU_layers = []

        for i in range(1, len(layers_dims)):
            grcu_i = GRCUvH(layers_dims[i - 1], layers_dims[i])
            self.GRCU_layers.append(grcu_i.to(self.device))
            self.submodules.append(grcu_i)

        self.hca = HCA(input_dim=layers_dims[-1], nb_window=self.nb_window)
        self.submodules.append(self.hca)

        self.gru = GRUMirrored(input_dim=layers_dims[-1], output_dim=layers_dims[-1], act=torch.nn.Sigmoid())
        self.submodules.append(self.gru)

        self.submodules = torch.nn.ModuleList()

    def forward(self, inputs):
        adj_matrix_list = inputs[0]
        node_embs_list = inputs[1]
        nodes_mask_list = inputs[2]

        nodes_embs = node_embs_list

        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_layers:
            nodes_embs = unit.forward(adj_matrix_list, nodes_embs, nodes_mask_list)

        # The real output is just the last one since the sample is an history
        current = nodes_embs[-1]
        short = self.hca.forward(self.hiddens)

        # Mix the short-term information extracted with the current status to obtain the node embeddings
        # TODO: probabilmente qui un LSTM è meglio per ricordare lunghe sequenze
        hidden_t = self.act(self.gru.forward(current, short))

        # Add the current node embeddings to the history
        self.hiddens.pop(0)
        self.hiddens.append(hidden_t.detach())

        return hidden_t

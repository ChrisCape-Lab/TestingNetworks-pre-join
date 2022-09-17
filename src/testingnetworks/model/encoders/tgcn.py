import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.encoders._encoder import Encoder
from src.testingnetworks.model.layers.conv_layers import GraphConvolution
from src.testingnetworks.model.layers.recurrent_layers import LSTMMirrored, GLSTM
from src.testingnetworks.utils import init_glorot


class TGCN(Encoder):

    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def __init__(self, layers_dims, num_nodes, act=torch.nn.ReLU(), bias=False, device='cpu'):
        super(TGCN, self).__init__(input_list=TGCN.INPUTS)

        self.device = device

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(1, len(layers_dims)):
            gcn_i = GraphConvolution(input_dim=layers_dims[i - 1], output_dim=layers_dims[i], act=act, bias=bias)
            self.gcn_layers.append(gcn_i.to(self.device))
        self.rnn = LSTMMirrored(input_dim=layers_dims[-1], output_dim=layers_dims[-1])
        self.history = init_glorot((num_nodes, layers_dims[-1]))
        self.carousel = 0

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix = sample[DATA.ADJACENCY_MATRIX]
        node_embs = sample[DATA.NODE_FEATURES]

        if self.history is None:
            self.history = torch.zeros((node_embs.shape[0], self.history_dim))

        for gcn in self.gcn_layers:
            node_embs = gcn.forward(adj_matrix, node_embs)

        node_embs, carousel = self.rnn.forward(inputs=node_embs, hist=self.history, carousel=self.carousel)
        self.history = node_embs.detach()
        self.carousel = carousel.detach()

        return node_embs


class TGCNe(Encoder):

    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def __init__(self, layers_dims, num_nodes, act=torch.nn.ReLU(), bias=False, device='cpu'):
        super(TGCNe, self).__init__(input_list=TGCNe.INPUTS)

        self.device = device

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(1, len(layers_dims)):
            gcn_i = GraphConvolution(input_dim=layers_dims[i - 1], output_dim=layers_dims[i], act=act, bias=bias)
            self.gcn_layers.append(gcn_i.to(self.device))
        self.rnn = GLSTM(num_nodes, num_feats=layers_dims[-1], node_w=False, feats_w=True)
        self.history = init_glorot((num_nodes, layers_dims[-1]))
        self.carousel = 0

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix = sample[DATA.ADJACENCY_MATRIX]
        node_embs = sample[DATA.NODE_FEATURES]

        for gcn in self.gcn_layers:
            node_embs = gcn.forward(adj_matrix, node_embs)

        node_embs, carousel = self.rnn.forward(inputs=node_embs, hist=self.history, carousel=self.carousel)
        self.history = node_embs.detach()
        self.carousel = carousel.detach()

        return node_embs


class TGCNseq(Encoder):
    """
    The sequence of A_hat is passed through a GCN network, producing a list of results. All the results are stacked and
    passed through the LSTM network
    """
    INPUTS = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def __init__(self,  layers_dims, act=torch.nn.ReLU(), bias=False, device='cpu'):
        super(TGCNseq, self).__init__(input_list=TGCNseq.INPUTS)

        self.activation = act
        self.device = device
        self.layers = torch.nn.ModuleList()

        for i in range(1, len(layers_dims)):
            gcn_i = GraphConvolution(input_dim=layers_dims[i - 1], output_dim=layers_dims[i], act=act, bias=bias)
            self.layers.append(gcn_i.to(self.device))

        self.rnn = torch.nn.LSTM(input_size=layers_dims[-1], hidden_size=layers_dims[-1], num_layers=1)

    def forward(self, sample: dict) -> torch.Tensor:
        adj_matrix_list = sample[DATA.ADJACENCY_MATRIX]
        node_embs_list = sample[DATA.NODE_FEATURES]

        last_l_seq = []
        for t, adj in enumerate(adj_matrix_list):
            node_embs = node_embs_list[t]
            for layer in self.layers:
                node_embs = layer.forward(adj, node_embs)
            last_l_seq.append(node_embs)

        last_l_seq = torch.stack(last_l_seq)
        out, _ = self.rnn(last_l_seq, None)

        return out[-1]

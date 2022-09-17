import torch

from src.testingnetworks.model.layers.basic_layers import Dense
from src.testingnetworks.model.layers.conv_layers import GraphConvolution
from src.testingnetworks.utils import matmul


class AttributeDecoder(torch.nn.Module):
    """The base attribute decoder: takes as input the matrix and the node embeddings and try to reconstruct the original feature matrix"""
    def __init__(self, layers_dims, act=torch.nn.ReLU(), dropout=0.0, bias=False, device='cpu'):
        super().__init__()

        self.device = device

        self.gcn1 = GraphConvolution(input_dim=layers_dims[0], output_dim=layers_dims[1], act=act, bias=bias)
        self.gcn2 = GraphConvolution(input_dim=layers_dims[1], output_dim=layers_dims[2], act=act, bias=bias)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, adj_matrix, node_embs):
        node_embs = self.gcn1.forward(adj_matrix, node_embs)
        node_embs = self.dropout(node_embs)
        node_embs = self.gcn2.forward(adj_matrix, node_embs)

        return node_embs


class StructureDecoder(torch.nn.Module):
    """The base structure decoder: takes as input the matrix and the node embeddings and try to reconstruct the original adjacency matrix"""
    def __init__(self, layers_dims, act=torch.nn.ReLU(), dropout=0.0, bias=False, device='cpu'):
        super().__init__()

        self.device = device

        self.gcn1 = GraphConvolution(input_dim=layers_dims[0], output_dim=layers_dims[1], act=act, bias=bias)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, adj_matrix, node_embs):
        node_embs = self.gcn1.forward(adj_matrix, node_embs)
        node_embs = self.dropout(node_embs)

        # Matmul the node embeddings to obtain a adjacency matrix: NxF * (NxF).T -> NxN
        out = matmul(node_embs, node_embs.t())

        return out


class DenseDecoder(torch.nn.Module):
    """The base structure decoder: takes as input the matrix and the node embeddings and try to reconstruct the original adjacency matrix"""
    def __init__(self, layers_dims, act=torch.nn.ReLU(), dropout=0.0, bias=False, device='cpu'):
        super(DenseDecoder, self).__init__()

        self.device = device

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layers_dims)):
            self.layers.append(Dense(layers_dims[i-1], layers_dims[i], act, bias))

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)

        return out

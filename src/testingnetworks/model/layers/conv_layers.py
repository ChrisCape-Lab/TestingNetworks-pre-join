import torch
from torch.nn.parameter import Parameter

from src.testingnetworks.utils import init_glorot, init_zeros, matmul


class GraphConvolution(torch.nn.Module):
    """
    The basic graph convolution layer.
    NOTE: this layer DOES NOT the matrix degree normalization. This is done since it can be calculated just at the beginning, while inserting it into
    this class repeat the computation at each step.
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.LeakyReLU(), bias=False):
        super(GraphConvolution, self).__init__()

        self.act = act
        self.has_bias = bias

        # Weights and bias (the variables) are created as shared variables for an easy use and manipulation
        self.weights = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)
        if self.has_bias:
            self.bias = Parameter(init_zeros([output_dim]), requires_grad=True)

    def forward(self, adj_matrix, node_embs):
        """
        In this layer, the call first perform the dropout, then do the convolution between input and weights, applying
        the bias at the end. It's just the standard convolution layer
        :param node_embs: the input on which the layer has to operate
        :param adj_matrix: tensor, the adjacency matrix
        :return: the convolution between the inputs and the layer weights (and bias)
        """
        # Convolution step, output = a * (x * w)
        output = matmul(adj_matrix, matmul(node_embs, self.weights), sparse=adj_matrix.is_sparse)

        # Bias step
        if self.has_bias:
            output += self.bias

        return self.act(output)


class SampledGraphConvolution(torch.nn.Module):
    def __init__(self, input_dim, output_dim, act=torch.nn.LeakyReLU(), rank=100, bias=False):
        super(SampledGraphConvolution, self).__init__()

        self.act = act
        self.rank = rank

        # Weights and bias (the variables) are created as shared variables for an easy use and manipulation
        self.weights = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)
        if bias:
            self.bias = Parameter(init_zeros([output_dim]), requires_grad=True)

    def forward(self, adj_matrix, node_embs):
        x = node_embs
        norm_x = torch.nn.functional.normalize(x, p=2.0, dim=1)
        norm_support = torch.nn.functional.normalize(adj_matrix, p=2.0, dim=0)
        norm_mix = torch.cross(norm_x, norm_support)
        norm_mix = norm_mix * torch.inverse(torch.sum(norm_mix))
        sampledIndex = torch.multinomial(torch.log(norm_mix), self.rank)
        new_support = matmul(adj_matrix, torch.diag(norm_mix), sparse=adj_matrix.is_sparse)

        # Convolution step, output = a * (x * w)
        output = matmul(new_support, matmul(node_embs, self.weights), sparse=new_support.is_sparse)

        # Bias step
        if self.has_bias:
            output += self.bias

        return self.act(output)


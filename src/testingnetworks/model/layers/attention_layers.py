import torch
import torch.nn.functional
from torch.nn.parameter import Parameter

from src.testingnetworks.utils import init_glorot


class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(init_glorot([in_features, out_features]), requires_grad=True)
        self.a = Parameter(init_glorot([2*out_features, 1]), requires_grad=True)

        self.act = torch.nn.LeakyReLU(self.alpha)

    def forward(self, adj, h):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = self.act(Wh1 + Wh2.T)

        #zero_vec = -9e15*torch.ones_like(e, dtype=torch.float32)
        attention = torch.where(adj.to_dense() > 0, e, torch.tensor(-9e15, dtype=torch.float32))
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return torch.nn.functional.elu(h_prime)
        else:
            return h_prime


class GraphMultiAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GraphMultiAttentionLayer, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, adj, x):
        x = torch.nn.functional.dropout(x, self.dropout)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)

        return x

import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.layers.attention_layers import GraphAttentionLayer
from src.testingnetworks.model.layers.basic_layers import Dense
from src.testingnetworks.model.layers.conv_layers import GraphConvolution
from src.testingnetworks.model.layers.complex_layers import GRCUvH, GRCUvO, HCA, CommunityGCNLayer
from src.testingnetworks.model.layers.recurrent_layers import GRUMirrored, LSTMMirrored, GLSTM
from src.testingnetworks.model.other.samplers import FastGCNSampler
from src.testingnetworks.utils import init_glorot

from src.testingnetworks.commons.dataloader.sample import Sample


class Test(torch.nn.Module):
    def __init__(self, layers_dims, act=torch.nn.ReLU(), bias=False, device='cpu'):
        super().__init__()

        self.device = device

        self.dense = Dense(input_dim=layers_dims[0], output_dim=layers_dims[1], act=act, bias=bias)
        self.layers = torch.nn.ModuleList()

        for i in range(2, len(layers_dims)):
            gcn_i = GraphConvolution(input_dim=layers_dims[i - 1], output_dim=layers_dims[i], act=act, bias=bias)
            self.layers.append(gcn_i.to(self.device))

    def forward(self, sample):
        adj_matrix = sample.sp_adj_list
        node_embs = sample.nodes_features_list

        node_embs = self.dense.forward(node_embs)
        for layer in self.layers:
            node_embs = layer.forward(adj_matrix, node_embs)

        return node_embs


class EvolveGCNvHDouble(torch.nn.Module):
    def __init__(self, layers_dims, short_lenght, skipfeats=False, device='cpu'):
        super().__init__()

        self.short_lenght = short_lenght
        self.act = torch.nn.LeakyReLU()
        self.skipfeats = skipfeats
        self.device = device

        num_layers = len(layers_dims)

        self.GRCU_short_layers = torch.nn.ModuleList()
        for i in range(1, num_layers):
            grcu_i = GRCUvH(layers_dims[i - 1], layers_dims[i], act=torch.nn.LeakyReLU() if i != num_layers else lambda x: x)
            self.GRCU_short_layers.append(grcu_i.to(self.device))

        self.GRCU_long_layers = torch.nn.ModuleList()
        for i in range(1, num_layers):
            grcu_i = GRCUvH(layers_dims[i - 1], layers_dims[i], act=torch.nn.LeakyReLU() if i != num_layers else lambda x: x)
            self.GRCU_long_layers.append(grcu_i.to(self.device))

    def forward(self, sample):
        adj_matrix_list = sample.sp_adj_list
        node_embs_list = sample.nodes_features_list
        nodes_mask_list = sample.node_mask_list

        nodes_embs = node_embs_list
        node_feats = node_embs_list[-1]

        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_long_layers:
            nodes_embs = unit.forward(adj_matrix_list, nodes_embs, nodes_mask_list)

        adj_matrix_list_short = sample.sp_adj_list[-self.short_lenght:]
        node_embs_list_short = sample.nodes_features_list[-self.short_lenght:]
        nodes_mask_list_short = sample.node_mask_list[-self.short_lenght:]
        # For each layer, the sample is passed and processed in a sequential way
        for unit in self.GRCU_short_layers:
            node_embs_list_short = unit.forward(adj_matrix_list_short, node_embs_list_short, nodes_mask_list_short)

        # The real output is just the last one since the sample is an history
        out = nodes_embs[-1] + node_embs_list_short[-1]

        # If skipfeats, the features are skip-connected to the output with a torch.cat
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)

        return out


class AddGraph(torch.nn.Module):
    """GCN is a simple sequence of GraphConvolution layers, one after another"""

    def __init__(self, layers_dims, nb_window, num_nodes, act=torch.nn.ReLU(), bias=False, device='cpu'):
        super().__init__()

        self.nb_window = nb_window
        self.act = act

        self.device = device
        self.hiddens = []
        for i in range(0, self.nb_window):
            self.hiddens.append(torch.rand((num_nodes, layers_dims[0])))
        self.prev = self.hiddens[-1].detach()

        self.gcn = GCN(layers_dims=layers_dims, act=act, bias=bias, device=device)
        self.hca = HCA(input_dim=layers_dims[0], nb_window=self.nb_window)
        self.gru = GRUMirrored(input_dim=layers_dims[-1], output_dim=layers_dims[0], act=torch.nn.Sigmoid())

    def forward(self, sample):
        gcn_sample = Sample(index=sample.index, sp_adj_list=sample.sp_adj_list, sp_conn_matrix_list=[], sp_mod_matrix_list=[],
                            nodes_features_list=self.prev, label_list=[], node_mask_list=[])

        # apply the GCN to the current sample to extract the current node embeddings
        current = self.gcn.forward(gcn_sample)

        # Extract short-term information by comparing the history
        short = self.hca.forward(self.hiddens)

        # Mix the short-term information extracted with the current status to obtain the node embeddings
        hidden_t = self.act(self.gru.forward(current, short))

        # Add the current node embeddings to the history
        self.hiddens.pop(0)
        self.hiddens.append(hidden_t.detach())

        self.prev = hidden_t.detach()

        return hidden_t


class GAT(torch.nn.Module):
    def __init__(self, layers_dims, dropout, nheads, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(layers_dims[0], layers_dims[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(0, nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, sample):
        adj = sample.sp_adj_list
        x = sample.nodes_features_list
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)

        return x


class ComGAEncoder(torch.nn.Module):
    def __init__(self, layers_dims, num_nodes, act=torch.nn.LeakyReLU(), bias=False, device='cpu'):
        super(ComGAEncoder, self).__init__()

        self.device = device

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layers_dims)):
            num_nodes = num_nodes if i == 1 else None
            layer = CommunityGCNLayer(input_dim=layers_dims[i - 1], output_dim=layers_dims[i], num_nodes=num_nodes, act=act, bias=bias)
            self.layers.append(layer.to(self.device))
        self.gcn = GraphConvolution(input_dim=layers_dims[-1], output_dim=layers_dims[-1], act=act, bias=bias)

    def forward(self, sample):
        community_embs = sample.sp_mod_matrix_list
        adj_matrix = sample.sp_adj_list
        node_embs = sample.nodes_features_list

        for layer in self.layers:
            node_embs, community_embs = layer.forward(community_embs, adj_matrix, node_embs)
        node_embs = self.gcn.forward(adj_matrix, node_embs)

        return node_embs, community_embs

import torch
from torch.nn.parameter import Parameter

from src.testingnetworks.model.layers.basic_layers import Dense
from src.testingnetworks.model.layers.conv_layers import GraphConvolution
from src.testingnetworks.model.layers.recurrent_layers import GRUTopK, LSTMTopK
from src.testingnetworks.utils import init_glorot, matmul


class GRCUvH(torch.nn.Module):
    """
    The GRCUvH is the basic Evolve-h layer. This layer takes as input the features list and the A_hat list. At each step,
    it regulates the weights of the GCN layer using the GRUTopK and then apply the normal GCN layer A_hat * H * W
    """
    def __init__(self, input_dim: int, output_dim: int, act=torch.nn.RReLU(), skipfeats: bool = False):
        super().__init__()

        self.gru_topk = GRUTopK(input_dim=input_dim, output_dim=output_dim)
        self.skipfeats = skipfeats

        self.GCN_init_weights = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)

        self.act = act

    def forward(self, adj_matrix_list: list, node_embs_list: list, mask_list: list) -> list:
        """
        The GRCU-H layer of Evolve, use the node embeddings and the mask to evolve the weights of the GCN, then just
        perform the normal convolution.
        CAREFUL1: this layer do an ENTIRE PASS, for all elements in list
        CAREFUL2: this module implements a built-in GCN since the GCN weights MUST NOT BE parameters
        :param adj_matrix_list: list(tensor), the list of adjacency matrix
        :param node_embs_list: list(tensor), the list of nodes features
        :param mask_list:
        :return: tensor, the output of all the time instants
        """
        gcn_weights = self.GCN_init_weights
        out_seq = []
        for t, adj in enumerate(adj_matrix_list):
            node_embs = node_embs_list[t]
            # First, evolve the weights from the initial and use the new weights with the node_embs...
            gcn_weights = self.gru_topk(node_embs, gcn_weights, mask_list[t])
            # ...then convolve
            out = self.act(matmul(adj, matmul(node_embs, gcn_weights), sparse=adj.is_sparse))

            out_seq.append(out)

        return out_seq


class GRCUvO(torch.nn.Module):
    """
    The GRCUvO is the basic Evolve-o layer. This layer takes as input the features list and the A_hat list. At each step,
    it regulates the weights of the GCN layer using the LSTMTopK and then apply the normal GCN layer A_hat * H * W
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.LeakyReLU()):
        super().__init__()

        self.act = act

        self.lstm_topk = LSTMTopK(input_dim, output_dim)

        self.GCN_init_weights = Parameter(init_glorot([input_dim, output_dim]), requires_grad=False)

    def forward(self, adj_matrix_list: list, node_embs_list: list) -> list:
        """
        The GRCU-H layer of Evolve, use the node embeddings and the mask to evolve the weights of the GCN, then just
        perform the normal convolution.
        CAREFUL1: this layer do an ENTIRE PASS, for all elements in list
        CAREFUL2: this module implements a built-in GCN since the GCN weights MUST NOT BE parameters
        :param adj_matrix_list: list(tensor), the list of adjacency matrix
        :param node_embs_list: list(tensor), the list of nodes features
        :return: tensor, the output of all the time instants
        """
        gcn_weights = self.GCN_init_weights
        out_seq = []
        for t, adj in enumerate(adj_matrix_list):
            node_embs = node_embs_list[t]
            # First, evolve the weights from the initial and use the new weights...
            gcn_weights = self.lstm_topk(gcn_weights)
            # ...then convolve
            node_embs = self.act(matmul(adj, matmul(node_embs, gcn_weights), sparse=adj.is_sparse))

            out_seq.append(node_embs)

        return out_seq


class HCA(torch.nn.Module):
    def __init__(self, input_dim, nb_window, act=torch.nn.LeakyReLU()):
        super(HCA, self).__init__()

        self.input_dim = input_dim
        self.nb_window = nb_window
        self.act = act

        self.weight_r = Parameter(init_glorot([self.input_dim, 1]), requires_grad=True)
        self.weight_q = Parameter(init_glorot([self.input_dim, self.input_dim]), requires_grad=True)

    def forward(self, inputs):
        # The input is a list [H_(t-w),...,H_(t-1)] of W elements of matrices [NxD]
        cx = inputs                                                                        # [NxD]xW

        # Multiply each input for both the weights, first to convolve then to reduce the dimensionality (D -> 1)
        ce = [matmul(self.act(matmul(x, self.weight_q)), self.weight_r) for x in cx]       # [Nx1]xW <- [([NxD] * [DxD]) * [Dx1]]xW

        # Squeeze the previous list of W elements into a single [NxW] element
        ce_concat = torch.cat(ce, dim=-1)                                                   # [Nxw]

        ca = torch.nn.functional.softmax(ce_concat, dim=-1)                                 # [Nxw]
        # After the softmax re-split the [NxW] matrix into W [Nx1] matrices
        ca_split = torch.split(ca, 1, dim=1)                                                # [Nx1]xW

        # Apply the attention: each element is compared with all previous elements in history
        # TODO: add explicit broadcasting?
        cax_split = [torch.mul(cx[ii], ca_split[ii]) for ii in range(self.nb_window)]       # [NxD]xw
        cax = cax_split[0]
        # Sum every element
        for i in range(1, len(cax_split)):
            cax = torch.add(cax, cax_split[i])                                              # [NxD]

        return cax


class CommunityGCNLayer(torch.nn.Module):
    """
        The GRCUvH is the basic Evolve-h layer. This layer takes as input the features list and the A_hat list. At each step,
        it regulates the weights of the GCN layer using the GRUTopK and then apply the normal GCN layer A_hat * H * W
        """

    def __init__(self, input_dim, output_dim, num_nodes=None, act=torch.nn.LeakyReLU(), bias=False):
        super(CommunityGCNLayer, self).__init__()

        if num_nodes is not None:
            self.communityDense = Dense(input_dim=num_nodes, output_dim=output_dim, act=act, bias=bias)
        else:
            self.communityDense = Dense(input_dim=input_dim, output_dim=output_dim, act=act, bias=bias)
        self.gcn = GraphConvolution(input_dim=input_dim, output_dim=output_dim, act=act, bias=bias)

        self.act = act

    def forward(self, mod_matrix, adj_matrix, node_embs):
        community_embs = self.communityDense.forward(mod_matrix)
        node_embs = self.gcn.forward(adj_matrix, node_embs)

        return node_embs + community_embs, community_embs

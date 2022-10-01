import torch
import torch.sparse
from torch.nn.parameter import Parameter

from src.testingnetworks.utils import init_glorot, init_zeros, matmul, pad_with_last_val, broadcast


class Dense(torch.nn.Module):
    """
    Implementation of the basic Dense layer
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.LeakyReLU(), bias=False):
        super().__init__()

        self.act = act
        self.bias = None

        self.weights = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)
        if bias:
            self.bias = Parameter(init_zeros([output_dim]), requires_grad=True)

    def forward(self, inputs):
        """
        As a normal dense layer this takes the input and multiply them for the layer weights, sum the bias if present,
        and at the end return the activation of the previous result
        :param inputs: a tensor, the usual inputs of a dense layer
        :return: ReLU(W * X) or ReLU((W * X) + b)
        """
        # Convolve
        output = matmul(inputs, self.weights, sparse=inputs.is_sparse)

        # Bias
        if self.bias is not None:
            output += self.bias

        return self.act(output)


class TopK(torch.nn.Module):
    """
    The TopK layer is a summarization layer. Basically this layer takes as input a matrix NxM and a factor k and produce a matrix Mxk. To this, the
    layer extract the best k rows from the N initials, weight them according to some weights and return them transposed.
    This layer is used in GCRU to take node embs in the form RxI and produce a RxO (k = O) to fit them in the weights update.
    """
    def __init__(self, features_num: int, k: int):
        super().__init__()

        self.scorer = Parameter(init_glorot([features_num, 1]), requires_grad=True)
        self.k = k

    def forward(self, node_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        :param node_embs: Tensor (NxF), is the tensor containing the embedding of the elements
        :param mask: Tensor (Nx1), is the mask of the active nodes
        :return: Tensor (FxK)
        """
        # Generate a score for nodes depending on some parameters
        scores = matmul(node_embs, self.scorer, node_embs.is_sparse) / self.scorer.norm()
        scores = torch.add(scores, mask.view(-1, 1))  # view(-1,1) simply add the other dimension to match the scores

        # Extract the top k nodes and their indices
        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if node_embs.is_sparse:
            node_embs = node_embs.to_dense()

        # Multiply the topk node features for the tanh
        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


class Gate(torch.nn.Module):
    """
    A versatile definition of a GRU gate. It includes the normal formula and the activation and can constitute different
    types of gates depending on parameters.
    """
    def __init__(self, input_dim: int, output_dim: int, act):
        super().__init__()
        self.act = act

        self.W = Parameter(init_glorot([input_dim, input_dim]), requires_grad=True)
        self.U = Parameter(init_glorot([input_dim, input_dim]), requires_grad=True)
        self.bias = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)

    def forward(self, node_embs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # out = W * x + U * h + b
        return self.act(self.W.matmul(node_embs) + self.U.matmul(h) + self.bias)


class GateMirrored(torch.nn.Module):
    """
    A versatile definition of a GRU gate. It include the normal formula and the activation and can constitute different
    types of gates depending on parameters.
    """
    def __init__(self, input_dim, output_dim, act):
        super().__init__()
        self.act = act

        self.W = Parameter(init_glorot([input_dim, output_dim]), requires_grad=True)
        self.U = Parameter(init_glorot([output_dim, output_dim]), requires_grad=True)
        self.bias = Parameter(init_glorot([1, output_dim]), requires_grad=True)

    def forward(self, x, h):
        # out = x * W + h * U + b
        return self.act(matmul(x, self.W) + matmul(h, self.U) + self.bias)


class GGate(torch.nn.Module):
    """
    A versatile definition of a GRU gate. It include the normal formula and the activation and can constitute different
    types of gates depending on parameters.
    """
    def __init__(self, num_nodes, num_feats, act, node_w=True, feats_w=True):
        super().__init__()
        self.act = act

        self.in_node_w = Parameter(init_glorot([num_nodes, 1]) if node_w else torch.ones(1), requires_grad=True)
        self.in_feat_w = Parameter(init_glorot([1, num_feats]) if feats_w else torch.ones(1), requires_grad=True)
        self.hist_node_w = Parameter(init_glorot([num_nodes, 1]) if node_w else torch.ones(1), requires_grad=True)
        self.hist_feats_w = Parameter(init_glorot([1, num_feats]) if feats_w else torch.ones(1), requires_grad=True)
        self.bias = Parameter(init_glorot([1, num_feats]), requires_grad=True)

    def forward(self, x, h):
        # out = W * x + U * h + b
        in_weighted = (x * self.in_feat_w) * self.in_node_w
        hist_weighted = (h * self.hist_feats_w) * self.hist_node_w
        return self.act(in_weighted + hist_weighted + self.bias)


class Readout:
    def __init__(self):
        super().__init__()

    def scatter_sum(self, src, index, dim=-1, out=None, dim_size=None):
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)

    def scatter_add(self, src, index, dim=-1, out=None, dim_size=None):
        return self.scatter_sum(src, index, dim, out, dim_size)

    def scatter_mean(self, src, index, dim=-1, out=None, dim_size=None):
        out = self.scatter_sum(src, index, dim, out, dim_size)
        dim_size = out.size(dim)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        count = self.scatter_sum(ones, index, index_dim, None, dim_size)
        count[count < 1] = 1
        count = broadcast(count, out, dim)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode='floor')
        return out





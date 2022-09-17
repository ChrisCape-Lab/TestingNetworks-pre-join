import torch
import torch.sparse
import random
import numpy as np


# GENERAL UTILS
# ----------------------------------------------------

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _set_seed(seed):
    """Set all the random functions' seeds to make the experiment reproducible"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


#
# INITIALIZERS
# -----------------------------------------

def init_uniform(shape, scale=0.05):
    """Uniform init."""
    values = np.random.uniform(-scale, scale, size=shape)
    return torch.tensor(values, dtype=torch.float32)


def init_glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    values = np.random.uniform(-init_range, init_range, size=shape)
    return torch.tensor(values, dtype=torch.float32)


def init_zeros(shape):
    """All zeros."""
    return torch.zeros(shape, dtype=torch.float32)


def init_ones(shape):
    """All ones."""
    return torch.ones(shape, dtype=torch.float32)


# DATA MANIPULATION UTILS
# ----------------------------------------------------


def normalize_adjacency(sp_adj, num_nodes):
    """
    This function takes as input an adj matrix as a sparse_coo_tensor normalize it by:
        - adding an identity matrix
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    :param sp_adj: torch.sparse_coo_tensor, is the adjacency matrix in a sparse form to normalize
    :param num_nodes: int, is the number of nodes of the current adjacency matrix
    :return: torch.sparse_coo_tensor, is the normalized adjacency matrix in a sparse form
    """
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_adj

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)

    return torch.sparse_coo_tensor(idx, vals)


def get_modularity_matrix(sp_adj):
    m = len(sp_adj._values())
    k1 = torch.sparse.sum(sp_adj, dim=1).to_dense()
    k2 = k1.reshape(k1.shape[0], 1)
    k1k2 = k1 * k2
    eij = k1k2 / (2 * m)
    del k1k2

    indices = torch.nonzero(eij).t()
    values = eij[indices[0], indices[1]]
    sp_eij = torch.sparse_coo_tensor(indices, values, eij.size())
    del eij

    return sp_adj - sp_eij


def make_sparse_eye(size):
    """
    This function basically create an eye matrix a sparse_coo_tensor format
    :param size: int, is the size of the square eye matrix
    :return: torch.sparse_coo_tensor, is the sparse tensor representation of the eye matrix with the size dimension
    """
    eye_idxs = []
    eye_list = [i for i in range(0, size)]
    eye_idxs.append(eye_list)
    eye_idxs.append(eye_list)
    vals = [1 for i in range(0, size)]
    eye = torch.sparse_coo_tensor(eye_idxs, vals, (size, size), dtype=torch.float32)

    return eye


# OPERATIONS UTILS
# ----------------------------------------------------

def matmul(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    if vect.size(0) == 0:
        pad = torch.zeros(k, dtype=torch.long, device=device)
    else:
        pad = torch.ones(k - vect.size(0), dtype=torch.long, device=device) * vect[-1]
    vect = torch.cat([vect, pad])

    return vect


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

"""
def make_sparse_tensor(adj, tensor_type, torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size**2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')
"""

def logsumexp(logits):
    m, _ = torch.max(logits, dim=1)
    m = m.view(-1, 1)
    sum_exp = torch.sum(torch.exp(logits-m), dim=1, keepdim=True)
    return m + torch.log(sum_exp)

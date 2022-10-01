import torch
import torch.nn.functional
from torch.nn.parameter import Parameter

from torch.nn import Linear

from src.testingnetworks._constants import DATA
from src.testingnetworks.model.layers.basic_layers import Dense
from src.testingnetworks.model.layers.attention_layers import GraphAttentionLayer
from src.testingnetworks.utils import DotDict, init_glorot


class Classifier(torch.nn.Module):

    INPUTS = [DATA.LABELS]

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = DotDict(config)

        self.output_type = self.config.output_type
        self.out_logits = self.output_type == "Logits"
        self.dim_list = self.config.layers_dim

    def forward(self, inputs):
        raise NotImplementedError


class DenseClassifier(Classifier):
    """
    The DenseClassifier is just a wrapper for the creation of a n-layer gnn_classifier made by dense layers.

    # Properties

    # Methods
        forward: the standard forward method of the torch.nn.Module that implement the network behaviours

    # Private Methods
        _build: construct the network based on the input parameters
    """
    def __init__(self, config, num_classes):
        super(DenseClassifier, self).__init__(config)

        out_dim = self.config.output_dim
        self.dim_list.insert(0, self.config.input_feats)
        if out_dim == "auto" and self.output_type == "Logits":
            out_dim = num_classes
        elif out_dim == "auto" and self.output_type == "Probabilities":
            out_dim = 1
        else:
            raise NotImplementedError('The chosen gnn_classifier output type has not been implemented yet')
        self.dim_list.append(out_dim)

        self.act = torch.nn.ReLU()
        self.bias = self.config.bias

        self.dense_classifier = self._build()

    def forward(self, inputs):
        return self.dense_classifier(inputs)

    def _build(self):
        layers = []
        if len(self.dim_list) == 2:
            layers.append(Dense(self.dim_list[0], self.dim_list[1], self.act, self.bias))
        else:
            for i in range(1, len(self.dim_list)):
                layers.append(Dense(self.dim_list[i-1], self.dim_list[i], self.act, self.bias))

        if self.output_type == "Probabilities":
            layers.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*layers)


class LinearClassifier(Classifier):
    """
        The LinearClassifier is just a wrapper for the creation of a n-layer gnn_classifier made by linear layers.

        # Properties

        # Methods
            forward: the standard forward method of the torch.nn.Module that implement the network behaviours

        # Private Methods
            _build: construct the network based on the input parameters
        """

    def __init__(self, config, num_nodes, num_classes):
        super(LinearClassifier, self).__init__(config)
        self.out_logits = True if self.config.output_type == "Logits" else False

        out_dim = self.config.output_dim
        self.dim_list.insert(0, self.config.input_feats * num_nodes)
        if out_dim == "auto" and self.output_type == "Logits":
            out_dim = num_nodes * num_classes
        elif out_dim == "auto" and self.config.output_type == "Probabilities":
            out_dim = num_nodes
        else:
            raise NotImplementedError('The chosen gnn_classifier output type has not been implemented yet')
        self.dim_list.append(out_dim)

        self.act = torch.nn.ReLU()

        self.linear_classifier = self._build()

    def forward(self, inputs):
        return self.linear_classifier(inputs)

    def _build(self):
        layers = []
        if len(self.dim_list) == 2:
            layers.append(Linear(self.dim_list[0], self.dim_list[1]))
        elif len(self.dim_list) == 3:
            layers.append(Linear(self.dim_list[0], self.dim_list[1]))
            layers.append(self.act)
            layers.append(Linear(self.dim_list[1], self.dim_list[2]))
        else:
            layers.append(Linear(self.dim_list[0], self.dim_list[1]))
            for i in range(2, len(self.dim_list)):
                layers.append(self.act)
                layers.append(Linear(self.dim_list[i-1], self.dim_list[i]))

        if self.output_type == "Probabilities":
            layers.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*layers)


class GATClassifier(Classifier):
    def __init__(self, config, num_classes, act=torch.nn.ReLU(), device='cpu'):
        super(GATClassifier, self).__init__(config)

        out_dim = self.config.output_dim
        self.dim_list.insert(0, self.config.input_feats)
        if out_dim == "auto" and self.output_type == "Logits":
            out_dim = num_classes
        elif out_dim == "auto" and self.config.output_type == "Probabilities":
            out_dim = 1
        else:
            raise NotImplementedError('The chosen gnn_classifier output type has not been implemented yet')
        self.dim_list.append(out_dim)

        self.act = act
        self.device = device
        self.out_att = GraphAttentionLayer(self.dim_list[0], self.dim_list[-1], dropout=config.dropout, alpha=config.alpha, concat=False)

    def forward(self, inputs):
        adj_matrix = inputs[0]
        nodes_embs = inputs[1]

        # For each layer, the sample is passed and processed in a sequential way
        nodes_embs = self.out_att.forward(adj_matrix, nodes_embs)

        return torch.nn.functional.log_softmax(nodes_embs, dim=1)


class EdgeClassifier(Classifier):

    def __init__(self, config, num_nodes, num_classes):
        super(EdgeClassifier, self).__init__(config)

        self.input_dim = self.config.input_feats
        self._set_kernel(self.config.kernel)

    def forward(self, inputs):
        return self.linear_classifier(inputs)

    def _set_kernel(self, kernel):
        if kernel=="bilinear":
            self.vars['bilinear'] = Parameter(init_glorot([self.input_dim, self.input_dim]), requires_grad=True)
            self.func = self._bilinear
        else:
            self.vars['linear1'] = Parameter(init_glorot([1, self.input_dim]), requires_grad=True)
            self.vars['linear2'] = Parameter(init_glorot([1, self.input_dim]), requires_grad=True)
            self.func = self._linear

    def _linear(self, a, b):
        return tf.reduce_sum(tf.cast(a * self.vars['linear1'] + b * self.vars['linear2'], dtype=tf.float32), axis=-1)

    def _bilinear(self, a, b):
        return tf.reduce_sum(tf.cast(tf.matmul(a, self.vars['bilinear']) * b, dtype=tf.float32), axis=-1)

    def _score(self, h, t):
        a = (self.func(h, t) - self.miu) * self.beta
        return 1.0 / (1.0 + tf.exp(a))


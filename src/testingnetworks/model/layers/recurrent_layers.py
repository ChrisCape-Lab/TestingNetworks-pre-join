import torch

from src.testingnetworks.model.layers.basic_layers import TopK, Gate, GateMirrored, GGate
from src.testingnetworks.utils import init_ones


class GRU(torch.nn.Module):
    """
    The basic GRU class
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.Sigmoid()):
        super(GRU, self).__init__()

        self.act = act

        self.update = Gate(input_dim, output_dim, act)
        self.reset = Gate(input_dim, output_dim, act)
        self.htilda = Gate(input_dim, output_dim, torch.nn.Tanh())
        self.one = init_ones(shape=[input_dim, output_dim])

    def forward(self, inputs, hist):
        """
        This layer requires two different inputs, the network input and the previous layer output. Then. it calculates
        the gates value and combine the input with the previous output (memory)
        :param inputs: the couple (input, memory) for the GRU unit
        :return: the activation of the combination of the inputs
        """
        update = self.update.forward(node_embs=inputs, h=hist)
        reset = self.update.forward(node_embs=inputs, h=hist)

        h_cap = self.htilda.forward(node_embs=inputs, h=reset * hist)

        return (self.one - update) * hist + update * h_cap


class LSTM(torch.nn.Module):
    """
    The basic LSTM class
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.Sigmoid()):
        super(LSTM, self).__init__()

        self.act = act

        self.forget = Gate(input_dim, output_dim, act)
        self.input = Gate(input_dim, output_dim, act)
        self.output = Gate(input_dim, output_dim, act)
        self.cell = Gate(input_dim, output_dim, torch.nn.Tanh())

    def forward(self, inputs, hist, carousel):
        forget = self.forget.forward(node_embs=inputs, h=hist)
        input = self.input.forward(node_embs=inputs, h=hist)
        output = self.output.forward(node_embs=inputs, h=hist)
        cell = self.cell.forward(node_embs=inputs, h=hist)

        c = forget * carousel + input * cell
        h = output * c

        return h, c


class GRUMirrored(torch.nn.Module):
    """
    The basic GRU class
    """
    def __init__(self, input_dim, output_dim, act=torch.nn.Sigmoid()):
        super(GRUMirrored, self).__init__()

        self.act = act

        self.update = GateMirrored(input_dim, output_dim, act)
        self.reset = GateMirrored(input_dim, output_dim, act)
        self.htilda = GateMirrored(input_dim, output_dim, torch.nn.Tanh())
        self.one = init_ones(shape=[1, output_dim])

    def forward(self, inputs, hist):
        """
        This layer requires two different inputs, the network input and the previous layer output. Then. it calculates
        the gates value and combine the input with the previous output (memory)
        :param inputs: the couple (input, memory) for the GRU unit
        :return: the activation of the combination of the inputs
        """
        update = self.update.forward(x=inputs, h=hist)
        reset = self.update.forward(x=inputs, h=hist)

        h_cap = self.htilda.forward(x=inputs, h=reset * hist)

        return (self.one - update) * hist + update * h_cap


class LSTMMirrored(torch.nn.Module):
    """
    The basic LSTM class
    """
    def __init__(self, input_dim: int, output_dim: int, act=torch.nn.Sigmoid()):
        super(LSTMMirrored, self).__init__()

        self.act = act

        self.forget = GateMirrored(input_dim=input_dim, output_dim=output_dim, act=act)
        self.input = GateMirrored(input_dim=input_dim, output_dim=output_dim, act=act)
        self.output = GateMirrored(input_dim=input_dim, output_dim=output_dim, act=act)
        self.cell = GateMirrored(input_dim=input_dim, output_dim=output_dim, act=torch.nn.Tanh())

    def forward(self, inputs: torch.Tensor, hist: torch.Tensor, carousel: int) -> (torch.Tensor, int):
        forget = self.forget(x=inputs, h=hist)
        input = self.input(x=inputs, h=hist)
        output = self.output(x=inputs, h=hist)
        cell = self.cell(x=inputs, h=hist)

        c = forget * carousel + input * cell
        h = output * c

        return h, c


class GGRU(torch.nn.Module):
    """
    The basic GRU class
    """
    def __init__(self, num_nodes, num_feats, act=torch.nn.Sigmoid(), node_w=True, feats_w=True):
        super(GGRU, self).__init__()

        self.act = act

        self.update = GGate(num_nodes, num_feats, act, node_w, feats_w)
        self.reset = GGate(num_nodes, num_feats, act, node_w, feats_w)
        self.htilda = GGate(num_nodes, num_feats, torch.nn.Tanh(), node_w, feats_w)

    def forward(self, inputs, hist):
        """
        This layer requires two different inputs, the network input and the previous layer output. Then. it calculates
        the gates value and combine the input with the previous output (memory)
        :param inputs: the couple (input, memory) for the GRU unit
        :return: the activation of the combination of the inputs
        """
        update = self.update.forward(x=inputs, h=hist)
        reset = self.update.forward(x=inputs, h=hist)

        h_cap = self.htilda.forward(x=inputs, h=reset * hist)

        return (1 - update) * hist + update * h_cap


class GLSTM(torch.nn.Module):
    """
    The basic LSTM class
    """
    def __init__(self, num_nodes, num_feats, act=torch.nn.Sigmoid(), node_w=True, feats_w=True):
        super(GLSTM, self).__init__()

        self.act = act

        self.forget = GGate(num_nodes, num_feats, act, node_w, feats_w)
        self.input = GGate(num_nodes, num_feats, act, node_w, feats_w)
        self.output = GGate(num_nodes, num_feats, act, node_w, feats_w)
        self.cell = GGate(num_nodes, num_feats, torch.nn.Tanh(), node_w, feats_w)

    def forward(self, inputs, hist, carousel):
        forget = self.forget.forward(x=inputs, h=hist)
        input = self.input.forward(x=inputs, h=hist)
        output = self.output.forward(x=inputs, h=hist)
        cell = self.cell.forward(x=inputs, h=hist)

        c = forget * carousel + input * cell
        h = output * c

        return h, c


class GRUTopK(torch.nn.Module):
    """
    The GRUTopK layer is a variant of the normal GRU layer, used in Evolve. Basically is a normal GRU layer with the
    difference of the addition of the TopK layer: this layer bias the previous history according to the best k nodes
    before applying the normal GRU unit.
    """
    def __init__(self, input_dim: int, output_dim: int, act=torch.nn.Sigmoid()):
        super(GRUTopK, self).__init__()

        self.act = act

        self.topk = TopK(input_dim, output_dim)

        self.update = Gate(input_dim, output_dim, act)
        self.reset = Gate(input_dim, output_dim, act)
        self.htilda = Gate(input_dim, output_dim, torch.nn.Tanh())
        self.one = init_ones(shape=[input_dim, output_dim])

    def forward(self, node_embs: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        This layer requires two different inputs, the network input and the previous layer output. Then. it calculates
        the gates value and combine the input with the previous output (memory)
        :param node_embs: Tensor (RxI), the input matrix of the layer
        :param history: Tensor (IxO), the previous state of the network
        :param mask: Tensor (Rx1), the mask
        :return: Tensor (IxO), the activation of the combination of the inputs
        """
        # Topk: RxI -> IxO
        x_topk = self.topk(node_embs, mask)

        update = self.update(x_topk, history)
        reset = self.reset(x_topk, history)

        h_cap = self.htilda(x_topk, reset * history)

        return (self.one - update) * history + update * h_cap


class LSTMTopK(torch.nn.Module):
    """
    The LSTMTopK layer is a variant of the normal LSTM layer, used in Evolve. Basically is a normal LSTM layer with the
    difference of the addition of the TopK layer: this layer bias the previous history according to the best k nodes
    before applying the normal GRU unit.
    """
    def __init__(self, input_dim: int, output_dim: int, act=torch.nn.Sigmoid()):
        super(LSTMTopK, self).__init__()

        self.act = act

        self.update = Gate(input_dim, output_dim, act)
        self.reset = Gate(input_dim, output_dim, act)
        self.htilda = Gate(input_dim, output_dim, torch.nn.Tanh())
        self.one = init_ones(shape=[input_dim, output_dim])

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        This layer requires two different inputs, the network input and the previous layer output. Then. it calculates
        the gates value and combine the input with the previous output (memory)
        :param history: Tensor (RxI), the input matrix of the layer
        :return: Tensor(IxO), the activation of the combination of the inputs
        """
        x_topk = history

        update = self.update(node_embs=x_topk, h=history)
        reset = self.reset(node_embs=x_topk, h=history)

        h_cap = self.htilda(node_embs=x_topk, h=reset * history)

        return (self.one - update) * history + update * h_cap

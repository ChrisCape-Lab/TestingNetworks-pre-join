import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_list: list):
        super(Encoder, self).__init__()
        self.input_list = input_list

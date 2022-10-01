import torch


# ABSTRACT CLASS
# ----------------------------------------------------

class Encoder(torch.nn.Module):
    def __init__(self, input_list: list):
        super(Encoder, self).__init__()
        self.input_list = input_list


# BUILDER
# ----------------------------------------------------

def build_encoder(network_args: dict, num_features: int, num_nodes: int, device: str = 'cpu') -> Encoder:
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    layers_dim = network_args['layers_dim']
    layers_dim.insert(0, network_args['node_feats_num'] if network_args['node_feats_num'] != "auto" else num_features)
    layers_dim.append(network_args['output_dim'])

    # Build the correct GCN
    if network_args['network'] == "DenseNet":
        from src.testingnetworks.model.encoders.densenet import DenseNetwork
        return DenseNetwork(layers_dimensions=layers_dim, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "GCN":
        from src.testingnetworks.model.encoders.gcn import GCN
        return GCN(layers_dimensions=layers_dim, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "FastGCN":
        from src.testingnetworks.model.encoders.fastgcn import FastGCN
        return FastGCN(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "TGCN":
        from src.testingnetworks.model.encoders.tgcn import TGCN
        return TGCN(layers_dimensions=layers_dim,  num_nodes=num_nodes, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "TGCNe":
        from src.testingnetworks.model.encoders.tgcn import TGCNe
        return TGCNe(layers_dim, num_nodes=num_nodes, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "TGCNseq":
        from src.testingnetworks.model.encoders.tgcn import TGCNseq
        return TGCNseq(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args['bias'], device=device)

    elif network_args['network'] == "Evolve-h":
        from src.testingnetworks.model.encoders.evolvegcn import EvolveGCNvH
        return EvolveGCNvH(layers_dim, act=torch.nn.RReLU(), skipfeats=network_args['skipfeats'], device=device)

    elif network_args['network'] == "Evolve-o":
        from src.testingnetworks.model.encoders.evolvegcn import EvolveGCNvO
        return EvolveGCNvO(layers_dim, act=torch.nn.RReLU(), skipfeats=network_args['skipfeats'], device=device)
    else:
        raise NotImplementedError('The chosen GCN has not been implemented yet')

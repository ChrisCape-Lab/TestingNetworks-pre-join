import torch

from src.testingnetworks.utils import DotDict


# MODEL
# ----------------------------------------------------

class GCNClassifier(torch.nn.Module):
    def __init__(self, config, data, device):
        super().__init__()
        self.config = DotDict(config)

        self.gcn = _build_network(self.config.network_args, data, device)
        self.classifier = _build_classifier(self.config.classifier_args, data.num_nodes, data.num_classes)
        self.connector = self.config.connector

        self.node_embeddings = torch.zeros(1)
        self.out = torch.zeros(1)

    def forward(self, sample: dict) -> torch.Tensor:
        self.node_embeddings = self.gcn.forward(sample)
        cls_input = self.node_embeddings

        if self.connector == 'Flatten':
            cls_input = torch.flatten(cls_input)
            self.out = self.classifier(cls_input)
        if self.connector == 'Iterative':
            out = []
            for node in cls_input:
                out.append(self.classifier(node))
            self.out = torch.cat(out, dim=-1)
        else:
            self.out = self.classifier(cls_input)

        return self.out


# BUILDERS
# ----------------------------------------------------

def _build_network(network_args, dataset, device):
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    network_args = DotDict(network_args)
    layers_dim = network_args.layers_dim
    layers_dim.insert(0, network_args.feats_per_node if network_args.feats_per_node != "auto" else dataset.num_features)
    layers_dim.append(network_args.output_dim)

    # Build the correct GCN
    if network_args.network == "Dense":
        from src.testingnetworks.model.encoders.densenet import DenseNetwork
        return DenseNetwork(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "GCN":
        from src.testingnetworks.model.encoders.gcn import GCN
        return GCN(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "FastGCN":
        from src.testingnetworks.model.encoders.fastgcn import FastGCN
        return FastGCN(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "tGCN":
        from src.testingnetworks.model.encoders.tgcn import TGCN
        return TGCN(layers_dim,  num_nodes=dataset.num_nodes, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "TGCNe":
        from src.testingnetworks.model.encoders.tgcn import TGCNe
        return TGCNe(layers_dim, num_nodes=dataset.num_nodes, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "TGCNseq":
        from src.testingnetworks.model.encoders.tgcn import TGCNseq
        return TGCNseq(layers_dim, act=torch.nn.LeakyReLU(), bias=network_args.bias, device=device)
    elif network_args.network == "Evolve-h":
        from src.testingnetworks.model.encoders.evolvegcn import EvolveGCNvH
        return EvolveGCNvH(layers_dim, act=torch.nn.RReLU(), skipfeats=network_args.skipfeats, device=device)
    elif network_args.network == "Evolve-o":
        from src.testingnetworks.model.encoders.evolvegcn import EvolveGCNvO
        return EvolveGCNvO(layers_dim, act=torch.nn.RReLU(), skipfeats=network_args.skipfeats, device=device)
    else:
        raise NotImplementedError('The chosen GCN has not been implemented yet')


def _build_classifier(classifier_args, num_nodes, num_classes):
    """Return the correct classifier to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    classifier_args = DotDict(classifier_args)
    if classifier_args.classifier == "Linear":
        from src.testingnetworks.model.decoders.classifiers import LinearClassifier
        return LinearClassifier(classifier_args, num_nodes, num_classes)
    elif classifier_args.classifier == "Dense":
        from src.testingnetworks.model.decoders.classifiers import DenseClassifier
        return DenseClassifier(classifier_args, num_classes)
    if classifier_args.classifier == "GAT":
        from src.testingnetworks.model.decoders.classifiers import GATClassifier
        return GATClassifier(classifier_args, num_classes)
    else:
        raise NotImplementedError('The chosen classifier has not been implemented yet')

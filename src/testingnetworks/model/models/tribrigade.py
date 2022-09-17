import torch

from src.testingnetworks.utils import DotDict


class TriBrigade(torch.nn.Module):
    def __init__(self, config, autoencoder, gcnclassifier, rec_error_fun, data, device):
        super().__init__()
        self.config = DotDict(config)

        self.autoencoder = autoencoder
        self.gcn = gcnclassifier.gcn
        self.classifier = _build_classifier(self.config.classifier_args, data.num_nodes, data.num_classes)
        self.rec_error_fun = rec_error_fun

        # Disable autoencoder parameters gradient
        for _, p in enumerate(self.autoencoder.parameters()):
            p.requires_grad_(False)

        self.node_embeddings = torch.zeros(1)
        self.out = torch.zeros(1)

    def forward(self, inputs):
        adj_list = inputs[0]
        node_embs_list = inputs[1]
        # Since the input is a list, the AE input is just the last element of the list
        ae_inputs = [adj_list[-1], node_embs_list[-1]]
        # Reconstruct the initial adj and features
        reconstruction = self.autoencoder.forward(ae_inputs)
        # Get the reconstruction loss for the autoencoder
        anomaly_score = self.rec_error_fun(reconstruction, ae_inputs)
        # Get the node embeddings from the classifier
        embeddings = self.gcn.forward(inputs)
        anomaly_score = torch.reshape(anomaly_score, (anomaly_score.shape[0], 1))
        # Combine the reconstruction error and the node embeddings
        cls_input = torch.cat((embeddings, anomaly_score), dim=1)
        # Classify
        self.out = self.classifier.forward(cls_input)

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
    if network_args.network == "GCN":
        from src.testingnetworks.model.encoders.networks import GCN
        return GCN(layers_dim, act=torch.nn.ReLU(), bias=False, device=device)
    elif network_args.network == "Evolve-h":
        from src.testingnetworks.model.encoders.networks import EvolveGCNvH
        return EvolveGCNvH(layers_dim, skipfeats=network_args.skipfeats, device=device)
    elif network_args.network == "Evolve-o":
        from src.testingnetworks.model.encoders.networks import EvolveGCNvO
        return EvolveGCNvO(layers_dim, skipfeats=network_args.skipfeats, device=device)
    elif network_args.network == "addGraph":
        from src.testingnetworks.model.encoders.networks import AddGraph
        return AddGraph(layers_dim, nb_window=network_args.nb_window, num_nodes=dataset.num_nodes, bias=False, device=device)
    elif network_args.network == "GAT":
        from src.testingnetworks.model.encoders.networks import GAT
        return GAT(layers_dim, network_args.dropout, network_args.alpha, network_args.nheads)
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

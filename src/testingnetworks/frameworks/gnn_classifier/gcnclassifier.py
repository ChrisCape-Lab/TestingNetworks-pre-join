import torch

from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.model.encoders._encoder import build_encoder
from src.testingnetworks.model.decoders.classifiers import Classifier
from src.testingnetworks.utils import DotDict


# MODEL
# ----------------------------------------------------

class GCNClassifier(toch.nn.Module):
    def __init__(self, model_config: dict, data: GraphDataExtractor, device: str = 'cpu'):
        super(GCNClassifier, self).__init__()
        # Network, gnn_classifier and connector
        self.network = build_encoder(network_args=model_config['network_args'], num_nodes=data.num_nodes, num_features=data.num_features, device=device)
        self.classifier = _build_classifier(model_config['classifier_args'], data.num_nodes, data.num_classes)
        self.connector = model_config['connector']

        self.node_embeddings = torch.zeros(1)
        self.out = torch.zeros(1)

        self.input_list = self.network.input_list
        self.output_list = self.classifier.INPUTS

    def forward(self, sample: dict) -> torch.Tensor:
        self.node_embeddings = self.network(sample)
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

def _build_classifier(classifier_args, num_nodes, num_classes) -> Classifier:
    """Return the correct gnn_classifier to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
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
        raise NotImplementedError('The chosen gnn_classifier has not been implemented yet')

import torch
import numpy as np

from src.testingnetworks._constants import DATA
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor

from src.testingnetworks.utils import normalize_adjacency, get_modularity_matrix


class Tasker:
    """
    The Tasker is an abstract class that basically prepare the dataset in order to be used for a particular task. This
    abstract contains the main attributes and methods that a tasker must have.

    # Properties
        datasets_output: dataset.Dataset, is the Dataset class associated to the tasker dataset
        num_classes: int, is the classification task's number of classes (get from the dataset)

    # Methods
        get_sample(index): return the item(s) of the dataset associated to the time index passed, formatted accordingly
                            to the task and the window
    """

    def __init__(self, data_extractor: GraphDataExtractor, dataloading_config: dict, inputs_list: list, outputs_list: list):
        self.data_extractor = data_extractor
        self.dataloading_config = dataloading_config
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list

        self.adj_matrices_list = None
        self.connection_matrices_list = None
        self.modularity_matrices_list = None
        self.nodes_features_list = None
        self.edge_features_list = None
        self.nodes_masks_list = None

        if self.dataloading_config['in_memory_load']:
            self._in_memory_load()
            del self.data_extractor.node_dataset
            del self.data_extractor.edge_dataset

        self.class_weights = self._get_class_weights()
        print(self.class_weights)

    # PRIVATE METHODS
    # ----------------------------------------------------

    def _in_memory_load(self):
        lengths_list = list()

        if DATA.ADJACENCY_MATRIX in self.inputs_list:
            self.adj_matrices_list = self.data_extractor.load_all_adjacency_matrices(weighted=self.dataloading_config['weighted'],
                                                                                     directed=self.dataloading_config['directed'])
            if self.adj_matrices_list is not None: lengths_list.append(len(self.adj_matrices_list))
        if DATA.CONNECTION_MATRIX in self.inputs_list:
            self.connection_matrices_list = None
            if self.connection_matrices_list is not None: lengths_list.append(len(self.connection_matrices_list))

        if DATA.MODULARITY_MATRIX in self.inputs_list:
            self.modularity_matrices_list = None
            if self.modularity_matrices_list is not None: lengths_list.append(len(self.modularity_matrices_list))

        if DATA.NODE_FEATURES in self.inputs_list:
            self.nodes_features_list = self.data_extractor.load_all_nodes_features(normalize=self.dataloading_config['normalize_features'])
            if self.nodes_features_list is not None: lengths_list.append(len(self.nodes_features_list))

        if DATA.EDGE_FEATURES in self.inputs_list:
            self.edge_features_list = None
            if self.edge_features_list is not None: lengths_list.append(len(self.edge_features_list))

        if DATA.NODE_MASK in self.inputs_list:
            self.nodes_masks_list = self.data_extractor.load_all_nodes_masks()
            if self.nodes_masks_list is not None: lengths_list.append(len(self.nodes_masks_list))

        if DATA.NODE_LABELS in self.outputs_list:
            self.labels_list = self.data_extractor.load_all_nodes_labels()
            if self.labels_list is not None: lengths_list.append(len(self.labels_list))

        if DATA.EDGE_LABELS in self.outputs_list:
            self.labels_list = self.data_extractor.load_all_edges_labels()
            if self.labels_list is not None: lengths_list.append(len(self.labels_list))

        # Check if all the extracted dimensions are the same to avoid loading errors
        assert len(set(lengths_list)) == 1

    def _get_class_weights(self):
        """
        This method determine the class weights based on the number of occurrences of each class in the dataset.
        :return: a list containing for each class the associated weight
        """
        classes_weights = []

        # Concatenate all the values of lables on a single list
        label_values = []
        for label_t in self.labels_list:
            label_values = np.concatenate((label_values, label_t['vals']))
        # Get the classes and the occurrences of classes
        (classes, counts) = np.unique(label_values, return_counts=True)

        # Calculate the weights for each class
        for label, count in zip(classes, counts):
            weight_c = len(label_values) / (len(classes) * count)
            classes_weights.append(weight_c)

        classes_weights[0] = 1

        return classes_weights

    def _load_sample(self, start: int, end: int, output_list: list, windowed: bool):
        output = dict()

        if not windowed:
            output = self._load_single_sample(index=start, output_list=output_list)
        else:
            output = self._load_windowed_sample(start=start, end=end, output_list=output_list)

        if DATA.NODE_LABELS in output_list or DATA.EDGE_LABELS in output_list:
            output[DATA.LABELS] = self.labels_list[end - 1]['vals']

        # If there are only nodes or edge labels as labels (simple gnn_classifier, not autoencoder or else), output the direct tensor
        if len(output.keys()) == 1 and list(output.keys())[0] == DATA.LABELS:
            output = output[list(output.keys())[0]]

        return output

    def _load_single_sample(self, index: int, output_list: list):
        output = dict()

        in_memory_load = self.dataloading_config['in_memory_load']

        if DATA.ADJACENCY_MATRIX in output_list:
            if in_memory_load:
                adj_matrix = self.adj_matrices_list[index]
            else:
                adj_matrix = self.data_extractor.load_adjacency_matrix_at_time(time=index, weighted=self.dataloading_config['weighted'],
                                                                               directed=self.dataloading_config['directed'])
            idx_list, vals_list = adj_matrix
            adj_matrix = torch.sparse_coo_tensor(idx_list, vals_list, (self.data_extractor.num_nodes, self.data_extractor.num_nodes), dtype=torch.float32)
            if self.dataloading_config['normalize_adjacency']:
                adj_matrix = normalize_adjacency(adj_matrix, self.data_extractor.num_nodes)
            output[DATA.ADJACENCY_MATRIX] = adj_matrix
        if DATA.CONNECTION_MATRIX in output_list:
            pass
        if DATA.MODULARITY_MATRIX in output_list:
            pass
        if DATA.NODE_FEATURES in output_list:
            if in_memory_load:
                node_features = self.nodes_features_list[index]
            else:
                node_features = self.data_extractor.load_node_features_at_time(time=index, normalize=self.dataloading_config['normalize_features'])
            output[DATA.NODE_FEATURES] = torch.tensor(node_features, dtype=torch.float32)
        if DATA.NODE_MASK in output_list:
            if in_memory_load:
                nodes_mask = self.nodes_masks_list[index]
            else:
                nodes_mask = self.data_extractor.load_nodes_masks(time=index)
            output[DATA.NODE_MASK] = torch.tensor(nodes_mask, dtype=torch.float32)

        return output

    def _load_windowed_sample(self, start: int, end: int, output_list: list):
        output = dict()

        in_memory_load = self.dataloading_config['in_memory_load']

        if DATA.ADJACENCY_MATRIX in output_list:
            adjacency_matrices = list()
            if in_memory_load:
                adj_matrices = self.adj_matrices_list[start:end]
            else:
                adj_matrices = [self.data_extractor.load_adjacency_matrix_at_time(time=index, weighted=self.dataloading_config['weighted'],
                                                                                  directed=self.dataloading_config['directed']) for index in range(start, end)]
            for idx_list, vals_list in adj_matrices:
                adj_matrix = torch.sparse_coo_tensor(idx_list, vals_list, (self.data_extractor.num_nodes, self.data_extractor.num_nodes),
                                                     dtype=torch.float32)
                if self.dataloading_config['normalize_adjacency']:
                    adj_matrix = normalize_adjacency(adj_matrix, self.data_extractor.num_nodes)
                adjacency_matrices.append(adj_matrix)
            output[DATA.ADJACENCY_MATRIX] = adjacency_matrices
        if DATA.CONNECTION_MATRIX in output_list:
            pass
        if DATA.MODULARITY_MATRIX in output_list:
            pass
        if DATA.NODE_FEATURES in output_list:
            if in_memory_load:
                node_features = self.nodes_features_list[start:end]
            else:
                node_features = [self.data_extractor.load_node_features_at_time(time=index, normalize=self.dataloading_config['normalize_features']) for
                                 index in range(start, end)]
            output[DATA.NODE_FEATURES] = [torch.tensor(node_feats, dtype=torch.float32) for node_feats in node_features]
        if DATA.NODE_MASK in output_list:
            if in_memory_load:
                nodes_mask = self.nodes_masks_list[start:end]
            else:
                nodes_mask = [self.data_extractor.load_nodes_masks(time=index) for index in range(start, end)]
            output[DATA.NODE_MASK] = [torch.tensor(nm, dtype=torch.float32) for nm in nodes_mask]

        return output

    # GET SAMPLE
    # ----------------------------------------------------

    def get_sample(self, index: int, time_window: int):
        # Adjust index to be used with foreach
        end = index + 1
        start = index - time_window + 1
        start = start if start >= 0 else 0

        # Load the input of the model and then add the features in a unique dictionary
        sample = self._load_sample(start=start, end=end, output_list=self.inputs_list, windowed=time_window > 1)
        labels = self._load_sample(start=start, end=end, output_list=self.outputs_list, windowed=time_window > 1)
        sample[DATA.LABELS] = labels

        return sample


def _get_modularity_matrix(sp_adj_list, index, num_nodes):
    end = index + 1
    start = index - 4 + 1
    start = start if start >= 0 else 0

    cumulative_adj = torch.sparse_coo_tensor([[0], [0]], [0], (num_nodes, num_nodes), dtype=torch.float32)
    for idx_list, vals_list in sp_adj_list[start:end]:
        cumulative_adj += torch.sparse_coo_tensor(idx_list, vals_list, (num_nodes, num_nodes), dtype=torch.float32)

    return get_modularity_matrix(cumulative_adj)

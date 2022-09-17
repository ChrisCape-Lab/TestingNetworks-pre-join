import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler

from src.testingnetworks.commons.datapreprocess.amlsim_preprocess import ECOLS, NCOLS


# UTILS
# ----------------------------------------------------

def get_active_nodes_mask(num_nodes: int, sources: list, destinations: list):
    nodes_mask = np.zeros(num_nodes) - float("Inf")
    nodes_mask[sources] = 0
    nodes_mask[destinations] = 0

    return nodes_mask


# DATA EXTRACTOR
# ----------------------------------------------------

class GraphDataExtractor:
    def __init__(self, pd_node_dataset: pd.DataFrame, pd_edge_dataset: pd.DataFrame, dataset_config: dict):
        # Configuration data
        self.node_dataset = pd_node_dataset
        self.edge_dataset = pd_edge_dataset
        self.dataset_config = dataset_config

        # Dataset attributes
        self.num_classes = len(self.node_dataset['label'].unique())
        self.num_nodes = len(self.node_dataset[NCOLS.id].unique())
        self.num_features = len(self.node_dataset.columns) - 6  # The ones removed in load_nodes_features

        self.start_time = self.edge_dataset['time'].min()
        self.end_time = self.edge_dataset['time'].max()

        self.is_static = self.dataset_config['is_static']

    def load_all_adjacency_matrices(self, weighted: bool, directed: bool) -> list:
        adj_list = []

        # For each time instant (week), compute the [weighted][directed] adjacency matrix and store it in a list where index is the time instant
        for time in range(self.start_time, self.end_time+1):
            adj_list.append(self.load_adjacency_matrix_at_time(time=time, weighted=weighted, directed=directed))

        return adj_list

    def load_adjacency_matrix_at_time(self, time: int, weighted: bool, directed: bool) -> (list, list):
        tx_window = self.edge_dataset[self.edge_dataset['time'] == time]
        idx_list = []
        x_list = []
        y_list = []
        vals_list = []
        for _, row in tx_window.iterrows():
            x_list.append(row[ECOLS.source])
            y_list.append(row[ECOLS.dest])
            vals_list.append(row[ECOLS.weight] if weighted else 1)
            if not directed:
                x_list.append(row[ECOLS.dest])
                y_list.append(row[ECOLS.source])
                vals_list.append(row[ECOLS.weight] if weighted else 1)
        idx_list.append(x_list)
        idx_list.append(y_list)
        sp_adj = (idx_list, vals_list)

        return sp_adj

    def _load_sp_conn_matrix(self):
        sp_connection_matrix_list = []
        cumulative_adj = torch.sparse_coo_tensor([[0], [0]], [0], (self.num_nodes, self.num_nodes), dtype=torch.float32)
        for idx_list, vals_list in self.sparse_adjacency:
            cumulative_adj *= 0.5
            cumulative_adj += torch.sparse_coo_tensor(idx_list, vals_list, (self.num_nodes, self.num_nodes), dtype=torch.float32)
            sp_connection_matrix = (cumulative_adj._indices(), cumulative_adj._values())
            sp_connection_matrix_list.append(sp_connection_matrix)

        return sp_connection_matrix_list

    def load_all_nodes_features(self, normalize: bool) -> list:
        feat_list = []
        for time in range(self.start_time, self.end_time+1):
            feats = self.load_node_features_at_time(time=time, normalize=normalize)
            if feats is None:
                continue
            feat_list.append(feats)

        return feat_list

    def load_node_features_at_time(self, time: int, normalize: bool) -> dict or None:
        acc_window = self.node_dataset[self.node_dataset['time'] == time]
        if len(acc_window.index) == 0:
            return None

        acc_window_feats = acc_window.drop(columns=['id', 'time', 'exLaunderer', 'deposit', 'bankID', 'label'], axis=1)

        if normalize:
            scaler = MinMaxScaler()
            acc_window_feats = scaler.fit_transform(acc_window_feats)
        else:
            acc_window_feats = acc_window_feats.values

        return acc_window_feats

    def load_all_nodes_labels(self) -> list:
        node_label_list = []

        # For each time instant (week), compute the node labels and store it in a list where index is the time instant
        for time in range(self.start_time, self.end_time+1):
            node_window = self.node_dataset[self.node_dataset['time'] == time]
            if len(node_window.index) == 0:
                continue
            node_label_list.append({'idx': node_window[NCOLS.id].tolist(), 'vals': node_window[NCOLS.label].tolist()})

        return node_label_list

    def load_nodes_labels_at_time(self, time: int) -> dict or None:
        node_window = self.node_dataset[self.node_dataset['time'] == time]
        if len(node_window.index) == 0:
            return None
        return {'idx': node_window[NCOLS.id].tolist(), 'vals': node_window[NCOLS.label].tolist()}

    def load_all_edges_labels(self) -> list:
        edge_label_list = []

        for time in range(self.start_time, self.end_time+1):
            edges_labels = self.load_edges_labels_at_time(time=time)
            if edges_labels is None:
                continue

            edge_label_list.append(edges_labels)

        return edge_label_list

    def load_edges_labels_at_time(self, time: int) -> dict or None:
        tx_window = self.edge_dataset[self.edge_dataset['time'] == time]
        if len(tx_window.index) == 0:
            return None

        return {'idx': tx_window[[ECOLS.source, ECOLS.dest]].values, 'vals': tx_window[ECOLS.label].tolist()}

    def load_all_nodes_masks(self) -> list:
        nodes_mask_list = []

        for time in range(self.start_time, self.end_time+1):
            nodes_masks = self.load_nodes_masks(time=time)
            if nodes_masks is None:
                continue

            nodes_mask_list.append(nodes_masks)

        return nodes_mask_list

    def load_nodes_masks(self, time: int):
        tx_window = self.edge_dataset[self.edge_dataset['time'] == time]
        if len(tx_window.index) == 0:
            return None

        nodes_mask = get_active_nodes_mask(self.num_nodes, tx_window['orig_id'].unique(), tx_window['bene_id'].unique())

        return nodes_mask


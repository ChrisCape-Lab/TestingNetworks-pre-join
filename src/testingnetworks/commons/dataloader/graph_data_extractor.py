import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler

from src.testingnetworks._constants import NODE_COLUMNS as NCOLS, EDGE_COLUMNS as ECOLS


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
        self.num_classes = len(self.node_dataset[NCOLS.LABEL].unique())
        self.num_nodes = len(self.node_dataset[NCOLS.ID].unique())
        self.num_features = len(self.node_dataset.columns) - 6  # The ones removed in load_nodes_features

        self.start_time = self.edge_dataset[ECOLS.TIME].min()
        self.end_time = self.edge_dataset[ECOLS.TIME].max() + 1

        self.is_static = self.dataset_config['is_static']

    def load_all_adjacency_matrices(self, weighted: bool, directed: bool) -> list:
        adj_list = []

        # For each time instant (week), compute the [weighted][directed] adjacency matrix and store it in a list where index is the time instant
        for time in range(self.start_time, self.end_time):
            adj_list.append(self.load_adjacency_matrix_at_time(time=time, weighted=weighted, directed=directed))

        return adj_list

    def load_adjacency_matrix_at_time(self, time: int, weighted: bool, directed: bool) -> (list, list):
        tx_window = self.edge_dataset[self.edge_dataset[ECOLS.TIME] == time]
        idx_list = []

        x_list = tx_window[ECOLS.ORIGINATOR].tolist()
        y_list = tx_window[ECOLS.BENEFICIARY].tolist()
        vals_list = tx_window[ECOLS.AMOUNT].tolist() if weighted else [1]*len(tx_window[ECOLS.AMOUNT])
        if not directed:
            x_list = tx_window[ECOLS.BENEFICIARY].tolist()
            y_list = tx_window[ECOLS.ORIGINATOR].tolist()
            vals_list = tx_window[ECOLS.AMOUNT].tolist() if weighted else [1]*len(tx_window[ECOLS.AMOUNT])

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
        for time in range(self.start_time, self.end_time):
            feats = self.load_node_features_at_time(time=time, normalize=normalize)
            if feats is None:
                continue
            feat_list.append(feats)

        return feat_list

    def load_node_features_at_time(self, time: int, normalize: bool) -> dict or None:
        acc_window = self.node_dataset[self.node_dataset[NCOLS.TIME] == time]
        if len(acc_window.index) == 0:
            return None

        acc_window_feats = acc_window.drop(columns=[NCOLS.ID, NCOLS.TIME, NCOLS.EX_LAUNDERER, NCOLS.DEPOSIT, NCOLS.BANK_ID, NCOLS.LABEL], axis=1)

        if normalize:
            scaler = MinMaxScaler()
            acc_window_feats = scaler.fit_transform(acc_window_feats)
        else:
            acc_window_feats = acc_window_feats.values

        return acc_window_feats

    def load_all_nodes_labels(self) -> list:
        node_label_list = []

        # For each time instant (week), compute the node labels and store it in a list where index is the time instant
        for time in range(self.start_time, self.end_time):
            node_window = self.node_dataset[self.node_dataset[NCOLS.TIME] == time]
            if len(node_window.index) == 0:
                continue
            node_label_list.append({'idx': node_window[NCOLS.ID].tolist(), 'vals': node_window[NCOLS.LABEL].tolist()})

        return node_label_list

    def load_nodes_labels_at_time(self, time: int) -> dict or None:
        node_window = self.node_dataset[self.node_dataset[NCOLS.TIME] == time]
        if len(node_window.index) == 0:
            return None

        return {'idx': node_window[NCOLS.ID].tolist(), 'vals': node_window[NCOLS.LABEL].tolist()}

    def load_all_edges_labels(self) -> list:
        edge_label_list = []

        for time in range(self.start_time, self.end_time):
            edges_labels = self.load_edges_labels_at_time(time=time)
            if edges_labels is None:
                continue

            edge_label_list.append(edges_labels)

        return edge_label_list

    def load_edges_labels_at_time(self, time: int) -> dict or None:
        tx_window = self.edge_dataset[self.edge_dataset[ECOLS.TIME] == time]
        if len(tx_window.index) == 0:
            return None

        return {'idx': tx_window[[ECOLS.ORIGINATOR, ECOLS.BENEFICIARY]].values, 'vals': tx_window[ECOLS.LABEL].tolist()}

    def load_all_nodes_masks(self) -> list:
        nodes_mask_list = []

        for time in range(self.start_time, self.end_time):
            nodes_masks = self.load_nodes_masks(time=time)
            nodes_mask_list.append(nodes_masks)

        return nodes_mask_list

    def load_nodes_masks(self, time: int):
        tx_window = self.edge_dataset[self.edge_dataset[ECOLS.TIME] == time]

        nodes_mask = get_active_nodes_mask(self.num_nodes, tx_window[ECOLS.ORIGINATOR].unique(), tx_window[ECOLS.BENEFICIARY].unique())

        return nodes_mask

import torch
import torch.nn.functional

from src.testingnetworks._constants import DATA


def build_reconstruction_error(reconstruction_error_type: str):
    """Return the correct reconstruction error function according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    if reconstruction_error_type == 'standard':
        return reconstruction_error
    elif reconstruction_error_type == 'community':
        return community_reconstruction_error
    else:
        raise NotImplementedError


def reconstruction_error(output_dict: dict, labels_dict: dict, alpha: float = 0.5) -> (torch.Tensor, torch.Tensor):
    rec_adjacency = output_dict[DATA.ADJACENCY_MATRIX]
    rec_features = output_dict[DATA.NODE_FEATURES]

    adjacency = labels_dict[DATA.ADJACENCY_MATRIX]
    features = labels_dict[DATA.NODE_FEATURES]

    # Attribute reconstruction loss
    diff_attribute = torch.pow(rec_features - features, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))

    # structure reconstruction loss
    diff_structure = torch.pow(rec_adjacency - adjacency, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))

    error = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    cost = alpha * torch.mean(attribute_reconstruction_errors) + (1 - alpha) * torch.mean(structure_reconstruction_errors)
    del attribute_reconstruction_errors, structure_reconstruction_errors

    return error, cost


def community_reconstruction_error(output, labels):
    attr_out, struct_out, comm_out, node_embs, comm_embs = output
    adj_labels, features_labels, comm_labels, _ = labels

    # ==== COMMUNITY RECONSTRUCTION LOSS ====
    diff_structure = torch.pow(comm_out - comm_labels, 2)
    community_rec_error = torch.sqrt(torch.sum(diff_structure, 1))
    del diff_structure

    # ==== RECONSTRUCTION LOSS ====
    # Attribute reconstruction loss
    diff_attribute = torch.pow(attr_out - features_labels, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    del diff_attribute

    # Structure reconstruction loss
    diff_structure = torch.pow(struct_out - adj_labels, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    del diff_structure

    rec_error = 0.5 * attribute_reconstruction_errors + (1 - 0.5) * structure_reconstruction_errors

    # ==== TOTAL LOSS ====
    cost = community_rec_error + rec_error

    return cost


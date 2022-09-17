import torch
import torch.nn.functional


def reconstruction_error(output, labels):
    attr_out, struct_out = output
    adj_labels, features_labels, _ = labels

    # Attribute reconstruction loss
    diff_attribute = torch.pow(attr_out - features_labels, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))

    # structure reconstruction loss
    diff_structure = torch.pow(struct_out - adj_labels, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))

    error = 0.5 * attribute_reconstruction_errors + (1 - 0.5) * structure_reconstruction_errors

    return error


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


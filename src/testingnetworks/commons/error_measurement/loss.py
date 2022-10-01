import torch
import torch.nn.functional

from src.testingnetworks.utils import logsumexp


# BUILDER
# ----------------------------------------------------

def build_loss(loss_type: str):
    """Return the correct loss according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    if loss_type == "cross_entropy":
        return cross_entropy
    elif loss_type == "bce_loss":
        return bce_loss
    elif loss_type == "bce_logits_loss":
        return bce_logits_loss
    elif loss_type == "autoencoder_loss":
        return autoencoder_loss
    elif loss_type == "community_autoencoder_loss":
        return community_autoencoder_loss
    else:
        raise NotImplementedError('The chosen loss has not been implemented yet')


# CLASSIFICATION LOSSES
# ----------------------------------------------------

def cross_entropy(output, labels, mask=None, class_weights=None, device='cpu'):
    true_classes = torch.tensor(labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss = torch.nn.functional.cross_entropy(input=output, target=true_classes, weight=class_weights, reduction='none')

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)
        loss *= mask

    return torch.mean(loss)


def bce_logits_loss(output, labels, mask=None, class_weights=None, device='cpu'):
    true_classes = torch.tensor(labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input=output, target=true_classes, reduction='none', pos_weight=class_weights)

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)
        loss *= mask

    return torch.mean(loss)


def bce_loss(output, labels, mask=None, class_weights=None, device='cpu'):
    predictions = output if output.dim() == 1 else output.argmax(dim=1)
    predictions = predictions.float()
    true_classes = torch.tensor(labels, dtype=torch.float32)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    weights = None
    if class_weights is not None:
        weights = torch.mul(true_classes, class_weights[1] - class_weights[0])
        weights = torch.add(weights, class_weights[0])
        weights = weights.float()

    loss = torch.nn.functional.binary_cross_entropy(input=predictions, target=true_classes, weight=weights, reduction='none')

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)
        loss *= mask

    return torch.mean(loss)


# AUTOENCODER LOSSES
# ----------------------------------------------------

def autoencoder_loss(output, labels):
    attr_out, struct_out, _ = output
    adj_labels, features_labels, _ = labels

    # Attribute reconstruction loss
    diff_attribute = torch.pow(attr_out - features_labels, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(struct_out - adj_labels, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = 0.5 * attribute_cost + (1 - 0.5) * structure_cost

    return cost


def community_autoencoder_loss(output, labels):
    attr_out, struct_out, comm_out, node_embs, comm_embs, _ = output
    adj_labels, features_labels, comm_labels, _ = labels

    # ==== COMMUNITY RECONSTRUCTION LOSS ====
    diff_structure = torch.pow(comm_out - comm_labels, 2)
    community_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    del diff_structure
    community_cost = torch.mean(community_reconstruction_errors)
    del community_reconstruction_errors

    # ==== KULLBACK-LEIBLER LOSS ====
    num_nodes = adj_labels.shape[0]
    kl_loss = -((0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * comm_embs - torch.pow(node_embs, 2) - torch.pow(torch.exp(comm_embs), 2), dim=1)))

    # ==== RECONSTRUCTION LOSS ====
    # Attribute reconstruction loss
    diff_attribute = torch.pow(attr_out - features_labels, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    del diff_attribute
    attribute_cost = torch.mean(attribute_reconstruction_errors)
    del attribute_reconstruction_errors

    # Structure reconstruction loss
    diff_structure = torch.pow(struct_out - adj_labels, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    del diff_structure
    structure_cost = torch.mean(structure_reconstruction_errors)
    del structure_reconstruction_errors

    rec_loss = 0.5 * attribute_cost + (1 - 0.5) * structure_cost
    del attribute_cost, structure_cost

    # ==== TOTAL LOSS ====
    cost = community_cost + 0.1 * kl_loss + rec_loss

    return cost


def binary_cross_entropy(output, labels, mask=None, out_logits=False, class_weights=None, device='cpu'):
    predictions = output if output.dim() == 1 else output.argmax(dim=1)
    predictions = predictions.type(torch.FloatTensor)

    if out_logits:
        true_classes = torch.tensor(labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        loss = torch.nn.functional.cross_entropy(input=output, target=true_classes, weight=class_weights)
    else:
        true_classes = torch.tensor(labels, dtype=torch.float32)
        weights = None
        if class_weights is not None:
            weights = torch.mul(true_classes, class_weights[1] - class_weights[0])
            weights = torch.add(weights, class_weights[0])
        loss = torch.nn.functional.binary_cross_entropy(input=predictions, target=true_classes, weight=weights)

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)
        loss *= mask

    return torch.mean(loss)


def evolvegcn_loss(logits, labels, args, device):
    """
    logits is a matrix M by C where m is the number of classifications and C are the number of classes
    labels is a integer tensor of size M where each element corresponds to the class that prediction i
    should be matching to
    """
    classes_weights = torch.tensor(args["class_weights"]).to(device)
    labels = labels.view(-1, 1)
    alpha = classes_weights[labels].view(-1, 1)
    loss = alpha * (- logits.gather(-1, labels) + logsumexp(logits))
    return loss.mean()



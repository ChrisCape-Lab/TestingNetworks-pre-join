import torch

from src.testingnetworks._constants import DATA
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.model.encoders._encoder import build_encoder
from src.testingnetworks.model.decoders.decoders import build_attr_decoder, build_community_decoder, build_struct_decoder


# BUILDER
# ----------------------------------------------------

def build_graph_auto_encoder(config: dict, data: GraphDataExtractor, device: str = 'cpu'):
    if config['model'] == 'GraphAutoEncoder':
        return GraphAutoEncoder(model_config=config, data=data, device=device)
    elif config['modelò'] == 'CommunityGraphAutoEncoder':
        return CommunityGraphAutoEncoder(model_config=config, data=data, device=device)
    else:
        raise NotImplementedError


# MODELS
# ----------------------------------------------------


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, model_config: dict, data: GraphDataExtractor, device: str = 'cpu'):
        super(GraphAutoEncoder, self).__init__()
        # Encoder and Decoders
        self.encoder = build_encoder(network_args=model_config['encoder_args'], num_features=data.num_features, num_nodes=data.num_nodes, device=device)
        self.attr_decoder = build_attr_decoder(decoder_args=model_config['attribute_decoder_args'], num_features=data.num_features, device=device)
        self.struct_decoder = build_struct_decoder(decoder_args=model_config['structure_decoder_args'], device=device)

        self.input_list = self.encoder.input_list
        self.output_list = [DATA.ADJACENCY_MATRIX, DATA.NODE_FEATURES]

    def forward(self, sample: dict) -> dict:
        # Encode the input features and the adjacency matrix
        node_embeddings = self.encoder(sample)

        # Reconstruct the inputs (adjacency and features)
        rec_adjacency = self.struct_decoder(sample[DATA.ADJACENCY_MATRIX], node_embeddings)
        rec_features = self.attr_decoder(sample[DATA.ADJACENCY_MATRIX], node_embeddings)
        out_dict = {DATA.ADJACENCY_MATRIX: rec_adjacency, DATA.NODE_FEATURES: rec_features}

        return out_dict

    def get_optimizers(self, optimizer_params: dict) -> list:
        # Choose and initialize optimizers with learning rates
        params = self.encoder.parameters()
        encoder_opt = torch.optim.Adam(params, lr=optimizer_params["encoder_lr"], weight_decay=optimizer_params["encoder_wd"])

        params = self.attr_decoder.parameters()
        attr_decoder_opt = torch.optim.Adam(params, lr=optimizer_params["attr_decoder_lr"], weight_decay=optimizer_params["attr_decoder_wd"])

        params = self.struct_decoder.parameters()
        struct_decoder_opt = torch.optim.Adam(params, lr=optimizer_params["struct_decoder_lr"], weight_decay=optimizer_params["struct_decoder_wd"])

        return [encoder_opt, attr_decoder_opt, struct_decoder_opt]


class CommunityGraphAutoEncoder(torch.nn.Module):
    def __init__(self, model_config: dict, data: GraphDataExtractor, device: str = 'cpu'):
        super(CommunityGraphAutoEncoder, self).__init__()

        self.encoder = build_encoder(network_args=model_config['encoder_args'], num_features=data.num_features, num_nodes=data.num_nodes, device=device)
        self.community_decoder = build_community_decoder(decoder_args=model_config['community_decoder_args'],  num_nodes=data.num_nodes, device=device)
        self.attr_decoder = build_attr_decoder(decoder_args=model_config['attribute_decoder_args'], num_features=data.num_features, device=device)
        self.struct_decoder = build_struct_decoder(decoder_args=model_config['structure_decoder_args'], device=device)

        self.input_list = self.encoder.input_list
        self.output_list = [DATA.ADJACENCY_MATRIX, DATA.MODULARITY_MATRIX, DATA.NODE_FEATURES]

    def forward(self, sample: dict) -> dict:
        node_embs, comm_embs = self.encoder(sample)
        rec_modularity = self.community_decoder(comm_embs)
        rec_features = self.attr_decoder(sample[DATA.ADJACENCY_MATRIX], node_embs)
        rec_adjacency = self.struct_decoder(sample[DATA.ADJACENCY_MATRIX], node_embs)
        out_dict = {DATA.ADJACENCY_MATRIX: rec_adjacency, DATA.MODULARITY_MATRIX: rec_modularity, DATA.NODE_FEATURES: rec_features}

        return out_dict

    def get_optimizers(self, optimizer_params: dict) -> list:
        # Choose and initialize optimizers with learning rates
        params = self.encoder.parameters()
        encoder_opt = torch.optim.Adam(params, lr=optimizer_params["encoder_lr"], weight_decay=optimizer_params["encoder_wd"])

        params = self.community_decoder.parameters()
        comm_decoder_opt = torch.optim.Adam(params, lr=optimizer_params["comm_decoder_lr"], weight_decay=optimizer_params["comm_decoder_wd"])

        params = self.attr_decoder.parameters()
        attr_decoder_opt = torch.optim.Adam(params, lr=optimizer_params["attr_decoder_lr"], weight_decay=optimizer_params["attr_decoder_wd"])

        params = self.struct_decoder.parameters()
        struct_decoder_opt = torch.optim.Adam(params, lr=optimizer_params["struct_decoder_lr"], weight_decay=optimizer_params["struct_decoder_wd"])

        return [encoder_opt, comm_decoder_opt, attr_decoder_opt, struct_decoder_opt]

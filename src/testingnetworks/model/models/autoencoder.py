import torch

from src.testingnetworks.utils import DotDict
from src.testingnetworks.commons.error_measurement.reconstruction_errors import reconstruction_error, community_reconstruction_error


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, config, data, device):
        super(GraphAutoEncoder, self).__init__()
        self.config = DotDict(config)

        self.encoder = _build_encoder(self.config.encoder_args, data, device)
        self.attr_decoder = _build_attr_decoder(self.config.attr_decoder_args, data, device)
        self.struct_decoder = _build_struct_decoder(self.config.struct_decoder_args, data, device)
        self.rec_error = _build_reconstruction_error(self.config.rec_error)

    def forward(self, sample):
        node_embeddings = self.encoder.forward(sample)
        attr = self.attr_decoder.forward(sample.sp_adj_list, node_embeddings)
        struct = self.struct_decoder.forward(sample.sp_adj_list, node_embeddings)
        rec_error = self.rec_error([attr, struct], sample.label_list)
        scores = (rec_error - torch.min(rec_error)) / (torch.max(rec_error) - torch.min(rec_error))

        return [attr, struct, scores]


class CommunityGraphAutoEncoder(torch.nn.Module):
    def __init__(self, config, data, device):
        super(CommunityGraphAutoEncoder, self).__init__()
        self.config = DotDict(config)

        self.encoder = _build_encoder(self.config.encoder_args, data, device)
        self.community_decoder = _build_community_decoder(self.config.community_decoder_args, data, device)
        self.attr_decoder = _build_attr_decoder(self.config.attr_decoder_args, data, device)
        self.struct_decoder = _build_struct_decoder(self.config.struct_decoder_args, data, device)
        self.rec_error = _build_reconstruction_error(self.config.rec_error)

    def forward(self, sample):
        node_embs, comm_embs = self.encoder.forward(sample)
        comm = self.community_decoder.forward(comm_embs)
        attr = self.attr_decoder.forward(sample.sp_adj_list, node_embs)
        struct = self.struct_decoder.forward(sample.sp_adj_list, node_embs)
        rec_error = self.rec_error([attr, struct], sample.label_list)
        scores = (rec_error - torch.min(rec_error)) / (torch.max(rec_error) - torch.min(rec_error))

        return [attr, struct, comm, node_embs, comm_embs, scores]


# BUILDERS
# ----------------------------------------------------

def _build_encoder(encoder_args, dataset, device):
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    encoder_args = DotDict(encoder_args)
    layers_dim = encoder_args.layers_dim
    layers_dim.insert(0, encoder_args.feats_per_node if encoder_args.feats_per_node != "auto" else dataset.num_features)
    layers_dim.append(encoder_args.output_dim)

    # Build the correct Encoder
    if encoder_args.encoder == "GCN":
        from src.testingnetworks.model.encoders.networks import GCN
        return GCN(layers_dim, act=torch.nn.ReLU(), bias=False, device=device)
    elif encoder_args.encoder == "tGCN-1":
        from src.testingnetworks.model.encoders.networks import TGCNe
        return TGCNe(layers_dim, act=torch.nn.ReLU(), bias=False, device=device)
    elif encoder_args.encoder == "tGCN-2":
        from src.testingnetworks.model.encoders.networks import TGCN
        return TGCN(layers_dim, act=torch.nn.ReLU(), bias=False, device=device)
    elif encoder_args.encoder == "Evolve-h":
        from src.testingnetworks.model.encoders.networks import EvolveGCNvH
        return EvolveGCNvH(layers_dim, skipfeats=encoder_args.skipfeats, device=device)
    elif encoder_args.encoder == "EvolveD-h":
        from src.testingnetworks.model.encoders.networks import EvolveGCNvHDouble
        return EvolveGCNvHDouble(layers_dim, short_lenght=encoder_args.short_lenght, skipfeats=network_args.skipfeats, device=device)
    elif encoder_args.encoder == "Evolve-o":
        from src.testingnetworks.model.encoders.networks import EvolveGCNvO
        return EvolveGCNvO(layers_dim, skipfeats=encoder_args.skipfeats, device=device)
    elif encoder_args.encoder == "addGraph":
        from src.testingnetworks.model.encoders.networks import AddGraph
        return AddGraph(layers_dim, nb_window=encoder_args.nb_window, num_nodes=dataset.num_nodes, bias=False, device=device)
    elif encoder_args.encoder == "ComGAEncoder":
        from src.testingnetworks.model.encoders.networks import ComGAEncoder
        return ComGAEncoder(layers_dim, num_nodes=dataset.num_nodes, bias=encoder_args.bias)
    else:
        raise NotImplementedError('The chosen GCN has not been implemented yet')


def _build_community_decoder(decoder_args, dataset, device):
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    decoder_args = DotDict(decoder_args)
    layers_dim = decoder_args.layers_dim
    layers_dim.insert(0, decoder_args.input_dim)
    layers_dim.append(decoder_args.output_dim if decoder_args.output_dim != "auto" else dataset.num_nodes)

    # Build the correct community decoder
    from src.testingnetworks.model.decoders.decoders import DenseDecoder
    return DenseDecoder(layers_dim, act=torch.nn.LeakyReLU(), bias=decoder_args.bias, device=device)


def _build_attr_decoder(decoder_args, dataset, device):
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    decoder_args = DotDict(decoder_args)
    layers_dim = decoder_args.layers_dim
    layers_dim.insert(0, decoder_args.input_dim)
    layers_dim.append(decoder_args.output_dim if decoder_args.output_dim != "auto" else dataset.num_features)

    # Build the correct Encoder
    from src.testingnetworks.model.decoders.decoders import AttributeDecoder
    return AttributeDecoder(layers_dim, act=torch.nn.LeakyReLU(), bias=decoder_args.bias, device=device)


def _build_struct_decoder(decoder_args, dataset, device):
    """Return the correct network to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    # Prepare dimensions
    decoder_args = DotDict(decoder_args)
    layers_dim = decoder_args.layers_dim
    layers_dim.insert(0, decoder_args.input_dim)
    layers_dim.append(decoder_args.output_dim)

    # Build the correct Encoder
    from src.testingnetworks.model.decoders.decoders import StructureDecoder
    return StructureDecoder(layers_dim, act=torch.nn.LeakyReLU(), bias=decoder_args.bias, device=device)


def _build_reconstruction_error(rec_error):
    if rec_error == 'rec_error':
        return reconstruction_error
    elif rec_error == 'community_rec_error':
        return community_reconstruction_error
    else:
        raise NotImplementedError('The chosen reconstruction error has not been implemented yet')

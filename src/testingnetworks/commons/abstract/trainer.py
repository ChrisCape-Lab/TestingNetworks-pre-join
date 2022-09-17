import os
import torch

from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.commons.dataloader.splitter import Splitter
from src.testingnetworks.commons.report_generator import ReportGenerator


class Trainer:
    def __init__(self, training_config: dict, data_extractor: GraphDataExtractor, splitter: Splitter, model: torch.nn.Module, loss, report_gen: ReportGenerator):
        self.training_config = training_config
        self.data = data_extractor
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.model = model
        self.loss = loss
        self.report_gen = report_gen

        self.num_nodes = data_extractor.num_nodes
        self.num_classes = data_extractor.num_classes

    #
    # INITIALIZERS
    # --------------------------------------------
    #
    def _init_optimizers(self, optimizer_params):
        raise NotImplementedError

    #
    # LOAD/SAVE STATE
    # --------------------------------------------
    #
    @staticmethod
    def save_checkpoint(state, path, filename='checkpoint.pth.tar'):
        """Save the network checkpoint in the corresponding file"""
        torch.save(state, os.path.join(path, filename))

    #
    # TRAIN
    # --------------------------------------------
    #
    def train(self):
        raise NotImplementedError

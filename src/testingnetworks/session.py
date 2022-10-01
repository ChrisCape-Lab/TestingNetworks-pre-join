import os

import torch
import torch.distributed as dist
import numpy as np
import random
import yaml

from src.testingnetworks.commons.report_generator import ReportGenerator
from src.testingnetworks.commons.datapreprocess.dataset_preprocess import preprocess_dataset
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor

from utils import DotDict


# UTILS
# ----------------------------------------------------

def _set_seed(seed):
    """Set all the random functions' seeds to make the experiment reproducible"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# BUILDERS
# ----------------------------------------------------

def _build_tasker(model_type, dataloder_args, dataset):
    """Return the correct tasker to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    dataloder_args = DotDict(dataloder_args)
    if model_type == "Classifier":
        from src.testingnetworks.commons.dataloader import ClassifierTasker
        return ClassifierTasker(dataset, task=dataloder_args.task, normalize_adj=dataloder_args.normalize_adj,
                                load_conn_matrix=dataloder_args.load_conn_matrix, load_mod_matrix=dataloder_args.load_mod_matrix)
    elif model_type == "AutoEncoder":
        from src.testingnetworks.commons.dataloader import AutoencoderTasker
        return AutoencoderTasker(dataset, task=dataloder_args.task, normalize_adj=dataloder_args.normalize_adj,
                                 load_conn_matrix=dataloder_args.load_conn_matrix,
                                 load_mod_matrix=dataloder_args.load_mod_matrix)
    else:
        raise NotImplementedError('The chosen tasker has not been implemented yet')


def _build_splitter(dataloader_args, tasker):
    """Return the correct splitter to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    dataloader_args = DotDict(dataloader_args)
    if dataloader_args.is_static:
        from src.testingnetworks.commons.dataloader import StaticSplitter
        return StaticSplitter(tasker, dataloader_args.split_proportions, dataloader_args.data_loading_params)
    else:
        from src.testingnetworks.commons.dataloader import DynamicSplitter
        return DynamicSplitter(tasker, dataloader_args.split_proportions, dataloader_args.time_window, dataloader_args.data_loading_params)


def _build_model(model_config, dataset, device):
    """Return the correct model to build according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    config = DotDict(model_config)

    if config.model == "GCNClassifier":
        from src.testingnetworks.frameworks.gnn_classifier.gcnclassifier import GCNClassifier
        return GCNClassifier(model_config=config, data=dataset, device=device)
    elif config.model == "GAE":
        from src.testingnetworks.frameworks.gae.autoencoder import GraphAutoEncoder
        return GraphAutoEncoder(config=config, data=dataset, device=device)
    elif config.model == "CommunityGAE":
        from src.testingnetworks.frameworks.gae.autoencoder import CommunityGraphAutoEncoder
        return CommunityGraphAutoEncoder(config=config, data=dataset, device=device)
    else:
        raise NotImplementedError('The chosen model has not been implemented yet')


def _build_loss(loss_type):
    """Return the correct loss according to the input (a simple if-elif). If not present raise a NotImplementedError"""
    if loss_type == "cross_entropy":
        from src.testingnetworks.commons.error_measurement.loss import cross_entropy
        return cross_entropy
    elif loss_type == "bce_loss":
        from src.testingnetworks.commons.error_measurement.loss import bce_loss
        return bce_loss
    elif loss_type == "bce_logits_loss":
        from src.testingnetworks.commons.error_measurement.loss import bce_logits_loss
        return bce_logits_loss
    elif loss_type == "autoencoder_loss":
        from src.testingnetworks.commons.error_measurement.loss import autoencoder_loss
        return autoencoder_loss
    elif loss_type == "community_autoencoder_loss":
        from src.testingnetworks.commons.error_measurement.loss import community_autoencoder_loss
        return community_autoencoder_loss
    else:
        raise NotImplementedError('The chosen loss has not been implemented yet')


def _build_trainer(model_type, args, dataset, splitter, model, loss, report_gen):
    if model_type == "Classifier":
        from src.testingnetworks.trainer.trainer import ClassifierTrainer
        return ClassifierTrainer(args, dataset, splitter, model, loss, report_gen)
    elif model_type == "AutoEncoder":
        from src.testingnetworks.trainer.trainer import AutoEncoderTrainer
        return AutoEncoderTrainer(args, dataset, splitter, model, loss, report_gen)
    else:
        raise NotImplementedError('The chosen model has not been implemented yet')


# SESSION
# ----------------------------------------------------

class Session:
    def __init__(self, config_file_path, dataset_file_path):
        # Load network config
        with open(config_file_path, "r") as stream:
            try:
                network_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        self.config_file_path = config_file_path
        self.network_config = DotDict(network_config)

        # Load dataset config
        with open(dataset_file_path, "r") as stream:
            try:
                dataset_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        self.dict_dataset_config = dataset_config
        self.dataset_config = DotDict(dataset_config)

        global rank, wsize, use_cuda
        self.network_config.use_cuda = (torch.cuda.is_available() and self.network_config.use_cuda)
        self.network_config.device = 'cpu'
        if self.network_config.use_cuda:
            self.network_config.device = 'cuda'
        print("\tuse CUDA:", self.network_config.use_cuda, "- device:", self.network_config.device)

        try:
            dist.init_process_group(backend='mpi')  # , world_size=4
            rank = dist.get_rank()
            wsize = dist.get_world_size()
            print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
            if self.network_config.use_cuda:
                torch.cuda.set_device(rank)  # are we sure of the rank+1????
                print('using the device {}'.format(torch.cuda.current_device()))
        except:
            rank = 0
            wsize = 1
            print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))

        # Set all the packages' seed to make the experiment repeatable (same results)
        _set_seed(self.network_config.seed)

    def run(self):
        print("Session start")

        # Create report generator
        dataset_full_name = 'AS_bank_mixed'
        experiment_name = 'GCN_Classifier'
        report_gen = ReportGenerator(experiments_folder_path="../../results", dataset_full_name=dataset_full_name, experiment_full_name=experiment_name,
                                     config_file_pth=self.config_file_path,
                                     loss_measure=self.network_config.trainer_args["loss_args"]["eval_measure"])

        # Preprocess data
        account_processed_df, transactions_processed_df = preprocess_dataset(dataset_folder_path='../../datasets_output/raw/AMLSim/bank_mixed',
                                                                             dataset_full_name='AS_bank_mixed',
                                                                             dataset_type='AS')

        # Build the dataset
        dataset = GraphDataExtractor(pd_node_dataset=account_processed_df, pd_edge_dataset=transactions_processed_df, dataset_config=self.dict_dataset_config['dataset_args'])

        report_gen.load_results_string("Dataset: \t\t" + str(self.dataset_config.data))

        # Build  the tasker
        tasker = _build_tasker(self.network_config.model_type, self.network_config.dataloder_args, dataset)
        report_gen.load_results_string("Tasker: \t\t" + str(self.network_config.dataloder_args["task"]))

        # Build the splitter
        splitter = _build_splitter(self.network_config.dataloder_args, tasker)
        report_gen.load_results_string(
            "Splitter:\t\ttrain: " + str(len(splitter.train)) + ', val: ' + str(len(splitter.val)) + ', test: ' + str(len(splitter.test)))

        # Build the models
        model = _build_model(self.network_config.model_args, dataset, self.network_config.device)
        """
        report_gen.log_network_data(
            "Model: \t\t\t" + str(self.args.model_args["network_args"]["network"]) + " "
            + str(self.args.model_args['classifier_args']['gnn_classifier']) + "\n")
        """
        for n, p in model.named_parameters():
            report_gen.load_results_string("\t" + str(n) + "  " + str(p.shape) + "")

        # Build the loss
        loss = _build_loss(self.network_config.trainer_args["loss_args"]["loss"])
        report_gen.load_results_string("Loss: \t\t\t" + str(self.network_config.trainer_args["loss_args"]["loss"]))

        # Build the Trainer
        trainer = _build_trainer(model_type=self.network_config.model_type,
                                 args=self.network_config.trainer_args,
                                 dataset=dataset,
                                 splitter=splitter,
                                 model=model,
                                 loss=loss,
                                 report_gen=report_gen)
        trainer.train()
        report_gen.generate_report()
        print(" Report generated")

        del report_gen
        del dataset
        del tasker
        del splitter
        del trainer

        print("Session ended")
        return model


class MultiSession:
    def __init__(self, config_folder_path):
        self.config_folder_path = config_folder_path
        self.configs = []
        self.models = []

        f = os.path.join(self.config_folder_path, "AMLSim_prj.yaml")
        with open(f, "r") as stream:
            try:
                args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        self.args = DotDict(args)

        global rank, wsize, use_cuda
        self.args.use_cuda = (torch.cuda.is_available() and self.args.use_cuda)
        self.args.device = 'cpu'
        if self.args.use_cuda:
            self.args.device = 'cuda'
        print("\tuse CUDA:", self.args.use_cuda, "- device:", self.args.device)

        try:
            dist.init_process_group(backend='mpi')  # , world_size=4
            rank = dist.get_rank()
            wsize = dist.get_world_size()
            print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
            if self.args.use_cuda:
                torch.cuda.set_device(rank)  # are we sure of the rank+1????
                print('using the device {}'.format(torch.cuda.current_device()))
        except:
            rank = 0
            wsize = 1
            print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))

        # Set all the packages' seed to make the experiment repeatable (same results)
        _set_seed(self.args.seed)

    def run(self):
        print("MultiSession start")

        for filename in os.listdir(self.config_folder_path):
            if filename == 'AMLSim_prj.yaml':
                continue
            f = os.path.join(self.config_folder_path, filename)
            if os.path.isfile(f):
                print('Configuration: ' + str(f))
                session = Session(str(f))
                self.models.append(session.run())

        # Create report generator
        report_gen = ReportGenerator(exp_path="../../results", exp_name=self.args.name, config_file=None,
                                     loss_measure=self.args.trainer_args["loss_args"]["eval_measure"])

        # Build the dataset
        dataset = _build_dataset(self.args.data, self.args.dataset_args)
        report_gen.load_results_string("Dataset: \t\t" + str(self.args.data))

        # Build  the tasker
        tasker = _build_tasker(self.args.model_type, self.args.dataloder_args, dataset)
        report_gen.load_results_string("Tasker: \t\t" + str(self.args.dataloder_args["task"]))

        # Build the splitter
        splitter = _build_splitter(self.args.dataloder_args, tasker)
        report_gen.load_results_string(
            "Splitter:\t\ttrain: " + str(len(splitter.train)) + ', val: ' + str(len(splitter.val)) + ', test: ' + str(len(splitter.test)))

        # Build the models
        from src.testingnetworks.model.models.tribrigade import TriBrigade
        from src.testingnetworks.commons.error_measurement.loss import reconstruction_error
        model = TriBrigade(self.args.model_args, self.models[1], self.models[0], reconstruction_error, dataset, self.args.device)
        for n, p in model.named_parameters():
            report_gen.load_results_string("\t" + str(n) + "  " + str(p.shape) + "")

        # Build the loss
        loss = _build_loss(self.args.trainer_args["loss_args"]["loss"])
        report_gen.load_results_string("Loss: \t\t\t" + str(self.args.trainer_args["loss_args"]["loss"]))

        # Build the Trainer
        trainer = _build_trainer(model_type=self.args.model_type,
                                 args=self.args.trainer_args,
                                 dataset=dataset,
                                 splitter=splitter,
                                 model=model,
                                 loss=loss,
                                 report_gen=report_gen)
        trainer.train()
        if self.args.model_type == "GCNClassifier":
            report_gen.generate_report()
            print(" Report generated")

        print("MultiSession ended")

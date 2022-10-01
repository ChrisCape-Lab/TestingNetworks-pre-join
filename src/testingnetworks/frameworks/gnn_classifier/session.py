import torch
import numpy as np
import random

from src.testingnetworks._constants import DATA
from src.testingnetworks.commons.abstract.session import Session
from src.testingnetworks.commons.report_generator import ReportGenerator
from src.testingnetworks.commons.datapreprocess.dataset_preprocess import preprocess_dataset
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.frameworks.gnn_classifier.gcnclassifier import GCNClassifier
from src.testingnetworks.commons.dataloader.splitter import Splitter
from src.testingnetworks.commons.dataloader.tasker import Tasker
from src.testingnetworks.commons.error_measurement.loss import build_loss
from src.testingnetworks.frameworks.gnn_classifier.trainer import ClassifierTrainer

from src.testingnetworks.utils import DotDict


# UTILS
# ----------------------------------------------------

def _set_seed(seed: int):
    """Set all the random functions' seeds to make the experiment reproducible"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# SESSION
# ----------------------------------------------------

class GCNClassifierSession(Session):
    def __init__(self, dataset_folder_path: str, experiment_config_file_path: str, results_folder_path: str):
        super().__init__(dataset_folder_path=dataset_folder_path, experiment_config_file_path=experiment_config_file_path, results_folder_path=results_folder_path)

    def run(self):
        # Create report generator
        dataset_full_name = self.dataset_type + '_' + self.dataset_name
        experiment_name = self.experiment_config['model'] + '_' + self.experiment_config['network']
        report_gen = ReportGenerator(experiments_folder_path=self.results_folder_path, dataset_full_name=dataset_full_name,
                                     experiment_full_name=experiment_name,
                                     config_file_pth=self.experiment_config_file_path,
                                     loss_measure=self.experiment_config['training_parameters']["eval_measure"])

        # Preprocess data
        account_processed_df, transactions_processed_df = preprocess_dataset(dataset_folder_path=self.dataset_folder_path,
                                                                             dataset_full_name=dataset_full_name,
                                                                             dataset_type=self.dataset_type)

        # Build the dataset
        data_extractor = GraphDataExtractor(pd_node_dataset=account_processed_df, pd_edge_dataset=transactions_processed_df,
                                            dataset_config=self.dataloading_config)

        # Build the models
        model_name = str(self.experiment_config['network']) + '_' + str(self.experiment_config['model'])
        model = GCNClassifier(model_config=DotDict(self.experiment_config['model_args']), data=data_extractor, device=self.experiment_config['device'])
        print('Model ' + model_name + ' built')

        # Build  the tasker
        tasker = Tasker(data_extractor=data_extractor, dataloading_config=self.dataloading_config,
                        inputs_list=model.input_list, outputs_list=[DATA.NODE_LABELS])

        # Build the splitter
        split_parameters = self.experiment_config['training_parameters']['split_parameters']
        splitter = Splitter(tasker=tasker, split_proportions=split_parameters['split_proportions'],
                            data_loading_params=split_parameters)

        # Build the loss
        loss = build_loss(loss_type=self.experiment_config['training_parameters']["loss"])

        # Log configuration data
        task = str(self.dataloading_config["task"])
        split_proportions = [len(splitter.train), len(splitter.val), len(splitter.test)]
        report_gen.log_config_data(dataset_name=dataset_full_name, model_name=model_name, task=task, split_proportions=split_proportions,
                                   loss_name=str(self.experiment_config['training_parameters']["loss"]))
        report_gen.log_model_data(name=model_name, parameters=model.named_parameters())

        # Build the Trainer
        trainer = ClassifierTrainer(training_config=self.experiment_config['training_parameters'],
                                    data_extractor=data_extractor,
                                    splitter=splitter,
                                    model=model,
                                    loss=loss,
                                    report_gen=report_gen)
        trainer.train()

        report_gen.generate_report()
        print("Report generated")

        del report_gen
        del data_extractor
        del tasker
        del splitter
        del trainer

        print("Session ended")
        return model

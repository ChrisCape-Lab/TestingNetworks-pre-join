import os.path

import torch

from datetime import datetime

from src.testingnetworks._constants import DATA
from src.testingnetworks.commons.abstract.trainer import Trainer
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.commons.dataloader.splitter import Splitter
from src.testingnetworks.frameworks.gnn_classifier.gcnclassifier import GCNClassifier
from src.testingnetworks.commons.report_generator import ReportGenerator
from src.testingnetworks.commons.evaluator import Evaluator

from src.testingnetworks._constants import WELL_KNOWN_FOLDERS as FOLDERS


class ClassifierTrainer(Trainer):
    def __init__(self, training_config: dict, data_extractor: GraphDataExtractor, splitter: Splitter, model: GCNClassifier, loss, report_gen: ReportGenerator):
        super().__init__(training_config=training_config, data_extractor=data_extractor, splitter=splitter, model=model, loss=loss, report_gen=report_gen)

        self._init_optimizers(self.training_config['optimizer_parameters'])
        self.class_weights = self.training_config['class_weights'] if self.training_config['class_weights'] != 'auto' else self.tasker.class_weights
        self.tr_step = 0

    #
    # INITIALIZERS
    # --------------------------------------------
    #
    def _init_optimizers(self, optimizer_params: dict):
        """
        This function initialize the optimizers. Needs to be called before the train start to initialize it (call this during the __init__, just need
        to be called once per train...)
        :param optimizer_params: DotDict, contains the learning rates arguments
        :return: -
        """
        # Choose and initialize optimizers with learning rates
        params = self.model.network.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=optimizer_params["network_lr"], weight_decay=optimizer_params["network_wd"])

        params = self.model.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr=optimizer_params["classifier_lr"], weight_decay=optimizer_params["classifier_wd"])

        # Initialize (zeroes) the gradient of the optimizers
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    #
    # TRAIN
    # --------------------------------------------
    #
    def train(self):
        # Initialize variables
        self.tr_step = 0
        eval_measure = torch.zeros(1)
        eval_measures = None
        best_eval_valid = 0
        best_loss = 1
        epochs_without_impr = 0

        # Load training parameters once
        num_epochs = self.training_config['epochs_args']["num_epochs"]
        eval_after_epochs = self.training_config['epochs_args']["eval_after_epochs"]
        eval_measure_str = self.training_config["eval_measure"]
        patience = self.training_config['epochs_args']["early_stop_patience"]

        # Train/validate
        print("Training start")
        start = datetime.now()
        for e in range(0, num_epochs):
            e_start = datetime.now()
            train_loss, _ = self.run_train_epoch(self.splitter.train)
            if len(self.splitter.val) > 0 and e >= eval_after_epochs:
                eval_measures, _ = self.run_eval_epoch(self.splitter.val)
                eval_measure = eval_measures[eval_measure_str]
                if eval_measure > best_eval_valid or (eval_measure == best_eval_valid and eval_measures['Loss'] < best_loss):
                    self.save_checkpoint(self.model.state_dict(), path=FOLDERS.CACHE)
                    best_eval_valid = eval_measure
                    best_loss = eval_measures['Loss']
                    epochs_without_impr = 0
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > patience:
                        step_output = self.report_gen.log_training_data(curr_epoch=e, tot_epoch=num_epochs, time='', train_loss=train_loss.item(),
                                                                        valid_loss=eval_measures['Loss'],
                                                                        eval_measure_name=self.training_config['eval_measure'],
                                                                        eval_measures=eval_measures,
                                                                        best_eval_measure=best_eval_valid, patience=(epochs_without_impr, patience),
                                                                        early_stop=True)
                        print(step_output)
                        break
            time = str(datetime.now() - e_start)
            if eval_measures is not None:
                step_output = self.report_gen.log_training_data(curr_epoch=e, tot_epoch=num_epochs, time=time, train_loss=train_loss.item(),
                                                                valid_loss=eval_measures['Loss'], eval_measure_name=self.training_config['eval_measure'],
                                                                eval_measures=eval_measures,
                                                                best_eval_measure=best_eval_valid, patience=(epochs_without_impr, patience))
                print(step_output)

        # Test
        self.model.load_state_dict(torch.load(os.path.join(FOLDERS.CACHE, "checkpoint.pth.tar")))
        self.report_gen.log_network_parameters(ckpt_path=os.path.join(FOLDERS.CACHE, "checkpoint.pth.tar"))
        test_loss, _ = self.run_eval_epoch(self.splitter.test)
        step_output = self.report_gen.log_training_results(time=str(datetime.now() - start), validation_measure=self.training_config['eval_measure'],
                                                           test_measures=test_loss)
        print(step_output)
        print(" Training ended")

    def run_train_epoch(self, split):
        tot_loss = torch.zeros(1)
        torch.set_grad_enabled(True)

        self.model.train()
        for sample in split:
            out = self.model(sample[0])
            labels = sample[0][DATA.LABELS]

            loss = self.loss(out, labels, class_weights=self.class_weights)
            self.optim_step(loss)
            tot_loss += loss

        return tot_loss / len(split), self.model.node_embeddings

    def run_eval_epoch(self, split):
        torch.set_grad_enabled(False)
        evaluator = Evaluator(self.data.num_classes)

        self.model.eval()
        for sample in split:
            out = self.model(sample[0])
            labels = sample[0][DATA.LABELS]

            loss = self.loss(out, labels, class_weights=self.class_weights)
            evaluator.save_step_results(out, labels, loss.detach())

        eval_measures = evaluator.evaluate()

        return eval_measures, self.model.node_embeddings

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.training_config['epochs_args']["steps_accum_gradients"] == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.9)
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()


def _print_step_result(epoch, num_epochs, time, train_loss, eval_loss, eval_measure_name, eval_measure, best_eval_valid, epochs_without_impr, patience):
    if eval_measure_name is not None:
        step_output = str('  - Epoch ' + str(epoch) + '/' + str(num_epochs) + ' - ' + time[:-7] +
                          ' :   train loss: ' + str(round(train_loss, 3)) +
                          '     valid loss: ' + str(round(eval_loss, 3)) +
                          '     ' + str(eval_measure_name) + ': ' + str(round(eval_measure, 3)) +
                          '   | Best: ' + str(round(best_eval_valid, 3)) + '.  patience: ' + str(epochs_without_impr) + '/' + str(patience))
    else:
        step_output = str('  - Epoch ' + str(epoch) + '/' + str(num_epochs) + ' - ' + time[:-7] +
                          ' :   train loss: ' + str(round(train_loss, 3)) +
                          '     valid loss: ' + str(round(eval_loss, 3)) +
                          '   | Best: ' + str(round(best_eval_valid, 3)) + '.  patience: ' + str(epochs_without_impr) + '/' + str(patience))
    return step_output

import os.path

import torch

from datetime import datetime

from src.testingnetworks._constants import DATA
from src.testingnetworks.commons.abstract.trainer import Trainer
from src.testingnetworks.commons.dataloader.graph_data_extractor import GraphDataExtractor
from src.testingnetworks.commons.dataloader.splitter import Splitter
from src.testingnetworks.commons.report_generator import ReportGenerator
from src.testingnetworks.commons.evaluator import Evaluator

from src.testingnetworks._constants import WELL_KNOWN_FOLDERS as FOLDERS


class GraphAutoEncoderTrainer(Trainer):
    def __init__(self, training_config: dict, data_extractor: GraphDataExtractor, splitter: Splitter, model, rec_error, report_gen: ReportGenerator):
        super().__init__(training_config=training_config, data_extractor=data_extractor, splitter=splitter, model=model, loss=rec_error, report_gen=report_gen)

        self._init_optimizers(self.training_config['optimizer_parameters'])
        self.class_weights = self.training_config['class_weights'] if self.training_config['class_weights'] != 'auto' else self.tasker.class_weights
        self.tr_step = 0

    #
    # INITIALIZERS
    # --------------------------------------------
    #
    def _init_optimizers(self, optimizer_params: dict) -> None:
        """
        This function initialize the optimizers. Needs to be called before the train start to initialize it (call this during the __init__, just need
        to be called once per train...)
        :param optimizer_params: DotDict, contains the learning rates arguments
        :return: -
        """
        # Choose and initialize optimizers with learning rates
        self.optimizers = self.model.get_optimizers(optimizer_params)

        # Initialize (zeroes) the gradient of the optimizers
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    #
    # TRAIN
    # --------------------------------------------
    #
    def train(self):
        # Initialize variables
        self.tr_step = 0
        best_eval_valid = 0
        best_loss = 1
        epochs_without_impr = 0

        # Load training parameters once
        num_epochs = self.training_config['epochs_args']["num_epochs"]
        eval_measure_str = self.training_config["eval_measure"]
        patience = self.training_config['epochs_args']["early_stop_patience"]

        # Train/validate
        print("Training start")
        start = datetime.now()
        for e in range(0, num_epochs):
            e_start = datetime.now()
            eval_measures = self.run_train_epoch(self.splitter.train)
            eval_measure = eval_measures[eval_measure_str]
            if eval_measure > best_eval_valid or (eval_measure == best_eval_valid and eval_measures['Loss'] < best_loss):
                self.save_checkpoint(self.model.state_dict(), path=FOLDERS.CACHE)
                best_eval_valid = eval_measure
                best_loss = eval_measures['Loss']
                epochs_without_impr = 0
            else:
                epochs_without_impr += 1
                if epochs_without_impr > patience:
                    step_output = self.report_gen.log_training_data(curr_epoch=e, tot_epoch=num_epochs, time='', train_loss=eval_measures['Loss'],
                                                                    valid_loss=None,
                                                                    eval_measure_name=self.training_config['eval_measure'],
                                                                    eval_measures=eval_measures,
                                                                    best_eval_measure=best_eval_valid, patience=(epochs_without_impr, patience),
                                                                    early_stop=True)
                    print(step_output)
                    break
            time = str(datetime.now() - e_start)
            if eval_measures is not None:
                step_output = self.report_gen.log_training_data(curr_epoch=e, tot_epoch=num_epochs, time=time, train_loss=eval_measures['Loss'],
                                                                valid_loss=None, eval_measure_name=self.training_config['eval_measure'],
                                                                eval_measures=eval_measures,
                                                                best_eval_measure=best_eval_valid, patience=(epochs_without_impr, patience))
                print(step_output)

        # Test
        self.model.load_state_dict(torch.load(os.path.join(FOLDERS.CACHE, "checkpoint.pth.tar")))
        self.report_gen.log_network_parameters(ckpt_path=os.path.join(FOLDERS.CACHE, "checkpoint.pth.tar"))
        test_loss = self.run_eval_epoch(self.splitter.test)
        step_output = self.report_gen.log_training_results(time=str(datetime.now() - start), validation_measure=self.training_config['eval_measure'],
                                                           test_measures=test_loss)
        print(step_output)
        print("Training ended")

    def run_train_epoch(self, split):
        torch.set_grad_enabled(True)

        evaluator = Evaluator(self.data.num_classes)

        self.model.train()
        for sample in split:
            model_input = sample[0]
            labels = model_input[DATA.LABELS]
            model_input[DATA.LABELS] = None

            out_dict = self.model(model_input)

            rec_error, cost = self.loss(out_dict, labels)
            scores = (rec_error - torch.min(rec_error)) / (torch.max(rec_error) - torch.min(rec_error))

            self.optim_step(cost)

            #e_scores = torch.nan_to_num(scores.detach(), nan=0, neginf=0, posinf=1)
            #e_cost = torch.nan_to_num(cost.detach(), nan=0, neginf=0, posinf=1)
            evaluator.save_step_results(scores.detach(), labels[DATA.LABELS], cost.detach())

        eval_measures = evaluator.evaluate()

        return eval_measures

    def run_eval_epoch(self, split):
        torch.set_grad_enabled(False)

        evaluator = Evaluator(self.data.num_classes)

        self.model.eval()
        for sample in split:
            model_input = sample[0]
            labels = model_input[DATA.LABELS]
            model_input[DATA.LABELS] = None

            out_dict = self.model(sample[0])

            rec_error, cost = self.loss(out_dict, labels)
            scores = (rec_error - torch.min(rec_error)) / (torch.max(rec_error) - torch.min(rec_error))

            #e_scores = torch.nan_to_num(scores.detach(), nan=0, neginf=0, posinf=1)
            #e_cost = torch.nan_to_num(cost.detach(), nan=0, neginf=0, posinf=1)
            evaluator.save_step_results(scores.detach(), labels[DATA.LABELS], cost.detach())

        eval_measures = evaluator.evaluate()

        return eval_measures

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.training_config['epochs_args']["steps_accum_gradients"] == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.9)

            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()


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

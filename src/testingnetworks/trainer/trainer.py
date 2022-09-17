import torch

from datetime import datetime

from src.testingnetworks.commons.evaluator import Evaluator
from src.testingnetworks.commons.dataloader import Sample
from src.testingnetworks.utils import DotDict


class Trainer:
    def __init__(self, args: dict, dataset, splitter, model, loss, report_gen):
        self.args = DotDict(args)
        self.data = dataset
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.model = model
        self.loss = loss
        self.report_gen = report_gen

        self.num_nodes = dataset.num_nodes
        self.num_classes = dataset.num_classes

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
    def save_checkpoint(state, filename='../checkpoint.pth.tar'):
        """Save the network checkpoint in the corresponding file"""
        torch.save(state, filename)

    #
    # TRAIN
    # --------------------------------------------
    #
    def train(self):
        raise NotImplementedError


class ClassifierTrainer(Trainer):
    def __init__(self, args, dataset, splitter, model, loss, report_gen):
        super().__init__(args, dataset, splitter, model, loss, report_gen)

        self._init_optimizers(self.args.optimizer_args)
        self.tr_step = 0

    #
    # INITIALIZERS
    # --------------------------------------------
    #
    def _init_optimizers(self, optimizer_params):
        """
        This function initialize the optimizers. Needs to be called before the train start to initialize it (call this during the __init__, just need
        to be called once per train...)
        :param optimizer_params: DotDict, contains the learning rates arguments
        :return: -
        """
        # Choose and initialize optimizers with learning rates
        params = self.model.gcn.parameters()
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
        self.tr_step = 0
        num_epochs = self.args.epochs_args["num_epochs"]
        eval_measure = torch.zeros(1)
        eval_measures = None
        best_eval_valid = 0
        best_loss = 1
        epochs_without_impr = 0
        patience = self.args.epochs_args["early_stop_patience"]

        print(" Network training start")
        start = datetime.now()
        for e in range(0, num_epochs):
            e_start = datetime.now()
            train_loss, _ = self.run_train_epoch(self.splitter.train)
            if len(self.splitter.val) > 0 and e >= self.args.epochs_args["eval_after_epochs"]:
                eval_measures, _ = self.run_eval_epoch(self.splitter.val)
                eval_measure = eval_measures[self.args.loss_args["eval_measure"]]
                if eval_measure > best_eval_valid or (eval_measure == best_eval_valid and eval_measures['Loss'] < best_loss):
                    self.save_checkpoint(self.model.state_dict())
                    best_eval_valid = eval_measure
                    best_loss = eval_measures['Loss']
                    epochs_without_impr = 0
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > patience:
                        step_output = str('   - Epoch ' + str(e) + '/' + str(num_epochs) + ': Early stop   | Best: ' + str(best_eval_valid))
                        self.report_gen.log_network_step(step_output, train_loss.detach().item(), eval_measures)
                        break
            time = str(datetime.now() - e_start)
            if eval_measures is not None:
                step_output = _print_step_result(e, num_epochs, time, train_loss.item(), eval_measures['Loss'], self.args.loss_args["eval_measure"],
                                                 eval_measure, best_eval_valid, epochs_without_impr, patience)
                self.report_gen.log_network_step(step_output, train_loss.detach().item(), eval_measures)
                print(step_output)

        self.model.load_state_dict(torch.load("../checkpoint.pth.tar"))
        test_loss, _ = self.run_eval_epoch(self.splitter.test)
        step_output = str('Time: ' + str(datetime.now() - start) + "\n" +
                          'Test loss: ' + str(test_loss["Loss"]) + "   validation measure: " + str(test_loss[self.args.loss_args["eval_measure"]]))
        self.report_gen.log_network_results(step_output, test_loss)
        print(step_output)
        print(" Training ended")

    def run_train_epoch(self, split):
        tot_loss = torch.zeros(1)
        torch.set_grad_enabled(True)

        class_weights = self.args.loss_args['class_weights']
        #class_weights = class_weights if class_weights != 'auto' else self.tasker.class_weights
        class_weights = [self.tasker.class_weights[0], self.tasker.class_weights[1]*class_weights]

        self.model.train()
        for s in split:
            sample = Sample()
            sample.from_dataloader_dict(s[0])

            out = self.model.forward(sample)
            labels = sample.label_list

            loss = self.loss(out, labels, class_weights=class_weights)
            self.optim_step(loss)
            tot_loss += loss

        return tot_loss / len(split), self.model.node_embeddings

    def run_eval_epoch(self, split):
        torch.set_grad_enabled(False)
        evaluator = Evaluator(self.data.num_classes)

        class_weights = self.args.loss_args['class_weights']
        #class_weights = class_weights if class_weights != 'auto' else self.tasker.class_weights
        class_weights = [self.tasker.class_weights[0], self.tasker.class_weights[1] * class_weights]

        self.model.eval()
        for s in split:
            sample = Sample()
            sample.from_dataloader_dict(s[0])

            out = self.model.forward(sample)
            labels = sample.label_list

            loss = self.loss(out, labels, class_weights=class_weights)
            evaluator.save_step_results(out, labels, loss.detach())

        eval_measures = evaluator.evaluate()

        return eval_measures, self.model.node_embeddings

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.epochs_args["steps_accum_gradients"] == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.9)
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()


class AutoEncoderTrainer(Trainer):
    def __init__(self, args, dataset, splitter, model, loss, report_gen):
        super().__init__(args, dataset, splitter, model, loss, report_gen)

        self._init_optimizers(self.args.optimizer_args)
        self.tr_step = 0

    #
    # INITIALIZERS
    # --------------------------------------------
    #
    def _init_optimizers(self, optimizer_params):
        """
        This function initialize the optimizers. Needs to be called before the train start to initialize it (call this during the __init__, just need
        to be called once per train...)
        :param optimizer_params: DotDict, contains the learning rates arguments
        :return: -
        """
        # Choose and initialize optimizers with learning rates
        params = self.model.parameters()
        self.opt = torch.optim.Adam(params, lr=optimizer_params["model_lr"], weight_decay=optimizer_params["model_wd"])

        # Initialize (zeroes) the gradient of the optimizers
        self.opt.zero_grad()

    #
    # TRAIN
    # --------------------------------------------
    #
    def train(self):
        self.tr_step = 0
        num_epochs = self.args.epochs_args["num_epochs"]
        best_loss = 1000
        eval_loss = 1000
        epochs_without_impr = 0
        patience = self.args.epochs_args["early_stop_patience"]

        print(" Network training start")
        start = datetime.now()
        for e in range(0, num_epochs):
            e_start = datetime.now()
            train_loss = self.run_train_epoch(self.splitter.train)
            if len(self.splitter.val) > 0 and e >= self.args.epochs_args["eval_after_epochs"]:
                eval_loss = self.run_eval_epoch(self.splitter.val)
                if eval_loss < best_loss:
                    self.save_checkpoint(self.model.state_dict())
                    best_loss = eval_loss
                    epochs_without_impr = 0
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > patience:
                        step_output = str('   - Epoch ' + str(e) + '/' + str(num_epochs) + ': Early stop   | Best: ' + str(best_loss.item()))
                        break
            time = str(datetime.now() - e_start)
            step_output = self.report_gen.log_network_step()
            step_output = _print_step_result(e, num_epochs, time, train_loss.item(), eval_loss.item(), None, None, best_loss.item(), epochs_without_impr, patience)
            print(step_output)

        self.model.load_state_dict(torch.load("../checkpoint.pth.tar"))
        test_measure = self.run_test(self.splitter.test)
        step_output = str('Time: ' + str(datetime.now() - start) + "\n" + 'Test loss: ' + str(test_measure["Loss"]))
        self.report_gen.log_network_step(step_output, test_measure['Loss'], test_measure.copy())
        self.report_gen.log_network_results(step_output, test_measure.copy())
        print(" Training ended")

    def run_train_epoch(self, split):
        tot_loss = torch.zeros(1)
        torch.set_grad_enabled(True)

        self.model.train()
        for s in split:
            sample = Sample()
            sample.from_dataloader_dict(s[0])

            out = self.model.forward(sample)
            labels = sample.label_list

            loss = self.loss(out, labels)
            self.optim_step(loss)
            tot_loss += loss
            del sample

        return tot_loss / len(split)

    def run_eval_epoch(self, split):
        tot_loss = torch.zeros(1)
        torch.set_grad_enabled(False)

        self.model.eval()
        for s in split:
            sample = Sample()
            sample.from_dataloader_dict(s[0])

            out = self.model.forward(sample)
            labels = sample.label_list

            loss = self.loss(out, labels)
            tot_loss += loss
            del sample

        return tot_loss / len(split)

    def run_test(self, split):
        torch.set_grad_enabled(False)
        evaluator = Evaluator(self.data.num_classes)

        class_weights = self.args.loss_args['class_weights']
        class_weights = class_weights if class_weights != 'auto' else self.tasker.class_weights

        self.model.eval()
        from src.testingnetworks.commons.error_measurement.loss import bce_loss
        for s in split:
            sample = Sample()
            sample.from_dataloader_dict(s[0])

            out = self.model.forward(sample)
            out = out[-1]
            labels = sample.label_list[-1]

            loss = bce_loss(out, labels, class_weights=class_weights)
            evaluator.save_step_results(out, labels, loss.detach())

        eval_measures = evaluator.evaluate()

        return eval_measures

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.epochs_args["steps_accum_gradients"] == 0:
            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 1)
            self.opt.step()
            self.opt.zero_grad()


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

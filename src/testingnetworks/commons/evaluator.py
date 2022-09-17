import torch
import numpy as np

from sklearn.metrics import confusion_matrix, average_precision_score
from scipy.sparse import coo_matrix

np.seterr(divide='ignore', invalid='ignore')


class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.batch_sizes = []
        self.losses = []
        self.errors = []
        self.MRRs = []
        self.MAPs = []
        self.confusion_matrix = np.array(0)
        self.confusion_matrix_list = []

    def save_step_results(self, output, labels, loss, **kwargs):
        # Need to get the predictions. If output is logits calculate the index with the max value, otherwise round the probability (binary cls)
        predictions = torch.round(output) if output.dim() == 1 else output.argmax(dim=1)
        true_classes = torch.tensor(labels, dtype=torch.float32)
        cf_matrix = self.get_confusion_matrix(predictions, labels, [i for i in range(0, self.num_classes)])

        MRR = torch.tensor([0.0])

        MAP = torch.tensor(self.get_MAP(predictions, true_classes))

        # Calculate error and confusion matrix
        failures = 0
        for i in range(len(predictions)):
            failures += 1 if predictions[i] != labels[i] else 0
        error = failures / predictions.size(0)

        # Save all th step datasets_output
        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)
        self.losses.append(loss)
        self.errors.append(error)
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.confusion_matrix = np.add(self.confusion_matrix, cf_matrix)
        self.confusion_matrix_list.append(cf_matrix)

    def evaluate(self):
        eval_measure = {}

        loss = 0
        for i in range(0, len(self.losses)):
            loss += self.losses[i].item()
        eval_measure['Loss'] = loss / len(self.losses)

        eval_measure['Error'] = 0
        for i in range(0, len(self.errors)):
            eval_measure['Error'] += self.errors[i]
        eval_measure['Error'] /= len(self.errors)

        accuracy = 0
        MRR = 0
        MAP = 0
        precision = 0
        recall = 0
        bACC = 0
        f1 = 0
        for conf_matrix in self.confusion_matrix_list:
            tn, fp, fn, tp = conf_matrix.ravel()
            accuracy += float(tp + tn) / (tn + fp + fn + tp)
            MRR += self._calc_epoch_metric(self.batch_sizes, self.MRRs)
            MAP += self._calc_epoch_metric(self.batch_sizes, self.MAPs)

            p, r, f = self._calc_microavg_eval_measures(tp, fn, fp)
            precision += p
            recall += r
            f1 += f
            tnr = float(tn) / (tn + fp)
            bACC += (r + tnr) / 2

        eval_measure["Accuracy"] = accuracy / len(self.confusion_matrix_list)
        eval_measure['MRR'] = MRR / len(self.confusion_matrix_list)
        eval_measure['MAP'] = MAP / len(self.confusion_matrix_list)
        eval_measure["Precision"] = precision / len(self.confusion_matrix_list)
        eval_measure["Recall"] = recall / len(self.confusion_matrix_list)
        eval_measure["bACC"] = bACC / len(self.confusion_matrix_list)
        eval_measure["F1"] = f1 / len(self.confusion_matrix_list)

        tn, fp, fn, tp = self.confusion_matrix.ravel()
        eval_measure["_cf_matrix_str"] = str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp)

        """
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        eval_measure["Accuracy"] = float(tp + tn) / (tn + fp + fn + tp)

        epoch_MRR = self._calc_epoch_metric(self.batch_sizes, self.MRRs)
        eval_measure['MRR'] = epoch_MRR

        epoch_MAP = self._calc_epoch_metric(self.batch_sizes, self.MAPs)
        eval_measure['MAP'] = epoch_MAP

        precision, recall, f1 = self._calc_microavg_eval_measures(tp, fn, fp)
        eval_measure["Precision"] = precision
        eval_measure["Recall"] = recall
        true_negative_rate = float(tn) / (tn + fp)
        eval_measure["bACC"] = (recall + true_negative_rate) / 2
        eval_measure["F1"] = f1

        eval_measure["_cf_matrix_str"] = str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp)
        """

        return eval_measure

    def get_confusion_matrix(self, predictions, true_classes, labels):
        y_pred = predictions
        y_true = true_classes

        return confusion_matrix(y_true, y_pred, labels=labels)

    def get_MAP(self, predictions, true_classes):
        predictions_np = predictions.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return average_precision_score(true_classes_np, predictions_np)

    def get_MRR(self, predictions, true_classes, adj):
        probs = predictions.detach().cpu().numpy()
        true_classes = true_classes.detach().cpu().numpy()
        adj = adj.cpu().numpy()

        pred_matrix = coo_matrix((probs, (adj[0], adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes, (adj[0], adj[1]))).toarray()

        row_MRRs = []
        for i, pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
            if np.isin(1, true_matrix[i]):
                row_MRRs.append(self._get_row_MRR(pred_row, true_matrix[i]))

        avg_MRR = torch.tensor(row_MRRs).mean()
        return avg_MRR

    def _get_row_MRR(self, probs, true_classes):
        existing_mask = true_classes == 1
        #descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]

        existing_ranks = np.arange(1, true_classes.shape[0]+1, dtype=np.float)[ordered_existing_mask]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]

        return MRR

    def _calc_epoch_metric(self, batch_sizes, metric_val):
        batch_sizes = torch.tensor(batch_sizes, dtype=torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum()/batch_sizes.sum()

        return epoch_metric_val.detach().item()

    def _calc_microavg_eval_measures(self, tp, fn, fp):
        if (tp + fp) != 0 and tp != 0:
            p = float(tp / (tp + fp))
            r = float(tp / (tp + fn))
            f1 = 2.0 * (p * r) / (p + r)
        else:
            return 0, 0, 0

        return p, r, f1

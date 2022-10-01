import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from matplotlib.backends.backend_pdf import PdfPages


CONFIG_LINE = "\n+=========================+\n|  CONFIGURATION          |\n+=========================+\n\n"
MODEL_LINE = "\n\n\n+=========================+\n|  MODEL                  |\n+=========================+\n\n"
TRAIN_LINE = "\n\n\n+=========================+\n|  TRAIN                  |\n+=========================+\n\n"
RESULTS_LINE = "\n\n\n+=========================+\n|  RESULTS                |\n+=========================+\n\n"


class ReportGenerator:
    def __init__(self, experiments_folder_path: str, dataset_full_name: str, experiment_full_name: str, config_file_pth: str, loss_measure: str):
        self.experiments_folder_path = experiments_folder_path
        self.config_file_pth = config_file_pth
        self.loss_measure = loss_measure

        # Create, if not exists, the experiments' folder for the selected dataset
        self.dataset_experiment_folder_path = os.path.join(self.experiments_folder_path, dataset_full_name)
        if not os.path.isdir(self.dataset_experiment_folder_path):
            try:
                os.mkdir(self.dataset_experiment_folder_path)
            except OSError:
                print("Creation of the directory %s failed" % self.dataset_experiment_folder_path)
                exit()

        # Create the sub-folder for the current experiment
        self.experiment_folder_path = ""
        self.exp_num = 0
        for i in range(0, 100):
            experiment_name = experiment_full_name + '_{:03d}'.format(i)
            self.experiment_folder_path = os.path.join(self.dataset_experiment_folder_path, experiment_name)
            if not os.path.isdir(self.experiment_folder_path):
                try:
                    os.mkdir(self.experiment_folder_path)
                    self.exp_num = i
                except OSError:
                    print("Creation of the directory %s failed" % self.experiment_folder_path)
                    exit()
                break

        # Move the configuration file
        if self.config_file_pth is not None:
            shutil.copy(self.config_file_pth, os.path.join(self.experiment_folder_path, experiment_full_name + '.yaml'))

        # Create the results.txt file
        self.results_file_path = os.path.join(self.experiment_folder_path, "Results.txt")
        open(self.results_file_path, 'w', encoding='utf-8')
        self.is_training_first_log = True

        # Dataframe for .xlsx file and report
        self.results_df = None

    def get_curr_exp_folder(self):
        return self.experiment_folder_path

    # RESULTS FILE
    # ----------------------------------------------------

    def log_config_data(self, dataset_name: str, model_name: str, task: str, split_proportions: list, loss_name: str) -> None:
        with open(self.results_file_path, mode='a', encoding='utf-8') as results_file:
            results_file.write(CONFIG_LINE)
            results_file.write('Dataset:'.ljust(20, ' ') + dataset_name + '\n')
            results_file.write('Model: '.ljust(20, ' ') + model_name + '\n')
            results_file.write('Task:'.ljust(20, ' ') + task + '\n')
            results_file.write('Train-validation: '.ljust(20, ' ') + 'test ' + str(split_proportions[0]) + ', valid: ' +
                               str(split_proportions[1]) + ', test: ' + str(split_proportions[2]) + '\n')
            results_file.write('Loss: '.ljust(20, ' ') + loss_name + '\n')

    def log_model_data(self, name: str, parameters: iter) -> None:
        with open(self.results_file_path, mode='a', encoding='utf-8') as results_file:
            results_file.write(MODEL_LINE)

            model_name_string = 'Model: ' + name + '\n'
            results_file.write(model_name_string)
            results_file.write('\n')

            lines_to_write = []
            try:
                layer_len = 25
                shape_len = 15
                parameters_len = 15

                header_row = 'Layer'.ljust(layer_len, ' ') + 'Shape'.rjust(shape_len, ' ') + 'Parameters'.rjust(parameters_len, ' ') + '\n'
                lines_to_write.append(header_row)

                separator = '='*(layer_len+shape_len+parameters_len) + '\n'
                lines_to_write.append(separator)

                tot_param_num = 0
                prev_name_split = ['', '']
                end = False
                name, parameter = next(parameters)
                n_name, n_parameter = next(parameters)
                while not end:
                    name_split = name.split('.')
                    depth = int((len(name_split) / 2) - 1)

                    if name_split[0] != prev_name_split[0]:
                        lines_to_write.append(name_split[0] + '\n')

                    prefix = ' ' + ' '*(4 * depth-2)
                    n_name0 = n_name.split('.')[0]
                    prefix += '\u2514' if (n_name == 'None' or name_split[0] != n_name0) else '\u251c'
                    prefix += '\u2500\u2500 layer' + name_split[-2] + ' (' + name_split[-1] + ')'

                    shape = '['
                    try:
                        shape += str(parameter.shape[0])
                        shape += 'x' + str(parameter.shape[1])
                    except Exception:
                        pass
                    shape += ']'

                    num_parameter = parameter.shape[0]
                    try:
                        num_parameter = num_parameter * parameter.shape[1]
                    except Exception:
                        pass
                    tot_param_num += num_parameter

                    line = prefix.ljust(layer_len, ' ') + shape.rjust(shape_len, ' ') + str(num_parameter).rjust(parameters_len, ' ') + '\n'
                    lines_to_write.append(line)

                    prev_name_split = name_split
                    name, parameter = n_name, n_parameter
                    n_name, n_parameter = next(parameters, ('None', 'None'))
                    if name == 'None':
                        end = True

                lines_to_write.append(separator)
                tot_line = 'TOTAL'.ljust(layer_len, ' ') + ' '*shape_len + str(tot_param_num).rjust(parameters_len, ' ') + '\n'
                lines_to_write.append(tot_line)

                for line in lines_to_write:
                    results_file.write(line)

            except Exception:
                for n, p in parameters:
                    results_file.write("\t" + str(n) + "  " + str(p.shape) + "")

    def log_training_data(self, curr_epoch: int, tot_epoch: int, time: str, train_loss: float, valid_loss: float or None, eval_measure_name: str,
                          eval_measures: dict, best_eval_measure: float, patience: (int, int) = None, early_stop: bool = False) -> str:
        # Write the data in the .txt file
        if self.is_training_first_log:
            with open(self.results_file_path, mode='a', encoding='utf-8') as results_file:
                results_file.write(TRAIN_LINE)
            self.is_training_first_log = False

        if not early_stop:
            step_output = str(' - Epoch ' + str(curr_epoch) + '/' + str(tot_epoch) + ' - ' + time[2:-7] + ' :')
            step_output += '  train loss: ' + str(round(train_loss, 3)).ljust(5, '0')
            if valid_loss is not None:
                step_output += '    valid loss: ' + str(round(valid_loss, 3)).ljust(5, '0')
            step_output += '    ' + str(eval_measure_name) + ': ' + str(round(eval_measures[eval_measure_name], 3)).ljust(5, '0')
            step_output += ' |  Best: ' + str(round(best_eval_measure, 3)).ljust(5, '0')
            step_output += '  patience: ' + str(patience[0]) + '/' + str(patience[1])
        else:
            step_output = str(' - Epoch ' + str(curr_epoch) + '/' + str(tot_epoch) + ' - Early Stop :')
            step_output += '   Best: ' + str(round(best_eval_measure, 3)).ljust(5, '0')

        with open(self.results_file_path, mode='a', encoding='utf-8') as results_file:
            results_file.write(step_output + '\n')

        # Sava data in the dataframe for .xlsx file
        new_dict = eval_measures
        new_dict['Train_loss'] = train_loss
        for key in new_dict:
            new_dict[key] = [new_dict[key]]

        df = pd.DataFrame.from_dict(new_dict)
        if self.results_df is None:
            self.results_df = df
        else:
            self.results_df = pd.concat([self.results_df, df], ignore_index=True)

        return ' ' + step_output

    def log_training_results(self, time: str, validation_measure: str, test_measures: dict) -> str:
        # Write the data in the .txt file
        results_output = 'Training time: ' + time + '\n'
        results_output += '\nResults:\n'
        results_output += '   Test loss: '.ljust(14, ' ') + str(test_measures['Loss']) + '\n'
        results_output += ('   Test ' + validation_measure + ': ').ljust(14, ' ') + str(test_measures[validation_measure]) + '\n'
        results_output += '\nOther measures:\n'
        for measure, value in list(test_measures.items())[:-1]:
            results_output += ('   ' + measure + ': ').ljust(14, ' ') + str(value) + '\n'

        with open(self.results_file_path, mode='a', encoding='utf-8') as results_file:
            results_file.write(RESULTS_LINE)
            results_file.write(results_output)

        # Sava data in the dataframe for .xlsx file
        new_dict = test_measures
        for key in new_dict:
            new_dict[key] = [new_dict[key]]

        df = pd.DataFrame.from_dict(new_dict)
        if self.results_df is None:
            self.results_df = df
        else:
            self.results_df = pd.concat([self.results_df, df], ignore_index=True)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        excel_path = os.path.join(self.experiment_folder_path, 'Training Data.xlsx')
        writer = pd.ExcelWriter(excel_path, mode='a' if os.path.isfile(excel_path) else 'w')
        self.results_df.to_excel(writer, sheet_name="Exp{:03d}".format(self.exp_num))
        writer.save()

        return results_output

    def log_network_predictions(self):
        return

    def log_network_parameters(self, ckpt_path: str) -> None:
        shutil.copy(ckpt_path, os.path.join(self.experiment_folder_path, 'model_parameters.pth.tar'))

    def generate_report(self) -> None:
        pdf_path = os.path.join(self.experiment_folder_path, 'Graphs.pdf')
        pdf = PdfPages(pdf_path)
        self._save_confusion_matrix_graph(pdf)
        self._save_confusion_matrix_data_graph(pdf)
        self._save_train_valid_loss_graph(pdf)
        self._save_valid_losses_graph(pdf)
        pdf.close()

    # SINGLE EXPERIMENT GRAPHS
    # ----------------------------------------------------

    def _save_confusion_matrix_graph(self, pdf):
        cf_matrix_df = self.results_df.iloc[-1]
        cf_matrix_values_list = cf_matrix_df['_cf_matrix_str'].split()
        tn = int(cf_matrix_values_list[0])
        fp = int(cf_matrix_values_list[1])
        fn = int(cf_matrix_values_list[2])
        tp = int(cf_matrix_values_list[3])
        cf_matrix = [[tn, fp, tn+fp],
                     [fn, tp, fn+tp],
                     [tn+fn, fp+tp, tn+fp+fn+tp]]

        df_cm = pd.DataFrame(cf_matrix, index=['tN', 'tP', 'pred'], columns=['oN', 'oP', 'true'], dtype=float)
        fig = plt.figure()
        sn.set(font_scale=1.5)  # for label size
        mask = np.zeros((3, 3))
        mask[2:2, 2] = True
        sn.heatmap(df_cm, mask=mask, annot=True, annot_kws={"size": 14}, fmt='n')  # font size
        pdf.savefig(fig)
        plt.clf()
        plt.close('all')

    def _save_confusion_matrix_data_graph(self, pdf):
        cf_matrix_df = pd.DataFrame(columns=['tn', 'fp', 'fn', 'tp'])
        cf_matrix_df[['tn', 'fp', 'fn', 'tp']] = self.results_df['_cf_matrix_str'].str.split(' ', expand=True)
        cf_matrix_df = cf_matrix_df.astype({'tn': int, 'fp': int, 'fn': int, 'tp': int})

        sn.set(font_scale=1)
        sn.set_style('whitegrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        x = self.results_df.index

        ax1.plot(x, cf_matrix_df['tn'])
        ax1.title.set_text('True Negative')
        ax2.plot(x, cf_matrix_df['fp'])
        ax2.title.set_text('False Positive')
        ax3.plot(x, cf_matrix_df['fn'])
        ax3.title.set_text('False Negative')
        ax4.plot(x, cf_matrix_df['tp'])
        ax4.title.set_text('True Positive')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.clf()
        plt.close('all')

    def _save_train_valid_loss_graph(self, pdf):
        sn.set(font_scale=1)
        sn.set_style('whitegrid')
        fig = plt.figure()
        x = self.results_df.index
        plt.plot(x, self.results_df['Train_loss'], label='Train')
        plt.plot(x, self.results_df['Loss'], label='Valid')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.clf()
        plt.close('all')

    def _save_valid_losses_graph(self, pdf):
        sn.set(font_scale=1)
        sn.set_style('whitegrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        x = self.results_df.index
        ax1.plot(x, self.results_df['Error'])
        ax1.title.set_text('Error')
        ax2.plot(x, self.results_df['Accuracy'])
        ax2.title.set_text('Accuracy')
        ax3.plot(x, self.results_df['MRR'])
        ax3.title.set_text('MRR')
        ax4.plot(x, self.results_df['MAP'])
        ax4.title.set_text('MAP')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.clf()
        plt.close('all')

        sn.set(font_scale=1)
        sn.set_style('whitegrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(x, self.results_df['Precision'])
        ax1.title.set_text('Precision')
        ax2.plot(x, self.results_df['Recall'])
        ax2.title.set_text('Recall')
        ax3.plot(x, self.results_df['bACC'])
        ax3.title.set_text('bACC')
        ax4.plot(x, self.results_df['F1'])
        ax4.title.set_text('F1')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.clf()
        plt.close('all')

    # COMPARATIVE GRAPHS
    # ----------------------------------------------------

    def _gen_comparative_cf_matrix_data_graph(self):
        return

    def _gen_comparative_train_loss_graph(self):
        return

    def _gen_comparative_table(self):
        return

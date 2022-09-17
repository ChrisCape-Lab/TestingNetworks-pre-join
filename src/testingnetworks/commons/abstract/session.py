import torch
import torch.distributed as dist
import yaml

from src.testingnetworks.utils import _set_seed


class Session:
    def __init__(self, dataset_folder_path: str, experiment_config_file_path: str, results_folder_path: str):
        self.dataset_folder_path = dataset_folder_path
        self.experiment_config_file_path = experiment_config_file_path
        self.results_folder_path = results_folder_path

        # Load network config
        with open(self.experiment_config_file_path, "r") as stream:
            try:
                experiment_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        self.experiment_config = experiment_config

        # Load dataset config
        self.dataloading_config = experiment_config['dataloading_parameters']
        dataset_full_name = self.dataset_folder_path.split('/')[-1]
        self.dataset_type = dataset_full_name.split('_')[0]
        self.dataset_name = dataset_full_name.replace(self.dataset_type + '_', '')

        print('\nTESTING: ' + self.dataset_type + '_' + self.dataset_name + ' on ' + str(self.experiment_config['network']) + '_' + str(self.experiment_config[
                                                                                                                                    'model']))
        print("---------------------------------------------", end='\n')

        # Set the correct device
        global rank, wsize
        use_cuda = (torch.cuda.is_available() and self.experiment_config['setup_parameters']['use_cuda'])
        self.experiment_config['setup_parameters']['use_cuda'] = use_cuda
        self.experiment_config['device'] = 'cpu'
        if use_cuda:
            self.experiment_config['device'] = 'cuda'

        try:
            dist.init_process_group(backend='mpi')  # , world_size=4
            rank = dist.get_rank()
            wsize = dist.get_world_size()
            print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
            if use_cuda:
                torch.cuda.set_device(rank)  # are we sure of the rank+1????
                print('using the device {}\n'.format(torch.cuda.current_device()))
        except:
            rank = 0
            wsize = 1
            print(('MPI backend not preset. Set process rank to {} (out of {})\n'.format(rank, wsize)))

        # Set all the packages' seed to make the experiment repeatable (same results)
        _set_seed(self.experiment_config['setup_parameters']['seed'])

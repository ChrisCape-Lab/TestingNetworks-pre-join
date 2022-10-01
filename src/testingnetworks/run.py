import os
import yaml

from session import Session, MultiSession

SINGLE = 0
SPECIAL = 1
SERIES = 2
BENCHMARK = 3

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file', default='config_experiments/AMLSim_EvolveGCN-h_Linear_node_cls_001.yaml',
                        type=argparse.FileType(mode='r'),
                        help='optional, yaml file containing parameters to be used, overrides command line parameters')

    args = parser.parse_args()
    """
    """
    mode = BENCHMARK
    if mode == SINGLE:
        f = '../config_experiments/GCN_Dense_000.yaml'
        print('Configuration: ' + str(f))
        session = Session(str(f))
        session.run()
    elif mode == SPECIAL:
        f = '../config_experiments/AMLSim_TriBrigade_000'
        print('Configuration: ' + str(f))
        session = MultiSession(str(f))
        session.run()
    elif mode == SERIES:
        # assign directory
        directory = '../config_experiments/'

        # iterate over files in that directory
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if f == '../../config_experiments/_gcn_classifier_example.yaml' or f == '../../config_experiments/_autoencoder_example.yaml':
                continue
            if os.path.isfile(f):
                print('Configuration: ' + str(f))
                session = Session(str(f))
                session.run()
            elif os.path.isdir(f):
                print('Configuration: ' + str(f))
                session = MultiSession(str(f))
                session.run()
                continue
    elif mode == BENCHMARK:
        # assign directory
        network_config_directory = '../../config_experiments/'
        dataset_config_directory = '../../datasets_configs/'

        # iterate over files in that directory
        for network_config_file in os.listdir(network_config_directory):
            network = os.path.join(network_config_directory, network_config_file)
            if network == '../../config_experiments/_gcn_classifier_example.yaml' or network == '../../config_experiments/_autoencoder_example.yaml':
                continue
            for dataset_config_file in os.listdir(dataset_config_directory):
                dataset = os.path.join(dataset_config_directory, dataset_config_file)

                # Run the simulation
                if os.path.isfile(network) and os.path.isfile(dataset):
                    print('Configuration: ' + str(network) + ' with ' + str(dataset))
                    session = Session(str(network), str(dataset))
                    session.run()
        else:
            raise NotImplementedError
    """
    # assign directory
    experiments_config_directory = '../../config_experiments/'
    datasets_directory = '../../datasets_output/'

    # iterate over files in that directory
    for experiment_config_file in os.listdir(experiments_config_directory):
        if experiment_config_file == '_gcn_classifier_example.yaml' or experiment_config_file == '_autoencoder_example.yaml':
            continue

        experiment = os.path.join(experiments_config_directory, experiment_config_file)

        for dataset_folder in os.listdir(datasets_directory):
            if dataset_folder == '_exclude':
                continue

            dataset_folder_path = os.path.join(datasets_directory, dataset_folder)

            # Run the simulation
            if os.path.isfile(experiment) and os.path.isdir(dataset_folder_path):
                # Load network config
                with open(str(experiment), "r") as stream:
                    try:
                        experiment_config = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                        exit()

                # Return the correct session
                if experiment_config['model'] == 'GCNClassifier':
                    from src.testingnetworks.frameworks.gnn_classifier.session import GCNClassifierSession

                    session = GCNClassifierSession(experiment_config_file_path=str(experiment), dataset_folder_path=str(dataset_folder_path),
                                                   results_folder_path='../../results')
                elif experiment_config['model'] == 'GraphAutoEncoder':
                    from src.testingnetworks.frameworks.gae.session import GraphAutoEncoderSession

                    session = GraphAutoEncoderSession(experiment_config_file_path=str(experiment), dataset_folder_path=str(dataset_folder_path),
                                                      results_folder_path='../../results')
                else:
                    raise NotImplementedError

                session.run()
                del session

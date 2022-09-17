import os
import pandas as pd

from src.testingnetworks._constants import WELL_KNOWN_FOLDERS as FOLDERS
from src.testingnetworks._constants import DATASET_TYPES


def preprocess_dataset(dataset_folder_path: str, dataset_full_name: str, dataset_type: str) -> (pd.DataFrame, pd.DataFrame):
    if not os.path.isdir(dataset_folder_path):
        raise NotADirectoryError('The path ' + dataset_folder_path + ' is not a folder')

    account_processed_df = pd.DataFrame()
    transactions_processed_df = pd.DataFrame()

    # If the selected dataset has alredy a valid cached processed version, return that version
    cached_dataset_path = os.path.join(FOLDERS.CACHE, dataset_full_name)
    if os.path.isdir(cached_dataset_path) and os.path.getmtime(cached_dataset_path) > os.path.getmtime(dataset_folder_path):
        account_processed_df = pd.read_csv(cached_dataset_path + '/accounts.csv', sep=',')
        transactions_processed_df = pd.read_csv(cached_dataset_path + '/transactions.csv', sep=',')
        print('Preprocessed dataset already in cache')
    # Otherwise, process the dataset, cache it and return
    else:
        account_csv_path = os.path.join(dataset_folder_path, 'accounts.csv')
        account_raw_df = pd.read_csv(account_csv_path, sep=',')
        transactions_csv_path = os.path.join(dataset_folder_path, 'transactions.csv')
        transactions_raw_df = pd.read_csv(transactions_csv_path, sep=',')

        if dataset_type == DATASET_TYPES.AMLDATAGEN:
            pass
        elif dataset_type == DATASET_TYPES.AMLSIM:
            from src.testingnetworks.commons.datapreprocess.amlsim_preprocess import preprocess_amlsim_with_timestamp
            account_processed_df, transactions_processed_df = preprocess_amlsim_with_timestamp(acc_df=account_raw_df, tx_df=transactions_raw_df)
        else:
            raise NotImplementedError

        if not os.path.isdir(cached_dataset_path):
            os.mkdir(cached_dataset_path)

        account_processed_df.to_csv(os.path.join(cached_dataset_path, 'accounts.csv'), sep=',', index=False)
        transactions_processed_df.to_csv(os.path.join(cached_dataset_path, 'transactions.csv'), sep=',', index=False)

    return account_processed_df, transactions_processed_df

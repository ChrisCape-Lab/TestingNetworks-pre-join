import os
import csv
import pandas as pd
import numpy as np

from src.testingnetworks.utils import DotDict


# CONSTANTS
# ----------------------------------------------------

AMLDataGen_raw_path = "../../../../datasets_output/raw/AMLDataGen/"
AMLDataGen_processed_path = "../../../../datasets_output/processed/AMLDataGen/"

# Columns of the accounts database
ncols = {'id': "id", 'exFraudster': "exFraudster", 'deposit': "deposit", 'bankid': "bankid", 'timestamp': "time", 'label': "label"}
NCOLS = DotDict(ncols)

# Columns of the transactions database
ecols = {'id': "tx_id", 'source': "orig_id", 'dest': "bene_id", 'weight': "amount", 'timestamp': "time", 'label': "label"}
ECOLS = DotDict(ecols)


# PREPARATION STEPS
# ----------------------------------------------------
def preprocess_csv(folder, csv_file):
    with open(folder + csv_file, 'r', newline='') as original_csv:
        with open(folder + 'tmp.csv', 'w', newline='') as destination_csv:
            writer = csv.writer(destination_csv)
            for row in csv.reader(original_csv):
                if row:
                    writer.writerow(row)
    os.remove(folder + csv_file)
    os.rename(folder + 'tmp.csv', folder + csv_file)


def _add_tx_counts(acc_df, tx_window_df):
    # Remove the cashes in-out from the list
    only_tx_df = tx_window_df[tx_window_df['orig_id'] != -1]
    only_tx_df = only_tx_df[only_tx_df['bene_id'] != -1]

    # Add tx_out_count
    tx_out_count = only_tx_df.groupby(['orig_id']).size().reset_index(name='tx_out_count')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_out_count": int})

    # Add tx_in_count
    tx_in_count = only_tx_df.groupby(['bene_id']).size().reset_index(name='tx_in_count')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_in_count": int})

    # Add tx_count
    acc_df['tx_count'] = acc_df['tx_in_count'] + acc_df['tx_out_count']

    return acc_df.sort_values(by=['id'])


def _add_tx_counts_unique(acc_df, tx_window_df):
    # Remove the cashes in-out from the list
    only_tx_df = tx_window_df[tx_window_df['orig_id'] != -1]
    only_tx_df = only_tx_df[only_tx_df['bene_id'] != -1]

    # Add tx_out_count
    tx_out_count = only_tx_df.drop_duplicates(subset=['orig_id', 'bene_id'], keep='first').groupby(['orig_id']).size()\
        .reset_index(name='tx_out_unique')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_out_unique": int})

    # Add tx_in_count
    tx_in_count = only_tx_df.drop_duplicates(subset=['orig_id', 'bene_id'], keep='first').groupby(['bene_id']).size() \
        .reset_index(name='tx_in_unique')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_in_unique": int})

    # Add tx_count
    acc_df['tx_count_unique'] = acc_df['tx_in_unique'] + acc_df['tx_out_unique']

    return acc_df.sort_values(by=['id'])


def _add_avg_tx_count(acc_df, history_df):
    if history_df.empty:
        acc_df['avg_tx_out_count'] = 0
        acc_df['avg_tx_in_count'] = 0
        return acc_df

    avg_amt_out = history_df.groupby(['id'])['tx_out_count'].mean().reset_index(name='avg_tx_out_count')
    avg_amt_out = avg_amt_out[['id', 'avg_tx_out_count']]
    acc_df = acc_df.merge(avg_amt_out, how='left', on=['id'])

    avg_amt_in = history_df.groupby(['id'])['tx_in_count'].mean().reset_index(name='avg_tx_in_count')
    avg_amt_in = avg_amt_in[['id', 'avg_tx_in_count']]
    acc_df = acc_df.merge(avg_amt_in, how='left', on=['id'])

    return acc_df.sort_values(by=['id'])


def _add_total_amt(acc_df, tx_window_df):
    # Remove the cashes in-out from the list
    only_tx_df = tx_window_df[tx_window_df['orig_id'] != -1]
    only_tx_df = only_tx_df[only_tx_df['bene_id'] != -1]

    # Add tot_amt_out
    tx_out_count = only_tx_df.groupby(['orig_id'])['amount'].sum().reset_index(name='tot_amt_out')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tot_amt_out": float})

    # Add tot_amt_in
    tx_in_count = only_tx_df.groupby(['bene_id'])['amount'].sum().reset_index(name='tot_amt_in')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tot_amt_in": float})

    # Add delta
    acc_df['delta'] = acc_df['tot_amt_out'] - acc_df['tot_amt_in']

    return acc_df.sort_values(by=['id'])


def _add_medium_amt(acc_df):
    acc_df['medium_amt_out'] = acc_df['tot_amt_out'] / acc_df['tx_out_count']
    acc_df['medium_amt_in'] = acc_df['tot_amt_in'] / acc_df['tx_in_count']
    acc_df = acc_df.fillna(0)

    return acc_df.sort_values(by=['id'])


def _add_avg_amt(acc_df, history_df):
    if history_df.empty:
        acc_df['avg_amt_out'] = 0
        acc_df = acc_df.astype({"avg_amt_out": float})
        acc_df['avg_amt_in'] = 0
        acc_df = acc_df.astype({"avg_amt_in": float})
        return acc_df

    avg_amt_out = history_df.groupby(['id'])['tot_amt_out'].mean().reset_index(name='avg_amt_out')
    avg_amt_out = avg_amt_out[['id', 'avg_amt_out']]
    acc_df = acc_df.merge(avg_amt_out, how='left', on=['id'])

    avg_amt_in = history_df.groupby(['id'])['tot_amt_in'].mean().reset_index(name='avg_amt_in')
    avg_amt_in = avg_amt_in[['id', 'avg_amt_in']]
    acc_df = acc_df.merge(avg_amt_in, how='left', on=['id'])

    return acc_df.sort_values(by=['id'])


def _set_exlaunderer(acc_df, tx_window_df):
    acc_df['exLaunderer'] = 0

    # Add tot_amt_out
    launder_value_df = tx_window_df.groupby(['orig_id'])['label'].sum().reset_index(name='exLaunderer1')
    launder_value_df = launder_value_df.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(launder_value_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"exLaunderer1": int})
    acc_df.loc[acc_df['exLaunderer1'] > 0, 'exLaunderer'] = 1

    # Add tot_amt_in
    launder_value_df = tx_window_df.groupby(['bene_id'])['label'].sum().reset_index(name='exLaunderer2')
    launder_value_df = launder_value_df.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(launder_value_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"exLaunderer2": int})
    acc_df.loc[acc_df['exLaunderer2'] > 0, 'exLaunderer'] = 1

    acc_df = acc_df.drop(columns=['exLaunderer1', 'exLaunderer2'], axis=1)

    return acc_df.sort_values(by=['id'])


def _adjust_deposit(acc_df, prev):
    if prev is None:
        return acc_df
    acc_df['deposit'] = acc_df['deposit'] + prev['tot_amt_in'] - prev['tot_amt_out']

    return acc_df.sort_values(by=['id'])


def _add_rounded_count(acc_df, tx_window_df):
    rounded_tx_df = tx_window_df[tx_window_df['amount'] % 10 == 0]
    rounded_tx_df = rounded_tx_df.groupby(['orig_id']).size().reset_index(name='tx_rounded_count')
    rounded_tx_df = rounded_tx_df.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(rounded_tx_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_rounded_count": int})

    return acc_df.sort_values(by=['id'])


def _add_small_count(acc_df, tx_window_df):
    small_tx_df = tx_window_df[tx_window_df['amount'] < 1000]
    small_tx_df = small_tx_df.groupby(['orig_id']).size().reset_index(name='tx_small_count')
    small_tx_df = small_tx_df.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(small_tx_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_small_count": int})

    return acc_df.sort_values(by=['id'])


def _add_repeated_amt_count(acc_df, tx_window_df):
    repeated_max_amts = tx_window_df.groupby(['orig_id', 'amount']).size().reset_index(name='repeated_amt_out_count')
    repeated_max_amts = repeated_max_amts.groupby(['orig_id'])['repeated_amt_out_count'].max().reset_index(name='repeated_amt_out_count')
    repeated_max_amts = repeated_max_amts.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(repeated_max_amts, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"repeated_amt_out_count": int})

    repeated_max_amts = tx_window_df.groupby(['bene_id', 'amount']).size().reset_index(name='repeated_amt_in_count')
    repeated_max_amts = repeated_max_amts.groupby(['bene_id'])['repeated_amt_in_count'].max().reset_index(name='repeated_amt_in_count')
    repeated_max_amts = repeated_max_amts.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(repeated_max_amts, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"repeated_amt_in_count": int})

    return acc_df.sort_values(by=['id'])


def _add_label_single_class(acc_df, tx_window_df):
    acc_df['label'] = 0

    tx_fraud = tx_window_df[tx_window_df['label'] == 1]
    # If the transaction is a fraud, the originator is not a launderer despite the transaction is related to ML
    tx_fraud = tx_fraud[tx_fraud['type'] != -2]
    frauds = tx_fraud.groupby(['orig_id'])['label'].mean().reset_index(name='label1')
    frauds = frauds.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(frauds, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"label1": int})
    acc_df.loc[acc_df['label1'] == 1, 'label'] = 1

    tx_fraud = tx_window_df[tx_window_df['label'] == 1]
    frauds = tx_fraud.groupby(['bene_id'])['label'].mean().reset_index(name='label2')
    frauds = frauds.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(frauds, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"label2": int})
    acc_df.loc[acc_df['label2'] == 1, 'label'] = 1

    acc_df = acc_df.drop(columns={'label1', 'label2'})

    return acc_df


def _add_label_multi_class(acc_df, tx_window_df):
    # TODO

    return acc_df


# PREPROCESS FUNCTION
# ----------------------------------------------------

def preprocess_amldatagen_dataset(folder, licit=False):
    print("\nStart preprocessing AMLDataGen dataset: (this may take a while)")
    print("Dataset: " + folder)

    # ======= Accounts loading =======
    # TODO: change it when changing dir
    acc_df = pd.read_csv(AMLDataGen_raw_path + folder + "Accounts.csv")
    acc_df = acc_df[['id', 'init_balance', 'bank_id']]
    acc_df = acc_df.rename(columns={'init_balance': 'deposit'})
    acc_df = acc_df.astype({"id": int, "deposit": float, "bank_id": str})
    acc_df.sort_values('id')

    # Convert bank names to IDs
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    idx = 0
    for letter in letters:
        acc_df.loc[acc_df['bank_id'] == 'bank_'+letter, 'bank_id'] = idx
        idx += 1
    acc_df = acc_df.astype({"bank_id": int})

    # ======= Transactions loading =======
    tx_df = pd.read_csv(AMLDataGen_raw_path + folder + "Transactions.csv")
    tx_df = tx_df[['id', 'src', 'dst', 'amt', 'time', 'type', 'is_aml']]
    tx_df = tx_df.rename(
        columns={'id': 'tx_id', 'src': 'orig_id', 'dst': 'bene_id', 'amt': 'amount', 'is_aml': 'label'})
    tx_df = tx_df.fillna(-1)
    tx_df = tx_df.astype({'tx_id': int, 'orig_id': int, 'bene_id': int, 'amount': float, 'time': int, 'label': str})

    # Convert boolean string to values
    tx_df.loc[tx_df['label'] == 'True', 'label'] = 1
    tx_df.loc[tx_df['label'] == 'False', 'label'] = 0
    tx_df = tx_df.astype({'label': int})

    if licit:
        tx_df = tx_df[tx_df['label'] == 0]

    # ======= Account Preprocessing =======
    max_time = tx_df[['time']].max()[0]
    acc_perpoc_df = pd.DataFrame()
    prev = None

    print("Accounts number: " + str(len(acc_df.index)))
    print("Transactions number: " + str(len(tx_df.index)))
    print("Timesteps: " + str(max_time))

    for t in range(0, max_time+1):
        tx_window_df = tx_df[tx_df['time'] == t]

        # If there is no correspondence, skip the computation
        if tx_window_df.empty:
            continue

        if (t == 0) or ((t+1) % 10 == 0) or (t == max_time):
            print('  step ' + str(t+1) + ' of ' + str(max_time+1))

        acc_time_df = acc_df
        acc_time_df['time'] = t

        acc_time_df = _add_tx_counts(acc_time_df, tx_window_df)
        acc_time_df = _add_tx_counts_unique(acc_time_df, tx_window_df)
        acc_time_df = _add_avg_tx_count(acc_time_df, acc_perpoc_df)
        acc_time_df = _add_total_amt(acc_time_df, tx_window_df)
        acc_time_df = _add_medium_amt(acc_time_df)
        acc_time_df = _add_avg_amt(acc_time_df, acc_perpoc_df)
        tx_window_laund_df = tx_df[tx_df['time'] < t]
        acc_time_df = _set_exlaunderer(acc_time_df, tx_window_laund_df)
        acc_time_df = _adjust_deposit(acc_time_df, prev)
        acc_time_df = _add_repeated_amt_count(acc_time_df, tx_window_df)
        acc_time_df = _add_rounded_count(acc_time_df, tx_window_df)
        acc_time_df = _add_small_count(acc_time_df, tx_window_df)
        acc_time_df = _add_label_single_class(acc_time_df, tx_window_df)

        acc_perpoc_df = pd.concat((acc_perpoc_df, acc_time_df))
        prev = acc_time_df

    quantile_df = acc_perpoc_df[['tx_in_count', 'tx_out_count']]
    tx_in_quant, tx_out_quant = np.quantile(quantile_df, 0.95, axis=0)
    acc_perpoc_df['high_fan_in'] = 0
    acc_perpoc_df.loc[acc_perpoc_df['tx_in_count'] > tx_in_quant, 'high_fan_in'] = 1
    acc_perpoc_df['high_fan_out'] = 0
    acc_perpoc_df.loc[acc_perpoc_df['tx_out_count'] > tx_out_quant, 'high_fan_out'] = 1

    if not os.path.isdir(AMLDataGen_processed_path + folder):
        os.mkdir(AMLDataGen_processed_path + folder)
    if not licit:
        acc_perpoc_df.to_csv(AMLDataGen_processed_path + folder + "accounts.csv", sep=',', index=False)
    else:
        acc_perpoc_df.to_csv(AMLDataGen_processed_path + folder + "accounts_licit.csv", sep=',', index=False)

    # Remove the cashes in-out from the list
    only_tx_df = tx_df[tx_df['orig_id'] != -1]
    only_tx_df = only_tx_df[only_tx_df['bene_id'] != -1]
    only_tx_df.to_csv(AMLDataGen_processed_path + folder + "transactions.csv", sep=',', index=False)

    print('Produced account dimension: ' + str(len(acc_perpoc_df)))
    print('Ended datasets_output preprocessing\n')


# "MAIN"
# ----------------------------------------------------

if __name__ == '__main__':
    # TODO: remove this when in correct folder
    folder = AMLDataGen_raw_path
    for file in os.listdir(folder):
        dataset_dir = os.path.join(folder, file)
        if os.path.isfile(dataset_dir):
            continue
        preprocess_amldatagen_dataset(file + "/", licit=False)

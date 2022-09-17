import os
import pandas as pd
import numpy as np

from datetime import datetime
from src.testingnetworks.utils import DotDict


# CONSTANTS
# ----------------------------------------------------

# Columns of the accounts database
ncols = {'id': "id", 'exFraudster': "exFraudster", 'deposit': "deposti", 'bankid': "bankid", 'timestamp': "timestamp", 'label': "label"}
NCOLS = DotDict(ncols)

# Columns of the transactions database
ecols = {'id': "tx_id", 'source': "orig_id", 'dest': "bene_id", 'weight': "amount", 'timestamp': "timestamp", 'label': "label"}
ECOLS = DotDict(ecols)


# PREPARATION STEPS
# ----------------------------------------------------

def _add_tx_counts(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['tx_out_count'] = 0
        acc_df['tx_in_count'] = 0
        acc_df['tx_count'] = 0
        return acc_df

    # Add tx_out_count
    tx_out_count = tx_window_df.groupby(['orig_id']).size().reset_index(name='tx_out_count')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_out_count": int})

    # Add tx_in_count
    tx_in_count = tx_window_df.groupby(['bene_id']).size().reset_index(name='tx_in_count')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_in_count": int})

    # Add tx_count
    acc_df['tx_count'] = acc_df['tx_in_count'] + acc_df['tx_out_count']

    return acc_df.sort_values(by=['id'])


def _add_tx_counts_unique(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['tx_out_unique'] = 0
        acc_df['tx_in_unique'] = 0
        acc_df['tx_count_unique'] = 0
        return acc_df

    # Add tx_out_count
    tx_out_count = tx_window_df.drop_duplicates(subset=['orig_id', 'bene_id'], keep='first').groupby(['orig_id']).size()\
        .reset_index(name='tx_out_unique')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_out_unique": int})

    # Add tx_in_count
    tx_in_count = tx_window_df.drop_duplicates(subset=['orig_id', 'bene_id'], keep='first').groupby(['bene_id']).size() \
        .reset_index(name='tx_in_unique')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_in_unique": int})

    # Add tx_count
    acc_df['tx_count_unique'] = acc_df['tx_in_unique'] + acc_df['tx_out_unique']

    return acc_df.sort_values(by=['id'])


def _add_avg_tx_count(acc_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
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


def _add_total_amt(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['tot_amt_out'] = 0
        acc_df['tot_amt_in'] = 0
        acc_df['delta'] = 0
        return acc_df

    # Add tot_amt_out
    tx_out_count = tx_window_df.groupby(['orig_id'])['amount'].sum().reset_index(name='tot_amt_out')
    tx_out_count = tx_out_count.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(tx_out_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tot_amt_out": float})

    # Add tot_amt_in
    tx_in_count = tx_window_df.groupby(['bene_id'])['amount'].sum().reset_index(name='tot_amt_in')
    tx_in_count = tx_in_count.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(tx_in_count, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tot_amt_in": float})

    # Add delta
    acc_df['delta'] = acc_df['tot_amt_out'] - acc_df['tot_amt_in']

    return acc_df.sort_values(by=['id'])


def _add_medium_amt(acc_df: pd.DataFrame) -> pd.DataFrame:
    acc_df['medium_amt_out'] = acc_df['tot_amt_out'] / acc_df['tx_out_count'].copy().replace(to_replace=0, value=1)
    acc_df['medium_amt_in'] = acc_df['tot_amt_in'] / acc_df['tx_in_count'].copy().replace(to_replace=0, value=1)
    acc_df = acc_df.fillna(0)

    return acc_df.sort_values(by=['id'])


def _add_avg_amt(acc_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
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


def _set_exlaunderer(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['exLaunderer'] = 0
        return acc_df

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


def _adjust_deposit(acc_df: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if prev is None:
        return acc_df
    acc_df['deposit'] = acc_df['deposit'] + prev['tot_amt_in'] - prev['tot_amt_out']

    return acc_df.sort_values(by=['id'])


def _add_rounded_count(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['tx_rounded_count'] = 0
        return acc_df

    rounded_tx_df = tx_window_df[tx_window_df['amount'] % 10 == 0]
    rounded_tx_df = rounded_tx_df.groupby(['orig_id']).size().reset_index(name='tx_rounded_count')
    rounded_tx_df = rounded_tx_df.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(rounded_tx_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_rounded_count": int})

    return acc_df.sort_values(by=['id'])


def _add_small_count(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
    if tx_window_df.empty:
        acc_df['tx_small_count'] = 0
        return acc_df

    small_tx_df = tx_window_df[tx_window_df['amount'] < 1000]
    small_tx_df = small_tx_df.groupby(['orig_id']).size().reset_index(name='tx_small_count')
    small_tx_df = small_tx_df.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(small_tx_df, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"tx_small_count": int})

    return acc_df.sort_values(by=['id'])


def _add_repeated_amt_count(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
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


def _add_label(acc_df: pd.DataFrame, tx_window_df: pd.DataFrame):
    acc_df['label'] = 0

    tx_fraud = tx_window_df[tx_window_df['label'] == 1]
    frauds = tx_fraud.groupby(['orig_id'])['label'].mean().reset_index(name='label1')
    frauds = frauds.rename(columns={'orig_id': 'id'})
    acc_df = acc_df.merge(frauds, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"label1": int})
    acc_df.loc[acc_df['label1'] == 1, 'label'] = 1

    frauds = tx_fraud.groupby(['bene_id'])['label'].mean().reset_index(name='label2')
    frauds = frauds.rename(columns={'bene_id': 'id'})
    acc_df = acc_df.merge(frauds, how='left', on=['id'])
    acc_df = acc_df.fillna(0)
    acc_df = acc_df.astype({"label2": int})
    acc_df.loc[acc_df['label2'] == 1, 'label'] = 1

    acc_df = acc_df.drop(columns={'label1', 'label2'})

    return acc_df


# PREPROCESS FUNCTION
# ----------------------------------------------------

def preprocess_amlsim_with_timestamp(acc_df: pd.DataFrame, tx_df: pd.DataFrame, licit: bool = False) -> (pd.DataFrame, pd.DataFrame):
    print("Start preprocessing AMLSim dataset (this may take a while)")

    # ======= Accounts loading =======
    acc_df = acc_df[['acct_id', 'initial_deposit', 'bank_id']]
    acc_df = acc_df.rename(columns={'acct_id': 'id', 'initial_deposit': 'deposit', 'bank_id': 'bankID'})
    acc_df = acc_df.astype({"id": int, "deposit": float, "bankID": str})
    acc_df.sort_values('id')

    # Convert bank names to IDs
    acc_df.loc[acc_df['bankID'] == 'bank_a', 'bankID'] = 0
    acc_df.loc[acc_df['bankID'] == 'bank_b', 'bankID'] = 1
    acc_df.loc[acc_df['bankID'] == 'bank_c', 'bankID'] = 2
    acc_df = acc_df.astype({"bankID": int})

    # ======= Transactions loading =======
    tx_df = tx_df[['tran_id', 'orig_acct', 'bene_acct', 'base_amt', 'tran_timestamp', 'is_sar', 'alert_id']]
    tx_df = tx_df.rename(
        columns={'tran_id': 'tx_id', 'orig_acct': 'orig_id', 'bene_acct': 'bene_id', 'base_amt': 'amount',
                 'tran_timestamp': 'timestamp', 'is_sar': 'label'})
    tx_df = tx_df.astype({'tx_id': int, 'orig_id': int, 'bene_id': int, 'amount': float, 'timestamp': str, 'label': str})

    # Convert boolean string to values
    tx_df.loc[tx_df['label'] == 'True', 'label'] = 1
    tx_df.loc[tx_df['label'] == 'False', 'label'] = 0
    tx_df = tx_df.astype({'label': int})

    # Convert date from string format into datetime format, to enable easy operations
    tx_df['timestamp'] = tx_df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

    # Add wee and day number for easy aggregation (verbose but easy to understand)
    tx_df['day_num'] = tx_df['timestamp'].dt.isocalendar().day
    tx_df['week_num'] = tx_df['timestamp'].dt.isocalendar().week
    tx_df['year_num'] = tx_df['timestamp'].dt.isocalendar().year
    tx_df.sort_values(['orig_id', 'week_num', 'year_num'])

    if licit:
        tx_df = tx_df[tx_df['label'] == 0]

    # ======= Account Preprocessing =======
    # Adjust time
    max_time = 0
    tx_df['time'] = 0
    for year in range(tx_df['year_num'].min(), tx_df['year_num'].max()+1):
        for week in range(tx_df['week_num'].min(), tx_df['week_num'].max()+1):
            # Find the starting year and week before update time
            if tx_df[(tx_df['week_num'] == week) & (tx_df['year_num'] == year)].empty and max_time == 0:
                continue

            tx_df.loc[(tx_df['week_num'] == week) & (tx_df['year_num'] == year), 'time'] = max_time
            max_time += 1
    tx_df = tx_df.drop(columns={'week_num', 'year_num'})

    acc_perpoc_df = pd.DataFrame()
    prev = None

    print("[DEBUG] Accounts number: " + str(len(acc_df.index)))
    print("[DEBUG] Transactions number: " + str(len(tx_df.index)))
    print("[DEBUG] Timesteps: " + str(max_time))

    for t in range(0, max_time-1):
        tx_window_df = tx_df[tx_df['time'] == t]

        """
        # If there is no correspondence, skip the computation
        if tx_window_df.empty and t != max_time-1:
            print('EMPTY at time ' + str(t))
            continue
        """

        if (t == 0) or ((t + 1) % 10 == 0) or (t == max_time-1):
            print('[DEBUG]   step ' + str(t + 1) + ' of ' + str(max_time))

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
        acc_time_df = _add_label(acc_time_df, tx_window_df)

        acc_perpoc_df = pd.concat((acc_perpoc_df, acc_time_df))
        prev = acc_time_df

    quantile_df = acc_perpoc_df[['tx_in_count', 'tx_out_count']]
    tx_in_quant, tx_out_quant = np.quantile(quantile_df, 0.95, axis=0)
    acc_perpoc_df['high_fan_in'] = 0
    acc_perpoc_df.loc[acc_perpoc_df['tx_in_count'] > tx_in_quant, 'high_fan_in'] = 1
    acc_perpoc_df['high_fan_out'] = 0
    acc_perpoc_df.loc[acc_perpoc_df['tx_out_count'] > tx_out_quant, 'high_fan_out'] = 1

    # ======= Transactions Preprocessing =======
    """
    # Would be useful for all transactions, but just for alert is no that much...
    alert_df = pd.read_csv(AMLSIM_raw_path + "alert_transactions.csv")
    alert_df = alert_df[['alert_id', 'tran_id', 'alert_type']]

    test = tx_df.merge(alert_df, on=['alert_id'])
    print(len(test.index))
    test.to_csv(AMLSIM_processed_path + folder + "test.csv", sep=',')
    exit()

    tx_df = tx_df.merge(alert_df, how='left', left_on=['tx_id'], right_on=['tran_id'])
    tx_df.drop(columns={'alert_id'})
    tx_df = tx_df.fillna('')
    print(len(tx_df.index))
    print(tx_df.columns)
    """

    print('[DEBUG] Produced account dimension: ' + str(len(acc_perpoc_df)))
    print('Ended datasets_output preprocessing\n')

    return acc_perpoc_df, tx_df

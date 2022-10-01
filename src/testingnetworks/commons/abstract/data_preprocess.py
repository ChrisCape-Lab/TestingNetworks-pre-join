import numpy as np
import pandas as pd

from src.testingnetworks._constants import NODE_COLUMNS as NCOLS, EDGE_COLUMNS as ECOLS


class DataPreprocess:
    def __init__(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame):
        self.account_df = account_df
        self.transactions_df = transactions_df

    # PREPROCEsS CALLABLE FUNCTION
    # ----------------------------------------------------

    def preprocess_dataset(self):
        account_df, transaction_df = self.uniform_dataset_structure(account_df=self.account_df, transactions_df=self.transactions_df)

        acc_perpoc_df = pd.DataFrame()
        prev = None
        max_time = max(transaction_df[ECOLS.TIME])+1

        print("[DEBUG] Accounts number: " + str(len(account_df.index)))
        print("[DEBUG] Transactions number: " + str(len(transaction_df.index)))
        print("[DEBUG] Timesteps: " + str(max_time))

        for t in range(0, max_time):
            tx_window_df = transaction_df[transaction_df[ECOLS.TIME] == t]

            # If there is no correspondence, skip the computation
            if tx_window_df.empty and t != max_time-1:
                print('EMPTY at time ' + str(t))
                continue

            if (t == 0) or ((t + 1) % 10 == 0) or (t == max_time - 1):
                print('[DEBUG]   step ' + str(t + 1) + ' of ' + str(max_time))

            acc_time_df = account_df.copy()
            acc_time_df[NCOLS.TIME] = t

            if len(account_df.columns) > 3:
                print('Problem in account preprocessing')
                exit()

            acc_time_df = self.add_tx_counts(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_tx_counts_unique(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_avg_tx_count(accounts_df=acc_time_df, history_df=acc_perpoc_df)
            acc_time_df = self.add_total_amt(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_medium_amt(accounts_df=acc_time_df)
            acc_time_df = self.add_avg_amt(accounts_df=acc_time_df, history_df=acc_perpoc_df)
            tx_window_laund_df = transaction_df[transaction_df[ECOLS.TIME] < t]
            acc_time_df = self.set_exlaunderer(accounts_df=acc_time_df, tx_window_df=tx_window_laund_df)
            acc_time_df = self.adjust_deposit(accounts_df=acc_time_df, precedent_accounts_df=prev)
            acc_time_df = self.add_repeated_amt_count(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_rounded_count(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_small_count(accounts_df=acc_time_df, tx_window_df=tx_window_df)
            acc_time_df = self.add_label(accounts_df=acc_time_df, tx_window_df=tx_window_df)

            acc_perpoc_df = pd.concat((acc_perpoc_df, acc_time_df))
            prev = acc_time_df

        quantile_df = acc_perpoc_df[[NCOLS.TX_IN_COUNT, NCOLS.TX_OUT_COUNT]]
        tx_in_quant, tx_out_quant = np.quantile(quantile_df, 0.95, axis=0)
        acc_perpoc_df[NCOLS.HIGH_FAN_IN] = 0
        acc_perpoc_df.loc[acc_perpoc_df[NCOLS.TX_IN_COUNT] > tx_in_quant, NCOLS.HIGH_FAN_IN] = 1
        acc_perpoc_df[NCOLS.HIGH_FAN_OUT] = 0
        acc_perpoc_df.loc[acc_perpoc_df[NCOLS.TX_OUT_COUNT] > tx_out_quant, NCOLS.HIGH_FAN_OUT] = 1

        print('[DEBUG] Produced account dimension: ' + str(len(acc_perpoc_df)))
        print('Ended datasets_output preprocessing\n')

        return acc_perpoc_df, transaction_df

    # PREPROCESSING FUNCTIONS
    # ----------------------------------------------------

    def uniform_dataset_structure(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        raise NotImplementedError

    def add_tx_counts(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        # If the transaction dataframe is empty, then there are no transaction in the timestep
        if tx_window_df.empty:
            accounts_df[NCOLS.TX_OUT_COUNT] = 0
            accounts_df[NCOLS.TX_IN_COUNT] = 0
            accounts_df[NCOLS.TX_COUNT] = 0
            return accounts_df

        # Add tx_out_count
        tx_out_count = tx_window_df.groupby([ECOLS.ORIGINATOR]).size().reset_index(name=NCOLS.TX_OUT_COUNT)
        tx_out_count = tx_out_count.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_out_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_OUT_COUNT: int})

        # Add tx_in_count
        tx_in_count = tx_window_df.groupby([ECOLS.BENEFICIARY]).size().reset_index(name=NCOLS.TX_IN_COUNT)
        tx_in_count = tx_in_count.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_in_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_IN_COUNT: int})

        # Add tx_count
        accounts_df[NCOLS.TX_COUNT] = accounts_df[NCOLS.TX_IN_COUNT] + accounts_df[NCOLS.TX_OUT_COUNT]

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_tx_counts_unique(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        # If the transaction dataframe is empty, then there are no transaction in the timestep
        if tx_window_df.empty:
            accounts_df[NCOLS.TX_OUT_UNIQUE] = 0
            accounts_df[NCOLS.TX_IN_UNIQUE] = 0
            accounts_df[NCOLS.TX_COUNT_UNIQUE] = 0
            return accounts_df

        # Add tx_out_count
        tx_out_count = tx_window_df.drop_duplicates(subset=[ECOLS.ORIGINATOR, ECOLS.BENEFICIARY], keep='first').groupby([ECOLS.ORIGINATOR]).size() \
            .reset_index(name=NCOLS.TX_OUT_UNIQUE)
        tx_out_count = tx_out_count.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_out_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_OUT_UNIQUE: int})

        # Add tx_in_count
        tx_in_count = tx_window_df.drop_duplicates(subset=[ECOLS.ORIGINATOR, ECOLS.BENEFICIARY], keep='first').groupby([ECOLS.BENEFICIARY]).size() \
            .reset_index(name=NCOLS.TX_IN_UNIQUE)
        tx_in_count = tx_in_count.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_in_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_IN_UNIQUE: int})

        # Add tx_count
        accounts_df[NCOLS.TX_COUNT_UNIQUE] = accounts_df[NCOLS.TX_IN_UNIQUE] + accounts_df[NCOLS.TX_OUT_UNIQUE]

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_avg_tx_count(self, accounts_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        # If the transaction dataframe is empty, then there are no transaction in the timestep
        if history_df.empty:
            accounts_df[NCOLS.AVG_TX_OUT_COUNT] = 0
            accounts_df[NCOLS.AVG_TX_IN_COUNT] = 0
            return accounts_df

        # Add the average number of out transactions
        avg_amt_out = history_df.groupby([NCOLS.ID])[NCOLS.TX_OUT_COUNT].mean().reset_index(name=NCOLS.AVG_TX_OUT_COUNT)
        avg_amt_out = avg_amt_out[[NCOLS.ID, NCOLS.AVG_TX_OUT_COUNT]]
        accounts_df = accounts_df.merge(avg_amt_out, how='left', on=[NCOLS.ID])

        # Add the average number of in transactions
        avg_amt_in = history_df.groupby([NCOLS.ID])[NCOLS.TX_IN_COUNT].mean().reset_index(name=NCOLS.AVG_TX_IN_COUNT)
        avg_amt_in = avg_amt_in[[NCOLS.ID, NCOLS.AVG_TX_IN_COUNT]]
        accounts_df = accounts_df.merge(avg_amt_in, how='left', on=[NCOLS.ID])

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_total_amt(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        # If the transaction dataframe is empty, then there are no transaction in the timestep
        if tx_window_df.empty:
            accounts_df[NCOLS.TOT_AMT_OUT] = 0
            accounts_df[NCOLS.TOT_AMT_IN] = 0
            accounts_df[NCOLS.DELTA] = 0
            return accounts_df

        # Add tot_amt_out
        tx_out_count = tx_window_df.groupby([ECOLS.ORIGINATOR])[ECOLS.AMOUNT].sum().reset_index(name=NCOLS.TOT_AMT_OUT)
        tx_out_count = tx_out_count.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_out_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TOT_AMT_OUT: float})

        # Add tot_amt_in
        tx_in_count = tx_window_df.groupby([ECOLS.BENEFICIARY])[ECOLS.AMOUNT].sum().reset_index(name=NCOLS.TOT_AMT_IN)
        tx_in_count = tx_in_count.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(tx_in_count, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TOT_AMT_IN: float})

        # Add delta
        # TODO: revert and check result
        accounts_df[NCOLS.DELTA] = accounts_df[NCOLS.TOT_AMT_OUT] - accounts_df[NCOLS.TOT_AMT_IN]

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_medium_amt(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        accounts_df[NCOLS.MEDIUM_AMT_OUT] = accounts_df[NCOLS.TOT_AMT_OUT] / accounts_df[NCOLS.TX_OUT_COUNT].copy().replace(to_replace=0, value=1)
        accounts_df[NCOLS.MEDIUM_AMT_IN] = accounts_df[NCOLS.TOT_AMT_IN] / accounts_df[NCOLS.TX_IN_COUNT].copy().replace(to_replace=0, value=1)
        accounts_df = accounts_df.fillna(0)

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_avg_amt(self, accounts_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        # If the history df is empty it's the first step, so all averages are 0
        if history_df.empty:
            accounts_df[NCOLS.AVG_AMT_OUT] = .0
            accounts_df[NCOLS.AVG_AMT_IN] = .0
            return accounts_df

        # Add average amount out
        avg_amt_out = history_df.groupby([NCOLS.ID])[NCOLS.TOT_AMT_OUT].mean().reset_index(name=NCOLS.AVG_AMT_OUT)
        avg_amt_out = avg_amt_out[[NCOLS.ID, NCOLS.AVG_AMT_OUT]]
        accounts_df = accounts_df.merge(avg_amt_out, how='left', on=[NCOLS.ID])

        # Add average amount in
        avg_amt_in = history_df.groupby([NCOLS.ID])[NCOLS.TOT_AMT_IN].mean().reset_index(name=NCOLS.AVG_AMT_IN)
        avg_amt_in = avg_amt_in[[NCOLS.ID, NCOLS.AVG_AMT_IN]]
        accounts_df = accounts_df.merge(avg_amt_in, how='left', on=[NCOLS.ID])

        return accounts_df.sort_values(by=[NCOLS.ID])

    def set_exlaunderer(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        if tx_window_df.empty:
            accounts_df[NCOLS.EX_LAUNDERER] = 0
            return accounts_df

        accounts_df[NCOLS.EX_LAUNDERER] = 0

        # Add tot_amt_out
        launder_value_df = tx_window_df.groupby([ECOLS.ORIGINATOR])[ECOLS.LABEL].sum().reset_index(name='exLaunderer1')
        launder_value_df = launder_value_df.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(launder_value_df, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({"exLaunderer1": int})
        accounts_df.loc[accounts_df['exLaunderer1'] > 0, NCOLS.EX_LAUNDERER] = 1

        # Add tot_amt_in
        launder_value_df = tx_window_df.groupby([ECOLS.BENEFICIARY])[ECOLS.LABEL].sum().reset_index(name='exLaunderer2')
        launder_value_df = launder_value_df.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(launder_value_df, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({"exLaunderer2": int})
        accounts_df.loc[accounts_df['exLaunderer2'] > 0, NCOLS.EX_LAUNDERER] = 1

        accounts_df = accounts_df.drop(columns=['exLaunderer1', 'exLaunderer2'], axis=1)

        return accounts_df.sort_values(by=[NCOLS.ID])

    def adjust_deposit(self, accounts_df: pd.DataFrame, precedent_accounts_df: pd.DataFrame) -> pd.DataFrame:
        if precedent_accounts_df is None:
            return accounts_df
        accounts_df[NCOLS.DEPOSIT] = accounts_df[NCOLS.DEPOSIT] + precedent_accounts_df[NCOLS.TOT_AMT_IN] - precedent_accounts_df[NCOLS.TOT_AMT_OUT]

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_rounded_count(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        if tx_window_df.empty:
            accounts_df[NCOLS.TX_ROUNDED_COUNT] = 0
            return accounts_df

        rounded_tx_df = tx_window_df[tx_window_df[ECOLS.AMOUNT] % 10 == 0]
        rounded_tx_df = rounded_tx_df.groupby([ECOLS.ORIGINATOR]).size().reset_index(name=NCOLS.TX_ROUNDED_COUNT)
        rounded_tx_df = rounded_tx_df.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(rounded_tx_df, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_ROUNDED_COUNT: int})

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_small_count(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        if tx_window_df.empty:
            accounts_df[NCOLS.TX_SMALL_COUNT] = 0
            return accounts_df

        small_tx_df = tx_window_df[tx_window_df[ECOLS.AMOUNT] < 1000]
        small_tx_df = small_tx_df.groupby([ECOLS.ORIGINATOR]).size().reset_index(name=NCOLS.TX_SMALL_COUNT)
        small_tx_df = small_tx_df.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(small_tx_df, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.TX_SMALL_COUNT: int})

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_repeated_amt_count(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame) -> pd.DataFrame:
        repeated_max_amts = tx_window_df.groupby([ECOLS.ORIGINATOR, ECOLS.AMOUNT]).size().reset_index(name=NCOLS.REPEATED_AMT_OUT_COUNT)
        repeated_max_amts = repeated_max_amts.groupby([ECOLS.ORIGINATOR])[NCOLS.REPEATED_AMT_OUT_COUNT].max().reset_index(name=NCOLS.REPEATED_AMT_OUT_COUNT)
        repeated_max_amts = repeated_max_amts.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(repeated_max_amts, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.REPEATED_AMT_OUT_COUNT: int})

        repeated_max_amts = tx_window_df.groupby([ECOLS.BENEFICIARY, ECOLS.AMOUNT]).size().reset_index(name=NCOLS.REPEATED_AMT_IN_COUNT)
        repeated_max_amts = repeated_max_amts.groupby([ECOLS.BENEFICIARY])[NCOLS.REPEATED_AMT_IN_COUNT].max().reset_index(name=NCOLS.REPEATED_AMT_IN_COUNT)
        repeated_max_amts = repeated_max_amts.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(repeated_max_amts, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({NCOLS.REPEATED_AMT_IN_COUNT: int})

        return accounts_df.sort_values(by=[NCOLS.ID])

    def add_label(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame):
        accounts_df[NCOLS.LABEL] = 0

        tx_fraud = tx_window_df[tx_window_df[ECOLS.LABEL] == 1]
        frauds = tx_fraud.groupby([ECOLS.ORIGINATOR])[ECOLS.LABEL].mean().reset_index(name='label1')
        frauds = frauds.rename(columns={ECOLS.ORIGINATOR: NCOLS.ID})
        accounts_df = accounts_df.merge(frauds, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({'label1': int})
        accounts_df.loc[accounts_df['label1'] == 1, NCOLS.LABEL] = 1

        frauds = tx_fraud.groupby([ECOLS.BENEFICIARY])[ECOLS.LABEL].mean().reset_index(name='label2')
        frauds = frauds.rename(columns={ECOLS.BENEFICIARY: NCOLS.ID})
        accounts_df = accounts_df.merge(frauds, how='left', on=[NCOLS.ID])
        accounts_df = accounts_df.fillna(0)
        accounts_df = accounts_df.astype({"label2": int})
        accounts_df.loc[accounts_df['label2'] == 1, NCOLS.LABEL] = 1

        accounts_df = accounts_df.drop(columns={'label1', 'label2'})

        return accounts_df

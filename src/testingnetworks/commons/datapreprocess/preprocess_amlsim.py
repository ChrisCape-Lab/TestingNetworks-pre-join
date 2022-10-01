import pandas as pd

from datetime import datetime

from src.testingnetworks._constants import NODE_COLUMNS as NCOLS, EDGE_COLUMNS as ECOLS
from src.testingnetworks.commons.abstract.data_preprocess import DataPreprocess


class AMLSimPreprocess(DataPreprocess):
    def __init__(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame):
        super().__init__(account_df, transactions_df)

    def preprocess_dataset(self):
        print("Start preprocessing AMLSim dataset (this may take a while)")
        return super().preprocess_dataset()

    def uniform_dataset_structure(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Load and correctly rename account file
        account_df_proc = account_df.copy()
        account_df_proc = account_df_proc[['acct_id', 'initial_deposit', 'bank_id']]
        account_df_proc = account_df_proc.rename(columns={'acct_id': NCOLS.ID, 'initial_deposit': NCOLS.DEPOSIT, 'bank_id': NCOLS.BANK_ID})
        account_df_proc = account_df_proc.astype({NCOLS.ID: int, NCOLS.DEPOSIT: float, NCOLS.BANK_ID: str})
        account_df_proc.sort_values(NCOLS.ID)

        # Convert bank names to IDs
        account_df_proc.loc[account_df_proc[NCOLS.BANK_ID] == 'bank_a', NCOLS.BANK_ID] = 0
        account_df_proc.loc[account_df_proc[NCOLS.BANK_ID] == 'bank_b', NCOLS.BANK_ID] = 1
        account_df_proc.loc[account_df_proc[NCOLS.BANK_ID] == 'bank_c', NCOLS.BANK_ID] = 2
        account_df_proc = account_df_proc.astype({NCOLS.BANK_ID: int})

        # Load and correctly rename transactions file
        transactions_df_proc = transactions_df.copy()
        transactions_df_proc = transactions_df_proc[['tran_id', 'orig_acct', 'bene_acct', 'base_amt', 'tran_timestamp', 'is_sar', 'alert_id']]
        transactions_df_proc = transactions_df_proc.rename(
            columns={'tran_id': ECOLS.ID, 'orig_acct': ECOLS.ORIGINATOR, 'bene_acct': ECOLS.BENEFICIARY, 'base_amt': ECOLS.AMOUNT,
                     'tran_timestamp': 'timestamp', 'is_sar': ECOLS.LABEL})
        transactions_df_proc = transactions_df_proc.astype({ECOLS.ID: int, ECOLS.ORIGINATOR: int, ECOLS.BENEFICIARY: int, ECOLS.AMOUNT: float, 'timestamp': str,
                                                            ECOLS.LABEL: str})

        # Convert boolean string to values
        transactions_df_proc.loc[transactions_df_proc[ECOLS.LABEL] == 'True', ECOLS.LABEL] = 1
        transactions_df_proc.loc[transactions_df_proc[ECOLS.LABEL] == 'False', ECOLS.LABEL] = 0
        transactions_df_proc = transactions_df_proc.astype({ECOLS.LABEL: int})

        # Convert date from string format into datetime format, to enable easy operations
        transactions_df_proc['timestamp'] = transactions_df_proc['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

        # Add wee and day number for easy aggregation (verbose but easy to understand)
        transactions_df_proc['week_num'] = transactions_df_proc['timestamp'].dt.isocalendar().week
        transactions_df_proc['year_num'] = transactions_df_proc['timestamp'].dt.isocalendar().year
        transactions_df_proc.sort_values([ECOLS.ORIGINATOR, 'week_num', 'year_num'])

        # Adjust time
        max_time = 0
        transactions_df_proc[ECOLS.TIME] = 0
        for year in range(transactions_df_proc['year_num'].min(), transactions_df_proc['year_num'].max() + 1):
            for week in range(transactions_df_proc['week_num'].min(), transactions_df_proc['week_num'].max() + 1):
                # Find the starting year and week before update time
                if transactions_df_proc[(transactions_df_proc['week_num'] == week) & (transactions_df_proc['year_num'] == year)].empty and max_time == 0:
                    continue

                transactions_df_proc.loc[(transactions_df_proc['week_num'] == week) & (transactions_df_proc['year_num'] == year), NCOLS.TIME] = max_time
                max_time += 1
        transactions_df_proc = transactions_df_proc.drop(columns={'week_num', 'year_num', 'timestamp'})

        return account_df_proc, transactions_df_proc

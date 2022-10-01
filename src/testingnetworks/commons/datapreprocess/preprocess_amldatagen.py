import pandas as pd

from src.testingnetworks._constants import NODE_COLUMNS as NCOLS, EDGE_COLUMNS as ECOLS
from src.testingnetworks.commons.abstract.data_preprocess import DataPreprocess


class AMLDataGenPreprocess(DataPreprocess):
    def __init__(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame):
        super().__init__(account_df, transactions_df)

    def preprocess_dataset(self):
        print("Start preprocessing AMLSim dataset (this may take a while)")
        return super().preprocess_dataset()

    def uniform_dataset_structure(self, account_df: pd.DataFrame, transactions_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Load and correctly rename account file
        account_df_proc = account_df.copy()
        account_df_proc = account_df_proc[['id', 'init_balance', 'bank_id']]
        account_df_proc = account_df_proc.rename(columns={'id': NCOLS.ID, 'init_balance': NCOLS.DEPOSIT, 'bank_id': NCOLS.BANK_ID})
        account_df_proc = account_df_proc.astype({NCOLS.ID: int, NCOLS.DEPOSIT: float, NCOLS.BANK_ID: str})
        account_df_proc.sort_values('id')

        # Convert bank names to IDs
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        idx = 0
        for letter in letters:
            account_df_proc.loc[account_df_proc[NCOLS.BANK_ID] == 'bank_' + letter, NCOLS.BANK_ID] = idx
            idx += 1
        account_df_proc = account_df_proc.astype({NCOLS.BANK_ID: int})

        # Load and correctly rename transactions file
        transactions_df_proc = transactions_df.copy()
        transactions_df_proc = transactions_df_proc[['id', 'src', 'dst', 'amt', 'time', 'type', 'is_aml']]
        transactions_df_proc = transactions_df_proc.rename(
            columns={'id': ECOLS.ID, 'src': ECOLS.ORIGINATOR, 'dst': ECOLS.BENEFICIARY, 'amt': ECOLS.AMOUNT, 'type': ECOLS.PATTERN, 'is_aml': ECOLS.LABEL})
        transactions_df_proc = transactions_df_proc.fillna(-1)
        transactions_df_proc = transactions_df_proc.astype({ECOLS.ID: int, ECOLS.ORIGINATOR: int, ECOLS.BENEFICIARY: int, ECOLS.AMOUNT: float, ECOLS.TIME: int, ECOLS.LABEL: str})

        # Convert boolean string to values
        transactions_df_proc.loc[transactions_df_proc[ECOLS.LABEL] == 'True', ECOLS.LABEL] = 1
        transactions_df_proc.loc[transactions_df_proc[ECOLS.LABEL] == 'False', ECOLS.LABEL] = 0
        transactions_df_proc = transactions_df_proc.astype({ECOLS.LABEL: int})

        return account_df_proc, transactions_df_proc

    def add_label(self, accounts_df: pd.DataFrame, tx_window_df: pd.DataFrame):
        accounts_df[NCOLS.LABEL] = 0

        tx_fraud = tx_window_df[tx_window_df[ECOLS.LABEL] == 1]
        # If the transaction is a fraud, the originator is not a launderer despite the transaction is related to ML
        tx_fraud = tx_fraud[tx_fraud[ECOLS.PATTERN] != -2]
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

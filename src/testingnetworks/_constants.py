
class NODE_COLUMNS:
    ID = 'ID'
    DEPOSIT = 'deposit'
    BANK_ID = 'bank_id'
    TIME = 'TIME'

    TX_OUT_COUNT = 'tx_out_count'
    TX_IN_COUNT = 'tx_in_count'
    TX_COUNT = 'tx_count'

    TX_OUT_UNIQUE = 'tx_out_unique'
    TX_IN_UNIQUE = 'tx_in_unique'
    TX_COUNT_UNIQUE = 'tx_count_unique'

    AVG_TX_OUT_COUNT = 'avg_tx_out_count'
    AVG_TX_IN_COUNT = 'avg_tx_in_count'

    TOT_AMT_OUT = 'tot_amt_out'
    TOT_AMT_IN = 'tot_amt_in'
    DELTA = 'delta'

    MEDIUM_AMT_OUT = 'medium_amt_out'
    MEDIUM_AMT_IN = 'medium_amt_in'

    AVG_AMT_OUT = 'avg_amt_out'
    AVG_AMT_IN = 'avg_amt_in'

    EX_LAUNDERER = 'ex_launderer'

    TX_ROUNDED_COUNT = 'tx_rounded_count'

    TX_SMALL_COUNT = 'tx_small_count'

    REPEATED_AMT_OUT_COUNT = 'repeated_amt_out_count'
    REPEATED_AMT_IN_COUNT = 'repeated_amt_in_count'

    HIGH_FAN_IN = 'high_fan_in'
    HIGH_FAN_OUT = 'high_fan_out'

    LABEL = 'label'


class EDGE_COLUMNS:
    ID = 'ID'
    ORIGINATOR = 'originator'
    BENEFICIARY = 'beneficiary'
    AMOUNT = 'amount'
    TIME = 'TIME'
    PATTERN = 'pattern'
    LABEL = 'label'





class WELL_KNOWN_FOLDERS:
    CURRENT = '.'
    CACHE = CURRENT + '/cache'


class DATASET_TYPES:
    AMLDATAGEN = 'ADG'
    AMLSIM = 'AS'


class DATA:
    ADJACENCY_MATRIX = 'adjacency_matrix'
    CONNECTION_MATRIX = 'connection_matrix'
    MODULARITY_MATRIX = 'modularity_matrix'
    NODE_FEATURES = 'node_features'
    EDGE_FEATURES = 'edge_features'

    LABELS = 'labels'
    NODE_LABELS = 'node_labels'
    EDGE_LABELS = 'edge_labels'

    NODE_MASK = 'node_mask'
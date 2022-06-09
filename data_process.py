import feather
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier
def get_training(cell_line):
    training = (
        feather.read_dataframe(f'{cell_line}-training.feather')
        .set_index(['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2'])
    )

    # chromhmm states
    active_chromhmm_states = {
        '1_TssA',
        '2_TssAFlnk',
        '3_TxFlnk',
        '4_Tx',
        '5_TxWk',
        '6_EnhG',
        '7_Enh'
    }
    inactive_chromhmm_states = {
        '8_ZNF/Rpts',
        '9_Het',
        '11_BivFlnk',
        '12_EnhBiv',
        '13_ReprPC',
        '14_ReprPCWk',
        '15_Quies'
    }

    # filtering on chromatin states
    f1_active_cols = [f'{state} (bin1)' for state in active_chromhmm_states]
    f2_active_cols = [f'{state} (bin2)' for state in active_chromhmm_states]
    f1_inactive_cols = [f'{state} (bin1)' for state in inactive_chromhmm_states]
    f2_inactive_cols = [f'{state} (bin2)' for state in inactive_chromhmm_states]

    training['active chromatin (bin1)'] = training[f1_active_cols].sum(axis = 1) > 0
    training['active chromatin (bin2)'] = training[f2_active_cols].sum(axis = 1) > 0
    training['inactive chromatin (bin1)'] = training[f1_inactive_cols].sum(axis = 1) > 0
    training['inactive chromatin (bin2)'] = training[f2_inactive_cols].sum(axis = 1) > 0
    training = training[
        training['active chromatin (bin1)'].values &
        training['active chromatin (bin2)'].values
    ]

    # subsampling for desired class balance
    n_negatives = training.eval('label == 0').sum()
    n_positives = training.eval('label == 1').sum()
    if n_negatives > n_positives * negatives_per_positive:
        pos_training = training.query('label == 1')
        neg_training = (
            training
            .query('label == 0')
            .sample(
                n_positives * negatives_per_positive,
                random_state = 0
            )
        )
        training = pd.concat([pos_training, neg_training])
        training.sort_index(inplace = True)

    return training


n_jobs = -1

# training data
negatives_per_positive = 10

df = get_training('HeLa-S3')
df.to_csv('HeLa-S3.csv')

df = get_training('GM12878')
df.to_csv('GM12878.csv')

df = get_training('IMR90')
df.to_csv('IMR90.csv')

df = get_training('K562')
df.to_csv('K562.csv')

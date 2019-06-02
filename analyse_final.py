import datasets
import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr


ALGORITHMS = [
    'RBU', 'RUS', 'AKNN', 'CC', 'CNN', 'ENN', 'IHT', 'NCL', 'NM', 'OSS', 'RENN', 'TL',
    'ROS', 'SMOTE', 'Bord', 'RBO', 'SMOTE+TL', 'SMOTE+ENN'
]
CLASSIFIERS = ['CART', 'KNN', 'NB', 'SVM']
METRICS = ['Precision', 'Recall', 'F-measure', 'AUC', 'G-mean']
P_VALUE = 0.10
RESULTS_PATH = Path(__file__).parent / 'results'


def load_final_dict(classifier, metric):
    csv_path = RESULTS_PATH / 'results.csv'

    df = pd.read_csv(csv_path)
    df = df[df['Description'] == 'Final']

    df['Scores'] = df['Scores'].str.replace('\'', '"')
    df['Parameters'] = df['Parameters'].str.replace('\'', '"')
    df['Classifier'] = df['Parameters'].apply(lambda x: json.loads(x)['classifier'])

    df[metric] = df['Scores'].apply(lambda x: json.loads(x)[metric])

    df = df.drop(['Parameters', 'Description', 'Scores'], axis=1)

    df = df[df['Classifier'] == classifier]

    df = df.groupby(
        ['Dataset', 'Classifier', 'Algorithm']
    )[metric].agg('mean').reset_index()

    rows = []

    for dataset in datasets.names('final'):
        row = [dataset]

        for algorithm in ALGORITHMS:
            row.append(np.round(list(df[(df['Algorithm'] == algorithm) & (df['Dataset'] == dataset)][metric])[0], 4))

        rows.append(row)

    ds = pd.DataFrame(rows, columns=['Dataset'] + ALGORITHMS)

    ds.to_csv(RESULTS_PATH / ('%s_%s.csv' % (classifier, metric)), index=False)

    measurements = OrderedDict()

    for algorithm in ALGORITHMS:
        measurements[algorithm] = []

        for dataset in datasets.names('final'):
            scores = df[(df['Algorithm'] == algorithm) & (df['Dataset'] == dataset)][metric]

            assert len(scores) == 1

            measurements[algorithm].append(list(scores)[0])

    return measurements


def test_friedman_shaffer(dictionary):
    df = pd.DataFrame(dictionary)

    columns = df.columns

    pandas2ri.activate()

    importr('scmamp')

    rFriedmanTest = r['friedmanTest']
    rPostHocTest = r['postHocTest']

    initial_results = rFriedmanTest(df)
    posthoc_results = rPostHocTest(df, test='friedman', correct='shaffer', use_rank=True)

    ranks = np.array(posthoc_results[0])[0]
    p_value = initial_results[2][0]
    corrected_p_values = np.array(posthoc_results[2])

    ranks_dict = {col: rank for col, rank in zip(columns, ranks)}
    corrected_p_values_dict = {}

    for outer_col, corr_p_val_vect in zip(columns, corrected_p_values):
        corrected_p_values_dict[outer_col] = {}

        for inner_col, corr_p_val in zip(columns, corr_p_val_vect):
            corrected_p_values_dict[outer_col][inner_col] = corr_p_val

    return ranks_dict, p_value, corrected_p_values_dict


if __name__ == '__main__':
    for classifier in CLASSIFIERS:
        for metric in METRICS:
            if metric == METRICS[0]:
                start = '\\parbox[t]{2mm}{\\multirow{5}{*}{\\rotatebox[origin=c]{90}{%s}}}' % classifier
            else:
                start = ''

            d = load_final_dict(classifier, metric)
            ranks, _, corrected_p_values = test_friedman_shaffer(d)

            row = [start, metric]

            best_rank = sorted(set(ranks.values()))[0]
            second_best_rank = sorted(set(ranks.values()))[1]

            for algorithm in ALGORITHMS:
                rank = ranks[algorithm]
                col = '%.1f' % np.round(rank, 1)

                if rank == best_rank:
                    col = '\\first{%s}' % col
                elif rank == second_best_rank:
                    col = '\\second{%s}' % col

                if corrected_p_values['RBU'][algorithm] <= P_VALUE:
                    if rank < ranks['RBU']:
                        col = '%s \\textsubscript{--}' % col
                    else:
                        col = '%s \\textsubscript{+}' % col

                row.append(col)

            print(' & '.join(row) + ' \\\\')

        if classifier != CLASSIFIERS[-1]:
            print('\\midrule')

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extract_dataset_info import extract
from pathlib import Path
from scipy.stats import pearsonr


CLASSIFIERS = ['CART', 'KNN', 'NB', 'SVM']
METRICS = ['Precision', 'Recall', 'F-measure', 'AUC', 'G-mean']
PTYPES = ['safe', 'borderline', 'rare', 'outlier']
P_VALUE = 0.10
RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


def prepare_df():
    try:
        info = pd.read_csv(RESULTS_PATH / 'dataset_info.csv')
    except FileNotFoundError:
        print('Extracting dataset info...')

        info = extract(verbose=False)

    info = info[info['name'].isin(datasets.names('final'))].reset_index(drop=True)
    info['Dataset'] = info['name']

    rows = []

    for clf in CLASSIFIERS:
        for metric in METRICS:
            res = pd.read_csv(RESULTS_PATH / ('%s_%s.csv' % (clf, metric)))
            res['rank'] = list(res.rank(axis=1, ascending=False)['RBU'])

            res = pd.merge(res, info, on='Dataset')

            assert len(res) == 30

            for ptype in PTYPES:
                for index, row in res.iterrows():
                    newrow = [row['name'], ptype, clf, metric, row['percentage_%s' % ptype], row['rank']]

                    rows.append(newrow)

    return pd.DataFrame(rows, columns=['name', 'type', 'clf', 'metric', 'percentage [%]', 'rank'])


def export_correlation(df):
    for clf in CLASSIFIERS:
        for metric in METRICS:
            if metric == METRICS[0]:
                start = '\\parbox[t]{2mm}{\\multirow{5}{*}{\\rotatebox[origin=c]{90}{%s}}}' % clf
            else:
                start = ''

            row = [start, metric]

            for ptype in PTYPES:
                ds = df[(df['clf'] == clf) & (df['metric'] == metric) & (df['type'] == ptype)]

                x = list(ds['percentage [%]'])
                y = list(ds['rank'])

                rho, pval = pearsonr(x, y)

                col = '%+.4f' % np.round(rho, 4)

                if pval <= P_VALUE:
                    col = '\\textbf{%s}' % col

                row.append(col)

            print(' & '.join(row) + ' \\\\')

        if clf != CLASSIFIERS[-1]:
            print('\\midrule')


def visualize(df, metric):
    g = sns.lmplot(
        'percentage [%]', 'rank', data=df[df['metric'] == metric],
        col='type', row='clf', hue='type', truncate=True,
        sharex=False, height=2.5
    )
    g.set(ylim=(0.5, 18.5))

    VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

    plt.savefig(
        VISUALIZATIONS_PATH / ('type_impact_%s.pdf' % metric),
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':
    df = prepare_df()
    export_correlation(df)

    for metric in METRICS:
        visualize(df, metric)

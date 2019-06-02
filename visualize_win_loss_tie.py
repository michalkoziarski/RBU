import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


CLASSIFIERS = ['CART', 'KNN', 'NB', 'SVM']
METRICS = ['Precision', 'Recall', 'F-measure', 'AUC', 'G-mean']
RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


def get_win_loss_tie_df():
    rows = []

    for clf in CLASSIFIERS:
        for metric in METRICS:
            results = pd.read_csv(RESULTS_PATH / ('%s_%s.csv' % (clf, metric)))

            methods = results.columns[2:]

            for method in methods:
                wins = sum(results['RBU'] > results[method])
                ties = sum(results['RBU'] == results[method])
                losses = sum(results['RBU'] < results[method])

                rows.append([clf, metric, method, wins, ties, losses])

    return pd.DataFrame(rows, columns=['clf', 'metric', 'method', 'wins', 'ties', 'losses'])


def visualize(df, clf, metric):
    ds = df[(df['clf'] == clf) & (df['metric'] == metric)]
    ds.loc[ds['method'] == 'SMOTE+ENN', 'method'] = 'SENN'
    ds.loc[ds['method'] == 'SMOTE+TL', 'method'] = 'STL'
    ds['losses'] = ds['wins'] + ds['ties'] + ds['losses']
    ds['ties'] = ds['wins'] + ds['ties']

    f, ax = plt.subplots(figsize=(12, 4))

    sns.set_color_codes('bright')
    sns.barplot(y='losses', x='method', data=ds,
                label='losses', color=sns.color_palette()[3])

    sns.barplot(y='ties', x='method', data=ds,
                label='ties', color='y')

    sns.barplot(y='wins', x='method', data=ds,
                label='wins', color=sns.color_palette()[2])

    ax.set(ylim=(0, 30), title=metric, ylabel='number of datasets', xlabel='')

    sns.despine(left=True, bottom=True)

    plt.savefig(
        VISUALIZATIONS_PATH / ('win-loss-tie_%s_%s.pdf' % (clf, metric)),
        bbox_inches='tight'
    )
    plt.close(f)


if __name__ == '__main__':
    df = get_win_loss_tie_df()

    for metric in ['F-measure', 'AUC', 'G-mean']:
        visualize(df, 'NB', metric)

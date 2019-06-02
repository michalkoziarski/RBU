import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


METRICS = ['Precision', 'Recall', 'F-measure', 'AUC', 'G-mean']


def visualize(parameter, xlabel=None, logscale=False, logscalebase=10, plot_function=sns.lineplot):
    if xlabel is None:
        xlabel = parameter

    results_path = Path(__file__).parent / 'results' / 'results.csv'
    visualizations_path = Path(__file__).parent / 'visualizations'
    visualizations_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(results_path)
    df = df[df['Description'] == 'Preliminary (%s)' % parameter]

    df['Scores'] = df['Scores'].str.replace('\'', '"')
    df['Parameters'] = df['Parameters'].str.replace('\'', '"')

    for metric in METRICS:
        df[metric] = df['Scores'].apply(lambda x: json.loads(x)[metric])

    df['Classifier'] = df['Parameters'].apply(lambda x: json.loads(x)['classifier'])
    df['Parameter'] = df['Parameters'].apply(lambda x: json.loads(x)[parameter][0])

    df = df.drop(['Algorithm', 'Parameters', 'Description', 'Scores'], axis=1)

    df = df.groupby(
        ['Dataset', 'Classifier', 'Parameter']
    )[METRICS].agg('mean').reset_index()

    for metric in METRICS:
        grid = sns.FacetGrid(df, col='Dataset', hue='Classifier', col_wrap=5, height=1.8, aspect=1.3)
        grid.map(plot_function, 'Parameter', metric)
        grid.set_titles('{col_name}')
        grid.set(ylim=(0, 1), xlabel=xlabel)

        handles = grid._legend_data.values()
        labels = grid._legend_data.keys()

        grid.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)

        if logscale:
            plt.xscale('log', basex=logscalebase)

        plt.tight_layout()

        grid.fig.subplots_adjust(bottom=0.14)

        plt.savefig(visualizations_path / ('preliminary_%s_individual_%s.pdf' % (parameter, metric)))

    rows = []

    for _, row in df.iterrows():
        for metric in METRICS:
            rows.append([
                row['Dataset'],
                row['Classifier'],
                row['Parameter'],
                metric,
                row[metric]
            ])

    df = pd.DataFrame(rows, columns=['Dataset', 'Classifier', 'Parameter', 'Metric', 'Score'])

    grid = sns.FacetGrid(df, row='Metric', col='Classifier', hue='Metric', height=1.8, aspect=1.3)
    grid.map(plot_function, 'Parameter', 'Score')
    grid.set_titles('{col_name}')
    grid.set(xlabel=xlabel)

    for i, axes_row in enumerate(grid.axes):
        for j, axes_col in enumerate(axes_row):
            if j == 0:
                axes_col.set_ylabel(METRICS[i])
            else:
                axes_col.set_ylabel('')

    if logscale:
        plt.xscale('log', basex=logscalebase)

    plt.tight_layout()

    plt.savefig(visualizations_path / ('preliminary_%s_average.pdf' % parameter))


if __name__ == '__main__':
    visualize('gamma', r'$\gamma$', logscale=True)
    visualize('ratio')

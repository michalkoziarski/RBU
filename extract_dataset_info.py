import datasets
import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def extract(verbose=True):
    dfs = []

    for partition in ['preliminary', 'final']:
        rows = []

        for name in tqdm(datasets.names(partition)):
            dataset = datasets.load(name)

            (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]

            X = np.concatenate([X_train, X_test])
            y = np.concatenate([y_train, y_test])

            n_samples = X.shape[0]
            n_features = X.shape[1]

            majority_class = Counter(y).most_common()[0][0]

            n_majority_samples = Counter(y).most_common()[0][1]
            n_minority_samples = Counter(y).most_common()[1][1]

            imbalance_ratio = np.round(n_majority_samples / n_minority_samples, 2)

            knn = NearestNeighbors(6).fit(X)

            n_safe = 0
            n_borderline = 0
            n_rare = 0
            n_outliers = 0

            for X_i, y_i in zip(X, y):
                if y_i == majority_class:
                    continue
                else:
                    indices = knn.kneighbors([X_i], return_distance=False)[0, 1:]
                    n_majority_neighbors = sum(y[indices] == majority_class)

                    if n_majority_neighbors in [0, 1]:
                        n_safe += 1
                    elif n_majority_neighbors in [2, 3]:
                        n_borderline += 1
                    elif n_majority_neighbors == 4:
                        n_rare += 1
                    elif n_majority_neighbors == 5:
                        n_outliers += 1
                    else:
                        raise ValueError

            n_total = n_safe + n_borderline + n_rare + n_outliers

            percentage_safe = np.round(n_safe / n_total * 100, 2)
            percentage_borderline = np.round(n_borderline / n_total * 100, 2)
            percentage_rare = np.round(n_rare / n_total * 100, 2)
            percentage_outlier = np.round(n_outliers / n_total * 100, 2)

            rows.append([
                name, imbalance_ratio, n_samples, n_features,
                percentage_safe, percentage_borderline,
                percentage_rare, percentage_outlier
            ])

        df = pd.DataFrame(rows, columns=[
            'name', 'imbalance_ratio', 'n_samples', 'n_features',
            'percentage_safe', 'percentage_borderline',
            'percentage_rare', 'percentage_outlier'
        ])
        df = df.sort_values('imbalance_ratio')

        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(Path(__file__).parent / 'results' / 'dataset_info.csv', index=False)

    if verbose:
        for i, row in df.iterrows():
            row = [str(v).replace('_', '\_') for v in row]

            print(' & '.join(row) + ' \\\\')

            if i == 19:
                print('\\midrule')

    return df


if __name__ == '__main__':
    extract()

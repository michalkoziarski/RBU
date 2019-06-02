import logging
import numpy as np
import os
import pandas as pd
import pickle
import zipfile

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
FOLDS_PATH = os.path.join(os.path.dirname(__file__), 'folds')

URLS = {
    'preliminary': [
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone19.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-2_vs_8.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/cleveland-0_vs_4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-1-4-7_vs_2-3-5-6.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-2-6-7_vs_3-5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-6-7_vs_3-5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/glass-0-1-4-6_vs_2.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-3-5-9_vs_7-8.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-20_vs_8-9-10.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kr-vs-k-zero_vs_eight.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_6.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8_vs_6.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-8_vs_6.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3-9_vs_5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli1.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass0.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/vehicle3.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/yeast3.zip'
    ],
    'final': [
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone9-18.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass-0-1-6_vs_2.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass2.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/page-blocks-1-3_vs_4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-0-5-6-7-9_vs_4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1-2-8-9_vs_7.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1-4-5-8_vs_7.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1_vs_7.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-2_vs_4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast4.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast6.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-6-7_vs_5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-2-5-6_vs_3-7-8-9.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-17_vs_7-8-9-10.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-19_vs_10-11-12-13.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-21_vs_8.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/car-good.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/car-vgood.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/flare-F.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-buffer_overflow_vs_back.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-rootkit-imap_vs_back.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_5.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3_vs_7.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli3.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass1.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/haberman.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/vehicle1.zip',
        'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/yeast1.zip'
    ]
}


def download(url):
    name = url.split('/')[-1]
    download_path = os.path.join(DATA_PATH, name)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(download_path):
        urlretrieve(url, download_path)

    if not os.path.exists(download_path.replace('.zip', '.dat')):
        if name.endswith('.zip'):
            with zipfile.ZipFile(download_path) as zip:
                zip.extractall(DATA_PATH)
        else:
            raise Exception('Unrecognized file type.')


def encode(X, y, encode_features=True):
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if encode_features:
        encoded = []

        for i in range(X.shape[1]):
            try:
                for j in range(len(X)):
                    float(X[j, i])

                encoded.append(X[:, i])
            except ValueError:
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))

        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def partition(X, y):
    partitions = []

    for _ in range(5):
        folds = []
        skf = StratifiedKFold(n_splits=2, shuffle=True)

        for train_idx, test_idx in skf.split(X, y):
            folds.append([train_idx, test_idx])

        partitions.append(folds)

    return partitions


def load(name, url=None, encode_features=True, remove_metadata=True, scale=True):
    file_name = '%s.dat' % name

    if url is not None:
        download(url)

    skiprows = 0

    if remove_metadata:
        with open(os.path.join(DATA_PATH, file_name)) as f:
            for line in f:
                if line.startswith('@'):
                    skiprows += 1
                else:
                    break

    df = pd.read_csv(os.path.join(DATA_PATH, file_name), header=None, skiprows=skiprows, skipinitialspace=True,
                     sep=' *, *', na_values='?', engine='python')

    matrix = df.dropna().values

    X, y = matrix[:, :-1], matrix[:, -1]
    X, y = encode(X, y, encode_features)

    partitions_path = os.path.join(FOLDS_PATH, file_name.replace('.dat', '.folds.pickle'))

    if not os.path.exists(FOLDS_PATH):
        os.mkdir(FOLDS_PATH)

    if os.path.exists(partitions_path):
        partitions = pickle.load(open(partitions_path, 'rb'))
    else:
        partitions = partition(X, y)
        pickle.dump(partitions, open(partitions_path, 'wb'))

    folds = []

    for i in range(5):
        for j in range(2):
            train_idx, test_idx = partitions[i][j]
            train_set = [X[train_idx].copy(), y[train_idx].copy()]
            test_set = [X[test_idx].copy(), y[test_idx].copy()]

            if scale:
                scaler = StandardScaler().fit(train_set[0])
                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

            folds.append([train_set, test_set])

    return folds


def load_all(partition_type=None):
    assert partition_type in [None, 'preliminary', 'final']

    urls = []

    for current_type in ['preliminary', 'final']:
        if partition_type in [None, current_type]:
            urls += URLS[current_type]

    datasets = {}

    for url in urls:
        name = url.split('/')[-1].replace('.zip', '')
        datasets[name] = load(name, url)

    return datasets


def names(partition_type=None):
    assert partition_type in [None, 'preliminary', 'final']

    urls = []

    for current_type in ['preliminary', 'final']:
        if partition_type in [None, current_type]:
            urls += URLS[current_type]

    return [url.split('/')[-1].replace('.zip', '') for url in urls]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading datasets...')

    load_all()

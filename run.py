import logging
import metrics
import multiprocessing as mp

from collections import Counter
from cv import ResamplingCV
from databases import pull_pending, submit_result
from datasets import load
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN as AKNN
from imblearn.under_sampling import ClusterCentroids as CC
from imblearn.under_sampling import CondensedNearestNeighbour as CNN
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.under_sampling import InstanceHardnessThreshold as IHT
from imblearn.under_sampling import RepeatedEditedNearestNeighbours as RENN
from imblearn.under_sampling import NearMiss as NM
from imblearn.under_sampling import NeighbourhoodCleaningRule as NCL
from imblearn.under_sampling import OneSidedSelection as OSS
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.under_sampling import TomekLinks as TL
from rbo import RBO
from rbu import RBU
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as CART


N_PROCESSES = 24


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        params = eval(trial['Parameters'])

        logging.info(trial)

        dataset = load(trial['Dataset'])
        fold = int(trial['Fold']) - 1

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        n_minority = Counter(y_train).most_common()[1][1]
        n_majority = Counter(y_train).most_common()[0][1]

        imblearn_ratios = [
            ((n_majority - n_minority) * ratio + n_minority) / n_majority
            for ratio in [0.5, 0.75, 1.0]
        ]

        clf = {
            'NB': NB(),
            'KNN': KNN(),
            'SVM': SVM(gamma='scale'),
            'CART': CART()
        }[params['classifier']]

        if (trial['Algorithm'] is None) or (trial['Algorithm'] == 'None'):
            algorithm = None
        else:
            algorithms = {
                'AKNN': ResamplingCV(
                    AKNN, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'Bord': ResamplingCV(
                    SMOTE, clf,
                    kind=['borderline1'],
                    k_neighbors=[1, 3, 5, 7, 9],
                    m_neighbors=[5, 10, 15],
                    sampling_strategy=imblearn_ratios
                ),
                'CC': ResamplingCV(
                    CC, clf,
                    sampling_strategy=imblearn_ratios
                ),
                'CNN': ResamplingCV(
                    CNN, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'ENN': ResamplingCV(
                    ENN, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'IHT': ResamplingCV(
                    IHT, clf,
                    sampling_strategy=imblearn_ratios,
                    cv=[2]
                ),
                'NCL': ResamplingCV(
                    NCL, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'NM': ResamplingCV(
                    NM, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'OSS': ResamplingCV(
                    OSS, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'RBO': ResamplingCV(
                    RBO, clf,
                    gamma=[0.01, 0.1, 1.0, 10.0],
                    ratio=[0.5, 0.75, 1.0]
                ),
                'RBU': ResamplingCV(
                    RBU, clf,
                    gamma=params.get('gamma'),
                    ratio=params.get('ratio')
                ),
                'RENN': ResamplingCV(
                    RENN, clf,
                    n_neighbors=[1, 3, 5, 7]
                ),
                'ROS': ResamplingCV(
                    ROS, clf,
                    sampling_strategy=imblearn_ratios
                ),
                'RUS': ResamplingCV(
                    RUS, clf,
                    sampling_strategy=imblearn_ratios
                ),
                'SMOTE': ResamplingCV(
                    SMOTE, clf,
                    k_neighbors=[1, 3, 5, 7, 9],
                    sampling_strategy=imblearn_ratios
                ),
                'SMOTE+ENN': ResamplingCV(
                    SMOTEENN, clf,
                    smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
                    sampling_strategy=imblearn_ratios
                ),
                'SMOTE+TL': ResamplingCV(
                    SMOTETomek, clf,
                    smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
                    sampling_strategy=imblearn_ratios
                ),
                'TL': TL(),
            }

            algorithm = algorithms.get(trial['Algorithm'])

            if algorithm is None:
                raise NotImplementedError

        if algorithm is not None:
            X_train, y_train = algorithm.fit_sample(X_train, y_train)

        clf = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        scores = {
            'Precision': metrics.precision(y_test, predictions),
            'Recall': metrics.recall(y_test, predictions),
            'F-measure': metrics.f_measure(y_test, predictions),
            'AUC': metrics.auc(y_test, predictions),
            'G-mean': metrics.g_mean(y_test, predictions)
        }

        submit_result(trial, scores)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    for _ in range(N_PROCESSES):
        mp.Process(target=run).start()

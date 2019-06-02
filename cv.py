import logging
import numpy as np

from itertools import product
from metrics import f_measure, g_mean, auc
from sklearn.model_selection import StratifiedKFold


class ResamplingCV:
    def __init__(self, algorithm, classifier, metrics=(f_measure, g_mean, auc), n=3, **kwargs):
        self.algorithm = algorithm
        self.classifier = classifier
        self.metrics = metrics
        self.n = n
        self.kwargs = kwargs

    def fit_sample(self, X, y):
        best_score = -np.inf
        best_parameters = None

        parameter_combinations = list((dict(zip(self.kwargs, x)) for x in product(*self.kwargs.values())))

        if len(parameter_combinations) == 1:
            return self.algorithm(**parameter_combinations[0]).fit_sample(X, y)

        for parameters in parameter_combinations:
            scores = []

            for _ in range(self.n):
                skf = StratifiedKFold(n_splits=2, shuffle=True)

                for train_idx, test_idx in skf.split(X, y):
                    try:
                        X_train, y_train = self.algorithm(**parameters).fit_sample(X[train_idx], y[train_idx])
                    except ValueError as e:
                        logging.warning('ValueError "%s" occurred during CV resampling with %s. '
                                        'Setting parameter score to -inf.' % (e, self.algorithm.__name__))

                        scores.append(-np.inf)

                        break
                    else:
                        if len(np.unique(y_train)) < 2:
                            logging.warning('One of the classes was eliminated during CV resampling with %s. '
                                            'Setting parameter score to -inf.' % self.algorithm.__name__)

                            scores.append(-np.inf)

                            break

                        classifier = self.classifier.fit(X_train, y_train)
                        predictions = classifier.predict(X[test_idx])

                        scores.append(np.mean([metric(y[test_idx], predictions) for metric in self.metrics]))

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameters = parameters

        return self.algorithm(**best_parameters).fit_sample(X, y)

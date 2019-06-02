import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets


GAMMAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


for dataset in datasets.names('preliminary'):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            for gamma in GAMMAS:
                trial = {
                    'Algorithm': 'RBU',
                    'Parameters': {
                        'gamma': [gamma],
                        'ratio': RATIOS,
                        'classifier': classifier
                    },
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (gamma)'
                }

                databases.add_to_pending(trial)

            for ratio in RATIOS:
                trial = {
                    'Algorithm': 'RBU',
                    'Parameters': {
                        'gamma': GAMMAS,
                        'ratio': [ratio],
                        'classifier': classifier
                    },
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (ratio)'
                }

                databases.add_to_pending(trial)

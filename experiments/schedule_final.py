import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets


REFERENCE_ALGORITHMS = [
    'RUS', 'AKNN', 'CC', 'CNN', 'ENN', 'IHT', 'NCL', 'NM', 'OSS', 'RENN',
    'TL', 'ROS', 'SMOTE', 'Bord', 'RBO', 'SMOTE+TL', 'SMOTE+ENN'
]


for dataset in datasets.names('final'):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            trial = {
                'Algorithm': 'RBU',
                'Parameters': {
                    'gamma': [0.01, 0.1, 1.0, 10.0],
                    'ratio': [0.5, 0.75, 1.0],
                    'classifier': classifier
                },
                'Dataset': dataset,
                'Fold': fold,
                'Description': 'Final'
            }

            databases.add_to_pending(trial)

            for algorithm in REFERENCE_ALGORITHMS:
                trial = {
                    'Algorithm': algorithm,
                    'Parameters': {
                        'classifier': classifier
                    },
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Final'
                }

                databases.add_to_pending(trial)

import numpy as np

from datasets import load
from rbu import RBU
from sklearn.manifold import TSNE
from tqdm import tqdm


if __name__ == '__main__':
    dataset = load('pima')
    (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    X = TSNE(n_components=2, random_state=42).fit_transform(X)

    gammas = [1.0, 2.5, 5.0, 10.0, 25.0, 1.0]
    ratios = [0.5, 0.5, 0.5, 0.5, 0.5, 0.0]

    for gamma, ratio in tqdm(zip(gammas, ratios), total=len(gammas)):
        RBU(gamma=gamma, ratio=ratio, visualize=True).fit_sample(X, y)

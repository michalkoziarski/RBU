import numpy as np


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point), gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), gamma)

    return result


def generate_possible_directions(n_dimensions, excluded_direction=None):
    possible_directions = []

    for dimension in range(n_dimensions):
        for sign in [-1, 1]:
            if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                possible_directions.append((dimension, sign))

    np.random.shuffle(possible_directions)

    return possible_directions


class RBO:
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, ratio=1.0, minority_class=None):
        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.ratio = ratio
        self.minority_class = minority_class

    def fit_sample(self, X, y):
        classes = np.unique(y)

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]

            minority_class = classes[np.argmin(sizes)]
            majority_class = classes[np.argmax(sizes)]
        else:
            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        n = int(np.round(self.ratio * (len(majority_points) - len(minority_points))))

        appended = []
        sorted_neighbors_indices = None
        considered_minority_points_indices = range(len(minority_points))

        n_synthetic_points_per_minority_object = {i: 0 for i in considered_minority_points_indices}

        for _ in range(n):
            idx = np.random.choice(considered_minority_points_indices)
            n_synthetic_points_per_minority_object[idx] += 1

        for i in considered_minority_points_indices:
            if n_synthetic_points_per_minority_object[i] == 0:
                continue

            point = minority_points[i]

            if self.approximate_potential:
                if sorted_neighbors_indices is None:
                    distance_vector = [distance(point, x) for x in X]
                    distance_vector[i] = -np.inf
                    indices = np.argsort(distance_vector)[:(self.n_nearest_neighbors + 1)]
                else:
                    indices = sorted_neighbors_indices[i][:(self.n_nearest_neighbors + 1)]

                closest_points = X[indices]
                closest_labels = y[indices]
                closest_minority_points = closest_points[closest_labels == minority_class]
                closest_majority_points = closest_points[closest_labels == majority_class]
            else:
                closest_minority_points = minority_points
                closest_majority_points = majority_points

            for _ in range(n_synthetic_points_per_minority_object[i]):
                translation = [0 for _ in range(len(point))]
                translation_history = [translation]
                potential = mutual_class_potential(point, closest_majority_points, closest_minority_points, self.gamma)
                possible_directions = generate_possible_directions(len(point))

                for _ in range(self.n_steps):
                    if len(possible_directions) == 0:
                        break

                    dimension, sign = possible_directions.pop()
                    modified_translation = translation.copy()
                    modified_translation[dimension] += sign * self.step_size
                    modified_potential = mutual_class_potential(point + modified_translation, closest_majority_points,
                                                                closest_minority_points, self.gamma)

                    if np.abs(modified_potential) < np.abs(potential):
                        translation = modified_translation
                        translation_history.append(translation)
                        potential = modified_potential
                        possible_directions = generate_possible_directions(len(point), (dimension, -sign))

                appended.append(point + translation)

        return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])

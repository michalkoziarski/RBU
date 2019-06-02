import numpy as np

from collections import Counter
from pathlib import Path


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma, p_norm):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point, p_norm), gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point, p_norm), gamma)

    return result


class RBU:
    def __init__(self, gamma=0.05, ratio=1.0, p_norm=2, minority_class=None, visualize=False):
        self.gamma = gamma
        self.ratio = ratio
        self.p_norm = p_norm
        self.minority_class = minority_class
        self.visualize = visualize

    def fit_sample(self, X, y):
        X = X.copy()
        y = y.copy()

        if self.minority_class is None:
            minority_class = Counter(y).most_common()[1][0]
            majority_class = Counter(y).most_common()[0][0]
        else:
            classes = np.unique(y)

            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        n = int(np.round(self.ratio * (len(majority_points) - len(minority_points))))

        majority_potentials = np.zeros(len(majority_points))

        for i, point in enumerate(majority_points):
            majority_potentials[i] = mutual_class_potential(
                point, majority_points, minority_points, self.gamma, self.p_norm
            )

        deleted_majority_points = []

        for _ in range(n):
            idx = np.argmax(majority_potentials)

            for i, point in enumerate(majority_points):
                majority_potentials[i] -= rbf(distance(point, majority_points[idx], self.p_norm), self.gamma)

            deleted_majority_points.append(majority_points[idx])

            majority_points = np.delete(majority_points, idx, axis=0)
            majority_potentials = np.delete(majority_potentials, idx)

        deleted_majority_points = np.array(deleted_majority_points)

        if self.visualize:
            self._visualize(
                'gamma_%.2f_ratio_%.2f_before.pdf' % (self.gamma, self.ratio),
                X, majority_points, minority_points, deleted_majority_points
            )

            self._visualize(
                'gamma_%.2f_ratio_%.2f_after.pdf' % (self.gamma, self.ratio),
                X, majority_points, minority_points
            )

        X = np.concatenate([majority_points, minority_points])
        y = np.concatenate([
            np.tile(majority_class, len(majority_points)),
            np.tile(minority_class, len(minority_points))
        ])

        return X, y

    def _visualize(self, file_name, X, majority_points, minority_points, deleted_majority_points=None):
        assert X.shape[1] == 2

        import matplotlib.pyplot as plt

        from matplotlib.colors import LinearSegmentedColormap

        ALPHA = 0.9
        BACKGROUND_COLOR = '#EEEEEE'
        BORDER_COLOR = '#161921'
        COLOR_MAJORITY = '#C44E52'
        COLOR_MINORITY = '#4C72B0'
        COLOR_NEUTRAL = '#F2F2F2'
        FIGURE_SIZE = (6, 6)
        LINE_WIDTH = 1.0
        MARGIN = 0.05
        MARKER_SIZE = 100
        MARKER_SYMBOL = 'o'
        ORIGINAL_EDGE_COLOR = '#F2F2F2'
        OVERSAMPLED_EDGE_COLOR = '#262223'
        POTENTIAL_GRID_N = 250
        VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'

        VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

        plt.style.use('ggplot')

        figure, axis = plt.subplots(figsize=FIGURE_SIZE)

        x_limits = [np.min(X[:, 0]), np.max(X[:, 0])]
        y_limits = [np.min(X[:, 1]), np.max(X[:, 1])]

        x_margin = MARGIN * (x_limits[1] - x_limits[0])
        y_margin = MARGIN * (y_limits[1] - y_limits[0])

        x_limits = [x_limits[0] - x_margin, x_limits[1] + x_margin]
        y_limits = [y_limits[0] - y_margin, y_limits[1] + y_margin]

        plt.xlim(x_limits)
        plt.ylim(y_limits)

        axis.grid(False)

        axis.set_xticks([])
        axis.set_yticks([])

        for key in axis.spines.keys():
            axis.spines[key].set_color(BORDER_COLOR)

        axis.tick_params(colors=BORDER_COLOR)
        axis.set_facecolor(BACKGROUND_COLOR)

        potential_grid = np.zeros((POTENTIAL_GRID_N + 1, POTENTIAL_GRID_N + 1))

        if deleted_majority_points is not None and len(deleted_majority_points) > 0:
            concatenated_majority_points = np.concatenate([majority_points, deleted_majority_points])
        else:
            concatenated_majority_points = majority_points

        for i, x1 in enumerate(np.linspace(x_limits[0], x_limits[1], POTENTIAL_GRID_N + 1)):
            for j, x2 in enumerate(np.linspace(y_limits[0], y_limits[1], POTENTIAL_GRID_N + 1)):
                potential_grid[i][j] = mutual_class_potential(
                    np.array([x1, x2]),
                    concatenated_majority_points,
                    minority_points,
                    self.gamma,
                    self.p_norm
                )

        potential_grid[potential_grid > 0] /= np.max(potential_grid)
        potential_grid[potential_grid < 0] /= -np.min(potential_grid)

        potential_grid = np.swapaxes(potential_grid, 0, 1)

        if deleted_majority_points is not None:
            color_map = LinearSegmentedColormap.from_list(
                'heatmap', (COLOR_MINORITY, COLOR_NEUTRAL, COLOR_MAJORITY), N=100
            )
        else:
            color_map = LinearSegmentedColormap.from_list(
                'blank', (COLOR_NEUTRAL, COLOR_NEUTRAL), N=1
            )

        plt.imshow(
            potential_grid, vmin=potential_grid.min(), vmax=potential_grid.max(),
            extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
            origin='lower', cmap=color_map
        )

        plt.scatter(
            majority_points[:, 0], majority_points[:, 1],
            facecolors=COLOR_MAJORITY, s=MARKER_SIZE,
            marker=MARKER_SYMBOL, linewidths=LINE_WIDTH, alpha=ALPHA,
            edgecolors=ORIGINAL_EDGE_COLOR
        )

        plt.scatter(
            minority_points[:, 0], minority_points[:, 1],
            facecolors=COLOR_MINORITY, s=MARKER_SIZE,
            marker=MARKER_SYMBOL, linewidths=LINE_WIDTH, alpha=ALPHA,
            edgecolors=ORIGINAL_EDGE_COLOR
        )

        if deleted_majority_points is not None and len(deleted_majority_points) > 0:
            plt.scatter(
                deleted_majority_points[:, 0], deleted_majority_points[:, 1],
                facecolors=COLOR_MAJORITY, s=MARKER_SIZE,
                marker=MARKER_SYMBOL, linewidths=LINE_WIDTH, alpha=ALPHA,
                edgecolors=OVERSAMPLED_EDGE_COLOR
            )

        plt.savefig(
            VISUALIZATIONS_PATH / file_name.replace('.', '-').replace('-pdf', '.pdf'),
            bbox_inches='tight'
        )

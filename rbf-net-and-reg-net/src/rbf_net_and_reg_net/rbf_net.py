import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError


class RBFLayer:
    def __init__(self, n_centers: int, random_state: int = 0):
        self._rand = np.random.RandomState(random_state)
        self._n_centers: int = n_centers
        self._centers: np.ndarray = None
        self._sigma: float = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        self._centers = self._calculate_centers(X)
        self._sigma = self._calculate_sigma()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray):
        if self._fitted:
            return RBFLayer._gaussian_rbf(X, self._centers, self._sigma)
        raise NotFittedError

    def _calculate_centers(self, X: np.ndarray):
        indices = self._rand.choice(X.shape[0], self._n_centers, replace=False)
        return X[indices]

    def _calculate_sigma(self):
        return np.max(cdist(self._centers, self._centers, "sqeuclidean")) / np.sqrt(
            2 * self._n_centers
        )

    @staticmethod
    def _gaussian_rbf(X: np.ndarray, centers: np.ndarray, sigma: float):
        return np.exp(-cdist(X, centers, "sqeuclidean") / (2 * sigma**2))


class RBFNetwork:
    def __init__(self, n_neurons: int, random_state: int = 0):
        self.n_neurons = n_neurons
        self.random_state = random_state
        self.layer = None
        self.weights = None
        self.fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.layer = RBFLayer(self.n_neurons, self.random_state).fit(X_train)
        phi: np.ndarray = self.layer.predict(X_train)
        self.weights: np.ndarray = np.linalg.pinv(phi) @ y_train
        self.fitted = True
        return self

    def predict(self, X: np.ndarray):
        if self.fitted:
            phi = self.layer.predict(X)
            y_pred = phi @ self.weights
            return (y_pred >= 0.5).astype(int)
        raise NotFittedError

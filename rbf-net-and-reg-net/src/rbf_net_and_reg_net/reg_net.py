from abc import ABC

import numpy as np
from lab5.logger import logger
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError


class RegularizationLayer:
    def __init__(self, random_state: int = 0):
        self._rand = np.random.RandomState(random_state)
        self._n_centers: int = None
        self._centers: np.ndarray = None
        self._sigma: float = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        self._n_centers = X.shape[0]
        self._centers = self._calculate_centers(X)
        self._sigma = self._calculate_sigma()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray):
        if self._fitted:
            return RegularizationLayer._gaussian_rbf(X, self._centers, self._sigma)
        raise NotFittedError

    def _calculate_centers(self, X: np.ndarray):
        return X

    def _calculate_sigma(self):
        return np.max(cdist(self._centers, self._centers, "sqeuclidean")) / np.sqrt(
            2 * self._n_centers
        )

    @staticmethod
    def _gaussian_rbf(X: np.ndarray, centers: np.ndarray, sigma: float):
        return np.exp(-cdist(X, centers, "sqeuclidean") / (2 * sigma**2))


class BaseNetwork(ABC):
    def __init__(self, random_state: int = 0):
        self.random_state: int = random_state
        self.layer: RegularizationLayer = None
        self.weights: np.ndarray = None
        self.fitted: bool = False

    def predict(self, X: np.ndarray):
        if self.fitted:
            phi = self.layer.predict(X)
            y_pred = phi @ self.weights
            return (y_pred >= 0.5).astype(int)
        raise NotFittedError


class NoRegularizationNetwork(BaseNetwork):
    def __init__(self, random_state: int = 0):
        super().__init__(random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.layer = RegularizationLayer(self.random_state).fit(X_train)
        phi: np.ndarray = self.layer.predict(X_train)
        try:
            self.weights: np.ndarray = np.linalg.inv(phi) @ y_train
            self.fitted = True
            return self
        except np.linalg.LinAlgError as e:
            logger.error(
                f"Matrix phi is singular (determinant = 0), generating the error: {e}.\nThis happens because there's no regularization (lambda = 0)."
            )


class RegularizationNetwork(BaseNetwork):
    def __init__(self, lambda_param: int, random_state: int = 0):
        super().__init__(random_state)
        self.lambda_param = lambda_param

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.layer = RegularizationLayer(self.random_state).fit(X_train)
        G = self.layer.predict(X_train)
        try:
            self.weights = self._calculate_regularized_inverse(G) @ y_train
            self.fitted = True
            return self
        except np.linalg.LinAlgError as e:
            logger.error(
                f"Matrix phi is singular (determinant = 0), generating the error: {e}.\nThis happens because there's no regularization (lambda = 0)."
            )

    def _calculate_regularized_inverse(self, G: np.ndarray):
        return np.linalg.inv(G + self.lambda_param * np.identity(G.shape[0]))

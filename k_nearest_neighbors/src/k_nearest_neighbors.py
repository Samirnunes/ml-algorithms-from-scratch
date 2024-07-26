import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class KNearestNeighbors(ABC):
    def __init__(self, k=5):
        self._k = k
        self._instances = None
        self._target = None
        self._fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._instances = pd.concat([X, y], axis=1)
        self._target = y.columns[0]
        self._fitted = True
        return self
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        return NotImplemented
    
    def _nearest_neighbors(self, instance: pd.DataFrame):
        def to_array(x):
            return np.array(x).reshape(-1)
        def distance(x):
            return KNearestNeighbors.euclidean_distance(to_array(instance), to_array(x))
        distances = self._instances.drop([self._target], axis=1).apply(distance, axis=1)
        nearest_distances = distances.sort_values(ascending=True)[:self._k]
        return np.array(nearest_distances.index)
    
    @staticmethod
    def euclidean_distance(instance1: np.array, instance2: np.array):
        return np.linalg.norm(instance1 - instance2)

class KNearestNeighborsClassifier(KNearestNeighbors):
    def __init__(self, k=5):
        super().__init__(k)
    
    def predict(self, X: pd.DataFrame):
        probability_predictions = self.predict_proba(X)
        predictions = list(map(lambda x: max(x, key=x.get), probability_predictions))
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame):
        assert self._fitted == True
        predictions = []
        for i in X.index:
            instance = X.loc[[i]]
            nearest_neighbors = self._nearest_neighbors(instance)
            labels = self._instances[self._target].loc[nearest_neighbors]
            prediction = dict((labels.value_counts()/len(labels)).sort_index())
            predictions.append(prediction)
        return np.array(predictions)
    
class KNearestNeighborsRegressor(KNearestNeighbors):
    def __init__(self, k=5):
        super().__init__(k) 
        
    def predict(self, X: pd.DataFrame):
        assert self._fitted == True
        predictions = []
        for i in X.index:
            instance = X.loc[[i]]
            nearest_neighbors = self._nearest_neighbors(instance)
            labels = self._instances[self._target].loc[nearest_neighbors].values
            prediction = labels.mean()
            predictions.append(prediction)
        return np.array(predictions)
            
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from sklearn import neighbors

class KNearestNeighbors(ABC):
    def __init__(self, k=5):
        self._k = k
        self._instances = None
        self._target = None
        self._fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._instances = X
        self._labels = y
        self._target = self._labels.columns[0]
        self._fitted = True
        return self
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        raise NotImplementedError("Must be implemented by the subclasses.")
    
    def _nearest_neighbors(self, instance: pd.DataFrame):
        distances = np.linalg.norm(self._instances.values - instance.values, axis=1)
        distances = pd.Series(distances, index=self._instances.index)        
        nearest_distances = distances.sort_values(ascending=True)[:self._k]
        nearest_neighbors = nearest_distances.index
        return np.array(nearest_neighbors)
    
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
            neighbors_labels = self._labels.loc[nearest_neighbors]
            prediction = dict((neighbors_labels.value_counts()/len(neighbors_labels)).sort_index())
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
            neighbors_labels = self._labels.loc[nearest_neighbors]
            prediction = neighbors_labels.values.mean()
            predictions.append(prediction)
        return np.array(predictions)
            
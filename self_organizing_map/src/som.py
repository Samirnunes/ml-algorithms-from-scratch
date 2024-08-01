import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class SOM:
    def __init__(self, nx: int, ny: int, learning_rate=0.1, neighborhood_radius=0.1, decay_ratio=1.001, tol=1e-4, max_iter=500, random_state=0):
        self.__nx = nx
        self.__ny = ny
        self.__learning_rate = learning_rate
        self.__neighborhood_radius = neighborhood_radius
        self.__decay_ratio = decay_ratio
        self.__max_iter = max_iter
        self.__random_state = random_state
        self.__fitted = False
        self.__features = []
        self.__coordinates_cols = ["x", "y"]
        x = np.arange(0, self.__nx, 1)
        y = np.arange(0, self.__ny, 1)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.column_stack([xv.flatten(), yv.flatten()])
        self.map = pd.DataFrame(coordinates, columns=self.__coordinates_cols)
    
    def fit(self, X: pd.DataFrame):
        self.__features = list(X.columns)
        self.__init_neurons(X)
        iter_count = 0
        while True:
            random_state = self.__random_state + iter_count
            rand = np.random.RandomState(random_state)
            index = rand.choice(list(X.index), 1)
            point = X.loc[index].values
            weights = self.map[self.__features]
            distances = np.linalg.norm(weights.values - point, axis=1)
            best_matching_unit = np.argmin(distances)
            self.__update_neurons(point, best_matching_unit)
            self.__update_hyperparameters()
            iter_count += 1
            if np.allclose(self.map[self.__features], weights) or iter_count >= self.__max_iter:
                self.__fitted = True
                return self
            
    def predict(self, X: pd.DataFrame):
        assert self.__fitted == True
        assert list(X.columns) == self.__features
        predictions = np.empty(X.shape[0], dtype=int)
        for i, point in enumerate(X.values):
            weights = self.map[self.__features]
            distances = np.linalg.norm(weights.values - point, axis=1)
            predictions[i] = np.argmin(distances)
        return predictions
    
    def pairplot(self, X: pd.DataFrame):
        assert self.__fitted == True
        assert list(X.columns) == self.__features
        
        neurons = self.predict(X)
        pca = PCA(n_components=2)
        X_pca_with_neuron = pd.DataFrame(pca.fit_transform(X), columns=["pca1", "pca2"], index=X.index)
        X_pca_with_neuron["neuron"] = neurons
        
        fig, axes = plt.subplots(nrows=self.__ny, ncols=self.__nx, figsize=(20, 10), sharex=True, sharey=True)
        for i in range(self.__ny):
            for j in range(self.__nx):
                neuron = self.map[(self.map[self.__coordinates_cols[0]] == j) & (self.map[self.__coordinates_cols[1]] == i)].index[0]
                data = X_pca_with_neuron[X_pca_with_neuron["neuron"] == neuron]
                sns.scatterplot(ax=axes[i][j], data=data, x="pca1", y="pca2", hue="neuron", palette="dark")
        plt.show()
        
    def scatterplot(self, X: pd.DataFrame):
        assert self.__fitted == True
        assert list(X.columns) == self.__features
        
        neurons = self.predict(X)
        pca = PCA(n_components=2)
        X_pca_with_neuron = pd.DataFrame(pca.fit_transform(X), columns=["pca1", "pca2"], index=X.index)
        X_pca_with_neuron["neuron"] = neurons
        sns.scatterplot(X_pca_with_neuron, x="pca1", y="pca2", hue="neuron", palette="dark")
        plt.show()
        
    def feature_hist(self, X_to_predict: pd.DataFrame, X_to_plot: pd.DataFrame, feature: str):
        assert self.__fitted == True
        assert feature in self.__features
        
        fig, axes = plt.subplots(nrows=self.__ny, ncols=self.__nx, figsize=(20, 10), sharex=True, sharey=True)
        X_with_neurons = X_to_plot.copy()
        X_with_neurons["neuron"] = self.predict(X_to_predict)
        for i in range(self.__ny):
            for j in range(self.__nx):
                neuron = self.map[(self.map[self.__coordinates_cols[0]] == j) & (self.map[self.__coordinates_cols[1]] == i)].index[0]
                data = X_with_neurons[X_with_neurons["neuron"] == neuron]
                sns.histplot(ax=axes[i][j], data=data, x=feature, hue="neuron", palette="dark")
        plt.show()
    
    def __init_neurons(self, X: pd.DataFrame):
        rand = np.random.RandomState(self.__random_state)
        neurons = X.loc[rand.choice(X.index, size=len(self.map), replace=False)].reset_index(drop=True)
        self.map = pd.concat([self.map, neurons], axis=1)
        return self.map
    
    def __update_neurons(self, point: np.array, best_matching_unit: int):
        for neuron in self.map.index:
            neuron_weight = self.map.loc[neuron, self.__features]
            neighborhood_func = self.__gaussian_neighborhood_function(neuron, best_matching_unit)
            point_neuron_difference = point.flatten() - neuron_weight.values
            self.map.loc[neuron, self.__features] = neuron_weight + neighborhood_func * self.__learning_rate * point_neuron_difference
    
    def __update_hyperparameters(self):
        self.__learning_rate = self.__learning_rate/self.__decay_ratio
        self.__neighborhood_radius  = self.__neighborhood_radius/self.__decay_ratio
    
    def __gaussian_neighborhood_function(self, neuron: int, best_matching_unit: int):
        neuron_xy = self.map.loc[neuron, self.__coordinates_cols].values
        bmu_xy = self.map.loc[best_matching_unit, self.__coordinates_cols].values
        distance = np.linalg.norm(neuron_xy - bmu_xy)
        return np.exp(-distance/(2*self.__neighborhood_radius**2))
        
import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter=100, tol=1e-4, random_state=0):
        self.__k = k
        self.__max_iter = max_iter
        self.__tol = tol
        self.__random_state = random_state
        self.__fitted = False
        self.centroids = None
        self.clusters = None
    
    def fit(self, X: pd.DataFrame):
        assert self.__k <= len(X)
        centroids = self.__init_centroids(X)
        clusters = self.__assign_clusters(X, centroids)
        iter_count = 0
        while True:
            new_centroids = self.__update_centroids(X, clusters)
            new_clusters = self.__assign_clusters(X, new_centroids)
            iter_count += 1
            if np.allclose(new_centroids, centroids, atol=self.__tol) or iter_count >= self.__max_iter:
                self.centroids = new_centroids
                self.clusters = new_clusters
                self.__fitted = True
                return self
            centroids = new_centroids
            clusters = new_clusters
    
    def predict(self, X: pd.DataFrame):
        assert self.__fitted == True
        return self.__assign_clusters(X, self.centroids)
    
    def __init_centroids(self, X):
        rand = np.random.RandomState(self.__random_state)
        centroids = X.loc[rand.choice(X.index, size=self.__k, replace=False)].reset_index(drop=True)
        return centroids

    def __update_centroids(self, X: pd.DataFrame, clusters: pd.Series):
        X_with_clusters = X.copy()
        X_with_clusters["cluster"] = clusters
        centroids = X_with_clusters.groupby(by="cluster").mean()
        return centroids
    
    def __assign_clusters(self, X: pd.DataFrame, centroids: pd.DataFrame):
        clusters = np.empty(X.shape[0], dtype=int)
        for i, point in enumerate(X.values):
            distances = np.linalg.norm(centroids.values - point, axis=1)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        return pd.Series(clusters, index=X.index)
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
        self.wcss = []
    
    def fit(self, X: pd.DataFrame):
        assert self.__k <= len(X)
        centroids = self.__init_centroids(X)
        clusters = self.__assign_clusters(X, centroids)
        iter_count = 0
        self.wcss.append(self.__within_cluster_sum_of_squares(X, centroids, clusters))
        while True:
            self.centroids = self.__update_centroids(X, clusters)
            self.clusters = self.__assign_clusters(X, self.centroids)
            iter_count += 1
            self.wcss.append(self.__within_cluster_sum_of_squares(X, self.centroids, self.clusters))
            if np.allclose(self.centroids, centroids, atol=self.__tol) or iter_count >= self.__max_iter:
                self.__fitted = True
                return self
            centroids = self.centroids
            clusters = self.clusters
    
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
    
    def __within_cluster_sum_of_squares(self, X: pd.DataFrame, centroids: pd.DataFrame, clusters: pd.Series):
        wcss = 0
        X_with_clusters = X.copy()
        X_with_clusters["cluster"] = clusters
        for cluster in clusters.unique():
            X_cluster = X_with_clusters[X_with_clusters["cluster"] == cluster].drop(["cluster"], axis=1)
            squared_distances = np.linalg.norm(centroids.loc[cluster].values - X_cluster.values, axis=1)**2
            wcss += squared_distances.sum()
        return wcss
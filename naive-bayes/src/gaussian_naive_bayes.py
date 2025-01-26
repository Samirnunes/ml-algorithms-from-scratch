import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    def __init__(self):
        self.__means = None
        self.__stds = None
        self.__class_priors = None
        self.__target = None
        self.__features = None
        self.__fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.__target = y.columns[0]
        self.__features = list(X.columns)
        data = pd.concat([X, y], axis=1)
        grouped = data.groupby(by=self.__target)
        self.__means = grouped.mean()
        self.__stds = grouped.std()
        self.__class_priors = grouped.size()/len(data)
        self.__classes = list(self.__class_priors.index)
        self.__fitted = True
        return self
            
    def predict(self, X: pd.DataFrame):
        assert self.__fitted == True
        assert list(X.columns) == self.__features
        
        def maximum_a_posteriori(posterior_dict):
            return max(posterior_dict, key=posterior_dict.get) # argmax for dict
        
        predictions = list(map(maximum_a_posteriori, self.__posteriors_by_row(X)))
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame):
        assert self.__fitted == True
        assert list(X.columns) == self.__features
        
        def likelihood_to_probability(posterior_dict):
            posteriors_sum = sum(posterior_dict.values())
            for key, val in posterior_dict.items():
                posterior_dict[key] = val/posteriors_sum
            return posterior_dict
        
        predictions = list(map(likelihood_to_probability, self.__posteriors_by_row(X)))
        return np.array(predictions)
    
    def __posteriors_by_row(self, X: pd.DataFrame):
        posteriors_by_class = {}
        posteriors_by_row = []
        for c in self.__classes:
            posteriors_by_class[c] = self.__posterior_by_class(c, X)
        for i in range(len(X)):
            posteriors_by_row.append({key: array[i] for key, array in posteriors_by_class.items()})
        return posteriors_by_row
    
    def __posterior_by_class(self, c: int, X: pd.DataFrame):
        def gaussian_pdf(x, mu, sigma):
            return (1/np.sqrt(2*np.pi*sigma**2))*np.exp((-(x-mu)**2)/(2*sigma**2))
        
        posterior = self.__class_priors[c]
        for feature in X.columns:
            mean = self.__means[feature][c]
            std = self.__stds[feature][c]
            gaussian = np.vectorize(gaussian_pdf, excluded=["mu", "sigma"])
            posterior = posterior * gaussian(np.array(X[feature]), mean, std)
        return posterior
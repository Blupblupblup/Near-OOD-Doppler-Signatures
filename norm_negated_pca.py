import numpy as np

class Norm_Negated_PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def decision_function(self, X):
        return np.sum(X[:,-self.n_components:]**2, axis=1)
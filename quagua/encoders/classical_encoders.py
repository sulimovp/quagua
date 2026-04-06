"""
Classical baseline encoding methods
"""

import numpy as np


class ClassicalXORMixer:
    """Classical XOR mixing for comparison"""
    
    def __init__(self):
        self.n_features = None
        
    def fit(self, X):
        self.n_features = X.shape[1]
        return self
    
    def transform(self, X):
        n_samples = X.shape[0]
        # Detect if data is float or integer
        is_float = np.issubdtype(X.dtype, np.floating) or np.any(X != X.astype(int))
        
        mixed_X = np.zeros((n_samples, self.n_features), dtype=X.dtype)
        
        if is_float:
            # For float data, use additive mixing instead of XOR
            # Normalize first to prevent overflow
            X_norm = X.copy()
            if X_norm.max() > 1.0 or X_norm.min() < 0.0:
                # Normalize to [0, 1] if needed
                X_min = X_norm.min(axis=0, keepdims=True)
                X_max = X_norm.max(axis=0, keepdims=True)
                X_range = X_max - X_min + 1e-8
                X_norm = (X_norm - X_min) / X_range
            
            for i in range(n_samples):
                for j in range(self.n_features):
                    next_j = (j + 1) % self.n_features
                    # Additive mixing with modulo to keep values bounded
                    mixed_X[i, j] = (X_norm[i, j] + X_norm[i, next_j]) % 1.0
        else:
            # For integer data, use XOR
            for i in range(n_samples):
                for j in range(self.n_features):
                    next_j = (j + 1) % self.n_features
                    mixed_X[i, j] = int(X[i, j]) ^ int(X[i, next_j])
        
        return mixed_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RandomProjectionEncoder:
    """Random projection (classical baseline)"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, X):
        n_features = X.shape[1]
        if self.n_components is None:
            self.n_components = n_features * 2
        
        # Random projection matrix
        self.projection = np.random.randn(n_features, self.n_components)
        
        return self
    
    def transform(self, X):
        return X @ self.projection
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)




"""
Random Walk Feature Encoder

Implements a privacy-preserving encoding method based on random walks through features.
The idea is to apply linear transformations within the feature vector itself, creating
a transformation that is reversible by a linear first layer in a neural network.

Key properties:
- Linear transformations preserve relationships for ML models
- Random walk order and weights are secret (provide privacy)
- First layer of NN should be fully-connected and linear to learn inverse
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from scipy.stats import uniform, beta, norm


class RandomWalkFeatureEncoder:
    """
    Random Walk Feature Encoder
    
    Applies linear transformations through a random walk over features:
    - Starts at a random feature Xn
    - Jumps to feature Xk, transforms: Xk = a*Xk + b*Xn
    - Continues walking, applying transformations
    - Ends at Xn, transforming it
    
    The walk order and transformation weights are secret, providing privacy.
    A linear first layer in NN can learn to invert the transformation.
    
    Parameters:
    -----------
    n_steps : int
        Number of steps in the random walk (default: 2 * n_features)
    weight_distribution : str
        Distribution for sampling weights: 'uniform', 'beta', 'normal' (default: 'beta')
    weight_params : dict
        Parameters for weight distribution (default: {'a': 0.5, 'b': 0.5} for beta)
    walk_strategy : str
        Strategy for choosing next feature: 'random', 'sequential', 'weighted' (default: 'random')
    binary_handling : str
        How to handle binary features: 'preserve', 'normalize', 'skip' (default: 'preserve')
    output_noise : float
        Std of Gaussian noise added to outputs (default: 0). >0 makes linear
        inversion harder; may slightly reduce utility.
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_steps: Optional[int] = None,
        weight_distribution: str = 'beta',
        weight_params: Optional[Dict] = None,
        walk_strategy: str = 'random',
        binary_handling: str = 'preserve',
        output_noise: float = 0.0,
        seed: Optional[int] = None
    ):
        self.n_steps = n_steps
        self.weight_distribution = weight_distribution
        self.weight_params = weight_params or {'a': 0.5, 'b': 0.5}
        self.walk_strategy = walk_strategy
        self.binary_handling = binary_handling
        self.output_noise = float(output_noise) if output_noise is not None else 0.0
        self.seed = seed
        
        # Will be set during fit
        self.n_features_ = None
        self.binary_features_ = None
        self.walk_path_ = None
        self.transformation_weights_ = None
        self.feature_order_ = None
        
    def _identify_binary_features(self, X: np.ndarray) -> List[int]:
        """
        Identify truly binary features (exactly 2 unique values).
        
        Only features with n_unique==2 are treated as binary. Categorical features
        (e.g. Pclass with 3 values, Embarked with 3 values) are not classified
        as binary and receive the full random-walk transformation.
        
        This avoids over-broad 'binary' detection that would leave too many
        features untransformed when using binary_handling='skip'.
        """
        binary_features = []
        for i in range(X.shape[1]):
            n_unique = len(np.unique(X[:, i]))
            if n_unique == 2:
                binary_features.append(i)
        return binary_features
    
    def _sample_weight(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Sample transformation weights (a, b) from specified distribution"""
        if self.weight_distribution == 'uniform':
            a = rng.uniform(0.1, 0.9)
            b = rng.uniform(0.1, 0.9)
        elif self.weight_distribution == 'beta':
            a = rng.beta(self.weight_params.get('a', 0.5), self.weight_params.get('b', 0.5))
            b = rng.beta(self.weight_params.get('a', 0.5), self.weight_params.get('b', 0.5))
            # Normalize to ensure reasonable scale
            a = 0.1 + 0.8 * a  # Scale to [0.1, 0.9]
            b = 0.1 + 0.8 * b
        elif self.weight_distribution == 'normal':
            a = abs(rng.normal(0.5, 0.2))
            b = abs(rng.normal(0.5, 0.2))
            a = np.clip(a, 0.1, 0.9)
            b = np.clip(b, 0.1, 0.9)
        else:
            raise ValueError(f"Unknown weight_distribution: {self.weight_distribution}")
        
        return a, b
    
    def _choose_next_feature(
        self,
        current_idx: int,
        visited: set,
        n_features: int,
        rng: np.random.Generator,
        feature_importance: Optional[np.ndarray] = None
    ) -> int:
        """Choose next feature in the walk based on strategy"""
        if self.walk_strategy == 'random':
            # Random choice from unvisited features (or all if all visited)
            available = [i for i in range(n_features) if i not in visited]
            if not available:
                available = list(range(n_features))
            return rng.choice(available)
        
        elif self.walk_strategy == 'sequential':
            # Sequential walk (with wrap-around)
            next_idx = (current_idx + 1) % n_features
            return next_idx
        
        elif self.walk_strategy == 'weighted':
            # Weighted by feature importance or variance
            if feature_importance is not None:
                # Use importance weights
                probs = feature_importance.copy()
                probs[list(visited)] = 0  # Don't revisit if possible
                if probs.sum() == 0:
                    probs = np.ones(n_features) / n_features
                else:
                    probs = probs / probs.sum()
                return rng.choice(n_features, p=probs)
            else:
                # Fallback to random
                return self._choose_next_feature(current_idx, visited, n_features, rng, None)
        
        else:
            raise ValueError(f"Unknown walk_strategy: {self.walk_strategy}")
    
    def _generate_walk_path(self, n_features: int, rng: np.random.Generator) -> List[Tuple[int, int, float, float]]:
        """
        Generate random walk path through features
        
        Returns:
        -------
        walk_path : List[Tuple[int, int, float, float]]
            Each tuple is (from_idx, to_idx, weight_a, weight_b)
            Transformation: X[to_idx] = a * X[to_idx] + b * X[from_idx]
        """
        if self.n_steps is None:
            n_steps = 2 * n_features  # Default: 2x number of features
        else:
            n_steps = self.n_steps
        
        # Start at random feature
        start_idx = rng.integers(0, n_features)
        current_idx = start_idx
        visited = {start_idx}
        
        walk_path = []
        
        for step in range(n_steps):
            # Choose next feature
            next_idx = self._choose_next_feature(
                current_idx, visited, n_features, rng
            )
            
            # Sample transformation weights
            a, b = self._sample_weight(rng)
            
            # Record transformation: X[next_idx] = a * X[next_idx] + b * X[current_idx]
            walk_path.append((current_idx, next_idx, a, b))
            
            # Update state
            current_idx = next_idx
            visited.add(next_idx)
        
        # End at starting feature (circular walk)
        if current_idx != start_idx:
            a, b = self._sample_weight(rng)
            walk_path.append((current_idx, start_idx, a, b))
        
        return walk_path
    
    def fit(self, X: np.ndarray) -> 'RandomWalkFeatureEncoder':
        """
        Fit the encoder (identify binary features, generate walk path)
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : RandomWalkFeatureEncoder
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Identify binary features
        self.binary_features_ = self._identify_binary_features(X)
        
        # Set default n_steps if not provided
        if self.n_steps is None:
            self.n_steps = 2 * n_features
        
        # Generate random walk path
        rng = np.random.Generator(np.random.PCG64(self.seed))
        self.walk_path_ = self._generate_walk_path(n_features, rng)
        
        # RNG for output noise (separate seed so it doesn't affect the walk)
        if self.output_noise > 0:
            self._rng_noise = np.random.Generator(np.random.PCG64((self.seed or 0) + 999))
        
        # Store transformation weights for analysis
        self.transformation_weights_ = self.walk_path_
        
        return self
    
    def _apply_binary_handling(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Apply special handling for binary features"""
        if feature_idx not in self.binary_features_:
            return X[:, feature_idx]
        
        if self.binary_handling == 'preserve':
            # Keep binary values as-is (0/1)
            return X[:, feature_idx]
        elif self.binary_handling == 'normalize':
            # Normalize to [0, 1] if not already
            feature = X[:, feature_idx]
            if feature.min() < 0 or feature.max() > 1:
                feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            return feature
        elif self.binary_handling == 'skip':
            # Skip binary features in transformation (return as-is)
            return X[:, feature_idx]
        else:
            raise ValueError(f"Unknown binary_handling: {self.binary_handling}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply random walk transformation to data
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : np.ndarray, shape (n_samples, n_features)
            Transformed data
            
        Note:
        -----
        Transformations are applied sequentially, where each transformation
        uses the CURRENT state of features (after previous transformations).
        This creates a chain of dependencies that makes inversion harder.
        """
        if self.walk_path_ is None:
            raise ValueError("Must call fit() before transform()")
        
        X_transformed = X.copy()
        
        # Apply transformations in walk order (sequentially, using current state)
        for from_idx, to_idx, a, b in self.walk_path_:
            # Handle binary features specially
            if self.binary_handling == 'skip' and to_idx in self.binary_features_:
                # Skip transformation for binary features
                continue
            
            # Get current state of features (after previous transformations)
            from_feature = X_transformed[:, from_idx]
            to_feature = X_transformed[:, to_idx]
            
            # Apply binary handling if needed
            if from_idx in self.binary_features_:
                from_feature = self._apply_binary_handling(X_transformed, from_idx)
            if to_idx in self.binary_features_:
                to_feature = self._apply_binary_handling(X_transformed, to_idx)
            
            # Apply transformation: X[to_idx] = a * X[to_idx] + b * X[from_idx]
            # This uses the CURRENT state (after all previous transformations)
            new_value = a * to_feature + b * from_feature
            
            # For binary features, clip to preserve binary nature
            if to_idx in self.binary_features_ and self.binary_handling == 'preserve':
                new_value = np.clip(new_value, 0.0, 1.0)
            
            # Update the feature (this affects subsequent transformations)
            X_transformed[:, to_idx] = new_value
        
        # Optional output noise (hardens against linear inversion)
        if self.output_noise > 0 and hasattr(self, '_rng_noise'):
            noise = self._rng_noise.normal(0, self.output_noise, X_transformed.shape)
            X_transformed = X_transformed + noise
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit encoder and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def get_walk_info(self) -> Dict:
        """Get information about the walk for analysis"""
        if self.walk_path_ is None:
            return {}
        
        return {
            'n_steps': len(self.walk_path_),
            'n_features': self.n_features_,
            'binary_features': self.binary_features_,
            'walk_path': self.walk_path_,
            'weight_distribution': self.weight_distribution,
            'walk_strategy': self.walk_strategy,
            'output_noise': getattr(self, 'output_noise', 0.0),
        }

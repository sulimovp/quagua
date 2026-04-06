"""
Irreversible encoding methods for binary/categorical data
"""

import numpy as np


class RandomOneToManyEncoder:
    """
    Random 1-to-Many encoding for binary/categorical data
    Each value maps to multiple random encodings (like dropout but for encoding)
    Inspired by: Irreversible encoding for LLM fine-tuning
    """
    
    def __init__(self, n_encodings_per_value=5, encoding_dim=3):
        """
        n_encodings_per_value: How many random encodings per unique value
        encoding_dim: Dimension of each encoding vector
        """
        self.n_encodings_per_value = n_encodings_per_value
        self.encoding_dim = encoding_dim
        self.encoding_maps = {}  # feature_idx -> {value: [encodings]}
        
    def fit(self, X):
        """Learn random encodings for each unique value in each feature"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            encoding_map = {}
            
            for value in unique_values:
                # Create multiple random encodings for this value
                encodings = []
                for _ in range(self.n_encodings_per_value):
                    # Random encoding vector
                    encoding = np.random.randn(self.encoding_dim)
                    # Normalize to unit vector
                    encoding = encoding / (np.linalg.norm(encoding) + 1e-8)
                    encodings.append(encoding)
                
                encoding_map[value] = np.array(encodings)
            
            self.encoding_maps[feature_idx] = encoding_map
        
        return self
    
    def transform(self, X):
        """Transform: randomly select one encoding for each value"""
        n_samples = X.shape[0]
        output_dim = self.n_features * self.encoding_dim
        encoded_X = np.zeros((n_samples, output_dim))
        
        for i in range(n_samples):
            encoded_sample = []
            
            for feature_idx in range(self.n_features):
                value = X[i, feature_idx]
                # Get all encodings for this value
                encodings = self.encoding_maps[feature_idx][value]
                
                # Randomly select one encoding (1-to-Many: same value can map to different encodings)
                selected_encoding = encodings[np.random.randint(len(encodings))]
                encoded_sample.extend(selected_encoding)
            
            encoded_X[i] = np.array(encoded_sample)
        
        # Validate output
        encoded_X = np.nan_to_num(encoded_X, nan=0.0, posinf=1.0, neginf=0.0)
        encoded_X = np.clip(encoded_X, -1e6, 1e6)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ChaoticSystemEncoder:
    """
    Use chaotic systems for irreversible encoding
    Chaotic maps: Logistic, Henon, Tent map
    Small changes in input → large changes in output (butterfly effect)
    """
    
    def __init__(self, map_type='logistic', n_iterations=10):
        """
        map_type: 'logistic', 'henon', 'tent'
        n_iterations: Number of chaotic iterations
        """
        self.map_type = map_type
        self.n_iterations = n_iterations
        self.chaos_params = {}  # Store parameters for each feature
        
    def _logistic_map(self, x, r=3.8):
        """Logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
        return r * x * (1 - x)
    
    def _henon_map(self, x, y, a=1.4, b=0.3):
        """Henon map: 2D chaotic system"""
        x_new = 1 - a * x**2 + y
        y_new = b * x
        return x_new, y_new
    
    def _tent_map(self, x, mu=2.0):
        """Tent map: x_{n+1} = mu * min(x_n, 1 - x_n)"""
        return mu * np.minimum(x, 1 - x)
    
    def fit(self, X):
        """Setup chaotic parameters for each feature"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Normalize features to [0, 1] for chaotic maps
        self.feature_mins = X.min(axis=0)
        self.feature_maxs = X.max(axis=0)
        self.feature_ranges = self.feature_maxs - self.feature_mins + 1e-8
        
        # Store random seeds/offsets for each feature (makes it harder to reverse)
        self.chaos_seeds = np.random.rand(n_features)
        self.chaos_offsets = np.random.rand(n_features) * 0.1
        
        return self
    
    def transform(self, X):
        """Apply chaotic transformation"""
        n_samples = X.shape[0]
        encoded_X = np.zeros((n_samples, self.n_features))
        
        # Normalize to [0, 1]
        X_norm = (X - self.feature_mins) / self.feature_ranges
        # Clip to ensure [0, 1]
        X_norm = np.clip(X_norm, 0.0, 1.0)
        
        for i in range(n_samples):
            encoded_sample = []
            
            for feature_idx in range(self.n_features):
                value = X_norm[i, feature_idx]
                
                # Add random offset (makes reversal harder)
                x = (value + self.chaos_offsets[feature_idx]) % 1.0
                # Ensure x is in [0, 1]
                x = np.clip(x, 0.0, 1.0)
                
                # Apply chaotic map iterations
                if self.map_type == 'logistic':
                    for _ in range(self.n_iterations):
                        x = self._logistic_map(x, r=3.8 + self.chaos_seeds[feature_idx])
                        # Clip to prevent overflow
                        x = np.clip(x, 0.0, 1.0)
                    encoded_sample.append(x)
                    
                elif self.map_type == 'henon':
                    y = self.chaos_seeds[feature_idx]
                    for _ in range(self.n_iterations):
                        x, y = self._henon_map(x, y)
                        # Clip to prevent overflow (Henon can produce large values)
                        x = np.clip(x, -10.0, 10.0)
                        y = np.clip(y, -10.0, 10.0)
                    # Normalize x back to [0, 1] for output
                    x = (x + 10.0) / 20.0  # Map [-10, 10] to [0, 1]
                    x = np.clip(x, 0.0, 1.0)
                    encoded_sample.append(x)
                    
                elif self.map_type == 'tent':
                    for _ in range(self.n_iterations):
                        x = self._tent_map(x, mu=2.0 + self.chaos_seeds[feature_idx])
                        # Tent map can exceed [0, 1], so clip
                        x = np.clip(x, 0.0, 1.0)
                    encoded_sample.append(x)
            
            encoded_X[i] = np.array(encoded_sample)
        
        # Final check: replace any inf or nan with 0
        encoded_X = np.nan_to_num(encoded_X, nan=0.0, posinf=1.0, neginf=0.0)
        # Clip to reasonable range
        encoded_X = np.clip(encoded_X, -1e6, 1e6)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class OrthogonalSubspaceEncoder:
    """
    Project features into orthogonal subspaces
    Each feature gets its own orthogonal subspace
    Inspired by: Fisher's Linear Discriminant, subspace clustering
    """
    
    def __init__(self, subspace_dim=3):
        """
        subspace_dim: Dimension of each feature's subspace
        """
        self.subspace_dim = subspace_dim
        
    def fit(self, X):
        """Learn orthogonal subspaces for each feature"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Create orthogonal basis for each feature
        self.subspaces = []
        
        for feature_idx in range(n_features):
            # Generate random orthogonal basis using QR decomposition
            random_matrix = np.random.randn(self.subspace_dim, self.subspace_dim)
            Q, _ = np.linalg.qr(random_matrix)
            self.subspaces.append(Q)
        
        # Store feature statistics for normalization
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        
        return self
    
    def transform(self, X):
        """Project features into orthogonal subspaces"""
        n_samples = X.shape[0]
        output_dim = self.n_features * self.subspace_dim
        encoded_X = np.zeros((n_samples, output_dim))
        
        # Normalize features
        X_norm = (X - self.feature_means) / self.feature_stds
        
        for i in range(n_samples):
            encoded_sample = []
            
            for feature_idx in range(self.n_features):
                value = X_norm[i, feature_idx]
                
                # Project value into orthogonal subspace
                # Use value as weight for first basis vector, add noise for others
                projection = np.zeros(self.subspace_dim)
                projection[0] = value
                
                # Add small random components to other dimensions (non-linear mixing)
                for dim in range(1, self.subspace_dim):
                    projection[dim] = np.random.randn() * 0.1 * abs(value)
                
                # Project onto orthogonal basis
                encoded_vector = self.subspaces[feature_idx] @ projection
                encoded_sample.extend(encoded_vector)
            
            encoded_X[i] = np.array(encoded_sample)
        
        # Validate output
        encoded_X = np.nan_to_num(encoded_X, nan=0.0, posinf=1.0, neginf=0.0)
        encoded_X = np.clip(encoded_X, -1e6, 1e6)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FastPrivacyEncoder:
    """
    Fast Privacy-Preserving Encoder (Inspired by FHE, but much faster)
    
    Design principles:
    1. One-way encoding (irreversible, no decoding needed)
    2. Fast computation (hash-based lookup, polynomial expansion)
    3. Works on binary/categorical/discrete data (continuous is bonus)
    4. Preserves ML utility (models can still learn from encoded data)
    
    Approach:
    - Hash-based lookup tables for discrete values (fast, one-way)
    - Polynomial feature expansion (preserves relationships for ML)
    - Random noise injection (adds privacy)
    - Feature mixing (non-linear combinations)
    
    Similar to FHE in spirit (computation on encrypted data) but:
    - No decryption needed (one-way transformation)
    - Much faster (hash tables vs cryptographic operations)
    - Optimized for ML utility (preserves statistical properties)
    """
    
    def __init__(self, encoding_dim=4, polynomial_degree=2, noise_level=0.1, 
                 use_mixing=True, hash_seed=None):
        """
        encoding_dim: Dimension of hash-based encoding per feature
        polynomial_degree: Degree of polynomial feature expansion (1=linear, 2=quadratic, etc.)
        noise_level: Amount of noise to add (0-1, higher = more privacy, less utility)
        use_mixing: Whether to mix features together (adds privacy)
        hash_seed: Random seed for hash functions (None = random each time)
        """
        self.encoding_dim = encoding_dim
        self.polynomial_degree = polynomial_degree
        self.noise_level = noise_level
        self.use_mixing = use_mixing
        self.hash_seed = hash_seed if hash_seed is not None else np.random.randint(0, 2**31)
        
    def fit(self, X):
        """Setup encoding parameters"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Detect data types (binary, categorical, continuous)
        self.feature_types = []
        self.discrete_features = []  # Features that are discrete (binary/categorical)
        self.continuous_features = []  # Features that are continuous
        
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            n_unique = len(unique_values)
            
            # Heuristic: if few unique values compared to samples, likely categorical
            is_discrete = n_unique < min(20, n_samples * 0.1) or n_unique < 10
            
            if is_discrete:
                self.feature_types.append('discrete')
                self.discrete_features.append(feature_idx)
            else:
                self.feature_types.append('continuous')
                self.continuous_features.append(feature_idx)
        
        # Create hash-based lookup tables for discrete features
        self.hash_tables = {}
        self.hash_seeds = {}
        
        for feature_idx in self.discrete_features:
            unique_values = np.unique(X[:, feature_idx])
            
            # Create random hash seeds for this feature
            np.random.seed(self.hash_seed + feature_idx)
            self.hash_seeds[feature_idx] = np.random.randint(0, 2**31, size=len(unique_values))
            
            # Create hash table: value -> random encoding vector
            hash_table = {}
            for val_idx, value in enumerate(unique_values):
                # Hash-based encoding: deterministic but looks random
                np.random.seed(self.hash_seeds[feature_idx][val_idx])
                # Create encoding vector using hash
                encoding = np.random.randn(self.encoding_dim)
                # Normalize to unit vector
                encoding = encoding / (np.linalg.norm(encoding) + 1e-8)
                hash_table[value] = encoding
            
            self.hash_tables[feature_idx] = hash_table
        
        # For continuous features, use quantization + hash
        # Store statistics for normalization
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        
        # Create mixing matrix (if mixing enabled)
        if self.use_mixing:
            np.random.seed(self.hash_seed)
            # Random mixing matrix
            self.mixing_matrix = np.random.randn(n_features, n_features)
            # Make it orthogonal-ish (normalize columns)
            self.mixing_matrix = self.mixing_matrix / (np.linalg.norm(self.mixing_matrix, axis=0, keepdims=True) + 1e-8)
        else:
            self.mixing_matrix = np.eye(n_features)
        
        return self
    
    def _hash_encode_discrete(self, value, feature_idx):
        """Hash-based encoding for discrete values"""
        if feature_idx in self.hash_tables:
            hash_table = self.hash_tables[feature_idx]
            # Round to nearest value in hash table (handles floating point issues)
            if value in hash_table:
                return hash_table[value]
            else:
                # If value not in hash table (e.g., new value), use nearest
                unique_vals = list(hash_table.keys())
                nearest_val = min(unique_vals, key=lambda x: abs(x - value))
                return hash_table[nearest_val]
        else:
            # Shouldn't happen, but fallback
            return np.random.randn(self.encoding_dim)
    
    def _hash_encode_continuous(self, value, feature_idx):
        """Hash-based encoding for continuous values (via quantization)"""
        # Normalize value
        normalized = (value - self.feature_means[feature_idx]) / self.feature_stds[feature_idx]
        
        # Quantize to discrete levels (for hash-based encoding)
        n_bins = 50  # Number of quantization levels
        quantized = int(np.clip((normalized + 3) * n_bins / 6, 0, n_bins - 1))
        
        # Hash the quantized value
        np.random.seed(self.hash_seed + feature_idx * 1000 + quantized)
        encoding = np.random.randn(self.encoding_dim)
        encoding = encoding / (np.linalg.norm(encoding) + 1e-8)
        
        return encoding
    
    def transform(self, X):
        """Apply fast privacy-preserving encoding"""
        n_samples = X.shape[0]
        
        # Step 1: Hash-based encoding (one-way transformation)
        # Expand each feature to encoding_dim dimensions
        encoded_parts = []
        
        for feature_idx in range(self.n_features):
            feature_encoded = np.zeros((n_samples, self.encoding_dim))
            
            for i in range(n_samples):
                value = X[i, feature_idx]
                
                if feature_idx in self.discrete_features:
                    # Discrete: use hash table
                    feature_encoded[i] = self._hash_encode_discrete(value, feature_idx)
                else:
                    # Continuous: quantize + hash
                    feature_encoded[i] = self._hash_encode_continuous(value, feature_idx)
            
            encoded_parts.append(feature_encoded)
        
        # Concatenate all encoded features
        X_hash_encoded = np.hstack(encoded_parts)  # Shape: (n_samples, n_features * encoding_dim)
        
        # Step 2: Polynomial feature expansion (preserves relationships for ML)
        if self.polynomial_degree > 1:
            # Create polynomial features from hash-encoded features
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_hash_encoded)
        else:
            X_poly = X_hash_encoded
        
        # Step 3: Feature mixing (non-linear combinations)
        if self.use_mixing and X_poly.shape[1] >= self.n_features:
            # Apply mixing to polynomial features
            # Take first n_features dimensions and mix
            X_mixed = X_poly[:, :self.n_features] @ self.mixing_matrix
            # Concatenate with remaining polynomial features
            if X_poly.shape[1] > self.n_features:
                X_final = np.hstack([X_mixed, X_poly[:, self.n_features:]])
            else:
                X_final = X_mixed
        else:
            X_final = X_poly
        
        # Step 4: Add noise for privacy (preserves utility through expectation)
        if self.noise_level > 0:
            noise = np.random.randn(*X_final.shape) * self.noise_level
            X_final = X_final + noise
        
        # Normalize to prevent extreme values
        X_final = np.nan_to_num(X_final, nan=0.0, posinf=1.0, neginf=0.0)
        X_final = np.clip(X_final, -10.0, 10.0)
        
        return X_final
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


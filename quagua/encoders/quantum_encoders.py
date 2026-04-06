"""
Quantum-inspired encoding methods
"""

import numpy as np


class QuantumSuperpositionEncoder:
    """
    Encode features as quantum superposition states
    Each value becomes a probability distribution over multiple states
    """
    
    def __init__(self, n_states=4, temperature=1.0):
        """
        n_states: Number of superposition states per feature
        temperature: Controls uncertainty (higher = more spread)
        """
        self.n_states = n_states
        self.temperature = temperature
        self.encoding_maps = {}  # For each feature, map value to distribution
        
    def fit(self, X):
        """Learn encoding distributions for each feature"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            n_unique = len(unique_values)
            
            # Create encoding distributions for each unique value
            encoding_map = {}
            
            for value in unique_values:
                # Create probability distribution centered around encoded position
                encoded_pos = self._value_to_position(value, n_unique)
                
                # Gaussian-like distribution centered at encoded_pos
                positions = np.arange(self.n_states)
                distances = np.abs(positions - encoded_pos)
                
                # Boltzmann distribution: p_i ∝ exp(-distance/temperature)
                logits = -distances / self.temperature
                probs = np.exp(logits - np.max(logits))  # Numerical stability
                probs = probs / probs.sum()
                
                # Store as amplitudes (square root for quantum analogy)
                amplitudes = np.sqrt(probs)
                encoding_map[value] = amplitudes
            
            self.encoding_maps[feature_idx] = encoding_map
        
        return self
    
    def _value_to_position(self, value, n_unique):
        """Map original value to position in superposition space"""
        # Linear mapping from [min, max] to [0, n_states-1]
        return int((value / max(1, n_unique - 1)) * (self.n_states - 1))
    
    def transform(self, X):
        """Transform data using superposition encoding"""
        n_samples = X.shape[0]
        encoded_X = np.zeros((n_samples, self.n_features * self.n_states))
        
        for i in range(n_samples):
            encoded_sample = []
            for feature_idx in range(self.n_features):
                value = X[i, feature_idx]
                amplitudes = self.encoding_maps[feature_idx][value]
                encoded_sample.extend(amplitudes)
            
            encoded_X[i] = np.array(encoded_sample)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantumEntanglementEncoder:
    """
    Encode features using quantum entanglement principles
    Features become non-separable (like Bell states)
    """
    
    def __init__(self, entanglement_strength=0.5):
        """
        entanglement_strength: How strongly to entangle features (0-1)
        """
        self.entanglement_strength = entanglement_strength
        
    def fit(self, X):
        """Setup entanglement patterns"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Create random entanglement matrix (like a Hamiltonian)
        # Each entry defines entanglement strength between feature pairs
        self.entanglement_matrix = np.random.randn(n_features, n_features)
        
        # Make symmetric (undirected entanglement)
        self.entanglement_matrix = (self.entanglement_matrix + 
                                   self.entanglement_matrix.T) / 2
        
        # Scale by entanglement strength
        self.entanglement_matrix *= self.entanglement_strength
        
        # Store feature statistics for normalization
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        
        return self
    
    def transform(self, X):
        """Apply entanglement transformation"""
        n_samples = X.shape[0]
        
        # Normalize features
        X_norm = (X - self.feature_means) / self.feature_stds
        
        # Create entangled representation
        # Inspired by quantum many-body systems
        encoded_X = np.zeros((n_samples, self.n_features))
        
        for i in range(n_samples):
            # Initial state (features as independent)
            state = X_norm[i].copy()
            
            # Apply entanglement (non-linear mixing)
            # This simulates quantum interaction between features
            
            # First-order entanglement (linear mixing)
            entangled = self.entanglement_matrix @ state
            
            # Second-order entanglement (non-linear)
            # Use tensor product-like interaction (simplified)
            for j in range(self.n_features):
                for k in range(j+1, self.n_features):
                    # Bell-state like correlation: x_j * x_k
                    correlation = state[j] * state[k] * self.entanglement_strength
                    entangled[j] += correlation
                    entangled[k] += correlation
            
            # Add quantum noise (Heisenberg uncertainty)
            noise = np.random.randn(self.n_features) * 0.1 * self.entanglement_strength
            entangled += noise
            
            encoded_X[i] = entangled
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantumFourierEncoder:
    """
    Use quantum Fourier transform principles for encoding
    Creates frequency-domain representations with phase information
    """
    
    def __init__(self, n_frequencies=3, use_phases=True):
        """
        n_frequencies: Number of frequency components to use
        use_phases: Whether to include phase information (complex -> real+imag)
        """
        self.n_frequencies = n_frequencies
        self.use_phases = use_phases
        
    def fit(self, X):
        """Setup frequency bases"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Create random frequency bases (like different measurement bases)
        self.frequency_bases = []
        for _ in range(self.n_frequencies):
            basis = np.random.randn(n_features, n_features)
            # Orthonormalize (like quantum measurement bases)
            Q, _ = np.linalg.qr(basis)
            self.frequency_bases.append(Q)
        
        return self
    
    def transform(self, X):
        """Apply quantum Fourier-like transform"""
        n_samples = X.shape[0]
        
        if self.use_phases:
            # Each frequency gives 2 components (real + imaginary)
            output_dim = self.n_features * self.n_frequencies * 2
        else:
            # Only magnitude
            output_dim = self.n_features * self.n_frequencies
        
        encoded_X = np.zeros((n_samples, output_dim))
        
        for i in range(n_samples):
            encoded_sample = []
            
            for freq_idx, basis in enumerate(self.frequency_bases):
                # Transform to frequency basis (like quantum Fourier transform)
                transformed = basis @ X[i]
                
                if self.use_phases:
                    # Convert to polar coordinates (amplitude and phase)
                    amplitudes = np.abs(transformed)
                    phases = np.angle(transformed)  # In radians
                    
                    # Store both amplitude and phase
                    encoded_sample.extend(amplitudes)
                    encoded_sample.extend(phases)
                else:
                    # Just use magnitude (like measurement probability)
                    probabilities = transformed ** 2
                    probabilities = probabilities / (probabilities.sum() + 1e-8)
                    encoded_sample.extend(probabilities)
            
            encoded_X[i] = np.array(encoded_sample)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantumPhaseEncoder:
    """
    Encode features as quantum phases (complex exponentials)
    Strong privacy due to phase wrapping ambiguity
    """
    
    def __init__(self, n_phases=3, add_noise=True, noise_level=0.02):
        """
        n_phases: Number of phase encodings per feature
        add_noise: Add quantum measurement noise
        noise_level: Level of noise to add (reduced from 0.05 for better accuracy)
        """
        self.n_phases = n_phases
        self.add_noise = add_noise
        self.noise_level = noise_level
        
    def fit(self, X):
        """Setup phase encoding parameters"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Store min/max for each feature for normalization
        self.feature_mins = X.min(axis=0)
        self.feature_maxs = X.max(axis=0)
        self.feature_ranges = self.feature_maxs - self.feature_mins + 1e-8
        
        # Create random phase shifts for each encoding
        self.phase_shifts = np.random.rand(n_features, self.n_phases) * 2 * np.pi
        
        # Create random amplitudes for superposition
        self.amplitudes = np.random.rand(n_features, self.n_phases)
        # Normalize each feature's amplitudes to sum to 1
        self.amplitudes = self.amplitudes / self.amplitudes.sum(axis=1, keepdims=True)
        
        return self
    
    def transform(self, X):
        """Apply phase encoding"""
        n_samples = X.shape[0]
        # Each feature becomes n_phases * 2 values (real + imaginary parts)
        encoded_X = np.zeros((n_samples, self.n_features * self.n_phases * 2))
        
        # Normalize features to [0, 1] for phase mapping
        X_norm = (X - self.feature_mins) / self.feature_ranges
        
        for i in range(n_samples):
            encoded_sample = []
            
            for feature_idx in range(self.n_features):
                value = X_norm[i, feature_idx]
                
                # Create superposition of phase states
                for phase_idx in range(self.n_phases):
                    # Calculate phase: 2π * value + random shift
                    phase = 2 * np.pi * value + self.phase_shifts[feature_idx, phase_idx]
                    
                    # Create complex exponential
                    complex_val = np.exp(1j * phase)
                    
                    # Weight by amplitude
                    amplitude = np.sqrt(self.amplitudes[feature_idx, phase_idx])
                    complex_val *= amplitude
                    
                    # Store real and imaginary parts
                    encoded_sample.append(complex_val.real)
                    encoded_sample.append(complex_val.imag)
            
            encoded_X[i] = np.array(encoded_sample)
        
        # Add quantum measurement noise if requested
        if self.add_noise:
            noise = np.random.randn(*encoded_X.shape) * self.noise_level
            encoded_X += noise
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantumWalkEncoder:
    """
    Use quantum random walk principles for mixing
    Creates complex interference patterns
    """
    
    def __init__(self, n_steps=5, graph_type='complete'):
        """
        n_steps: Number of quantum walk steps
        graph_type: Type of graph for walk ('complete', 'line', 'star')
        """
        self.n_steps = n_steps
        self.graph_type = graph_type
        
    def _create_graph(self, n_nodes):
        """Create adjacency matrix for quantum walk"""
        if self.graph_type == 'complete':
            # Complete graph: all nodes connected
            adj = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        elif self.graph_type == 'line':
            # Line graph
            adj = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes-1):
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        elif self.graph_type == 'star':
            # Star graph
            adj = np.zeros((n_nodes, n_nodes))
            for i in range(1, n_nodes):
                adj[0, i] = 1
                adj[i, 0] = 1
        
        return adj
    
    def fit(self, X):
        """Setup quantum walk parameters"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Create graph for quantum walk
        self.adjacency = self._create_graph(n_features)
        
        # Create coin operator (Hadamard-like mixing)
        self.coin_operator = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Store feature statistics
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        
        return self
    
    def transform(self, X):
        """Apply quantum walk mixing - preserves both magnitude and phase information"""
        n_samples = X.shape[0]
        
        # Normalize features
        X_norm = (X - self.feature_means) / self.feature_stds
        
        # Output both magnitude and phase information (2x dimensions)
        encoded_X = np.zeros((n_samples, self.n_features * 2))
        
        for i in range(n_samples):
            # Initial quantum state: features as amplitudes (complex for phase info)
            # Use original values as magnitude, add small phase variation
            state = X_norm[i].copy().astype(complex)
            # Add small imaginary component to preserve phase information
            state = state + 1j * X_norm[i] * 0.1
            
            # Perform quantum walk
            for step in range(self.n_steps):
                # Coin operation (mix directions)
                # Simplified: apply mixing matrix
                mixing = self.adjacency + np.eye(self.n_features)
                state = mixing @ state
                
                # Shift operation (move along graph)
                state = self.adjacency @ state
                
                # Preserve magnitude information (don't normalize to unit vector)
                # Only normalize if values become too large
                max_mag = np.max(np.abs(state))
                if max_mag > 10:
                    state = state / (max_mag / 10)
            
            # Extract magnitude and phase information
            magnitude = np.abs(state)
            phase = np.angle(state)
            
            # Store both magnitude and phase
            encoded_X[i, :self.n_features] = magnitude
            encoded_X[i, self.n_features:] = phase
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantumFingerprintingEncoder:
    """
    Quantum Fingerprinting Encoder
    
    Inspired by quantum fingerprinting from quantum information theory.
    Creates a compact "fingerprint" representation of data that:
    1. Uses quantum superposition to encode all features into a single quantum state
    2. Applies quantum measurements to extract a fingerprint
    3. Uses quantum interference to mix information non-linearly
    4. Creates a compact but information-rich representation
    
    The fingerprint is hard to reverse (many-to-one mapping) but preserves
    useful information for ML tasks through quantum measurement principles.
    """
    
    def __init__(self, fingerprint_dim=None, n_measurements=3, interference_strength=0.5):
        """
        fingerprint_dim: Dimension of the fingerprint (default: same as input)
        n_measurements: Number of quantum measurements to perform
        interference_strength: Strength of quantum interference effects (0-1)
        """
        self.fingerprint_dim = fingerprint_dim
        self.n_measurements = n_measurements
        self.interference_strength = interference_strength
        
    def fit(self, X):
        """Setup quantum fingerprinting parameters"""
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Set fingerprint dimension (default: same as input, or expand if specified)
        if self.fingerprint_dim is None:
            self.fingerprint_dim = n_features
        elif self.fingerprint_dim < n_features:
            # If smaller, we'll compress (many-to-one mapping)
            self.fingerprint_dim = max(1, self.fingerprint_dim)
        else:
            # If larger, we'll expand (one-to-many mapping)
            self.fingerprint_dim = self.fingerprint_dim
        
        # Store feature statistics for normalization
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        
        # Create quantum measurement bases (like different measurement angles)
        # Each measurement basis is a random orthogonal projection
        self.measurement_bases = []
        for m in range(self.n_measurements):
            # Create a random measurement basis (like measuring in different directions)
            # This simulates quantum measurements in different bases
            # Basis maps from fingerprint_dim to fingerprint_dim (square matrix)
            basis = np.random.randn(self.fingerprint_dim, self.fingerprint_dim)
            # Orthonormalize (like quantum measurement operators)
            Q, _ = np.linalg.qr(basis)
            self.measurement_bases.append(Q)
        
        # Create interference matrix (quantum interference between features)
        # This simulates how quantum states interfere with each other
        # Maps from n_features to fingerprint_dim
        self.interference_matrix = np.random.randn(self.fingerprint_dim, n_features)
        # Normalize to control interference strength
        self.interference_matrix = self.interference_matrix * self.interference_strength
        
        return self
    
    def transform(self, X):
        """Apply quantum fingerprinting transformation"""
        n_samples = X.shape[0]
        
        # Normalize features
        X_norm = (X - self.feature_means) / self.feature_stds
        
        # Output: fingerprint_dim * n_measurements * 2 (real + imaginary for each measurement)
        output_dim = self.fingerprint_dim * self.n_measurements * 2
        encoded_X = np.zeros((n_samples, output_dim))
        
        for i in range(n_samples):
            fingerprint_parts = []
            
            # Step 1: Create quantum superposition state from all features
            # Each feature contributes to a quantum state
            quantum_state = np.zeros(self.fingerprint_dim, dtype=complex)
            
            for feature_idx in range(self.n_features):
                value = X_norm[i, feature_idx]
                
                # Map feature value to quantum amplitude
                # Use complex exponential to create phase information
                amplitude = abs(value)
                phase = 2 * np.pi * value / (self.n_features + 1)  # Normalize phase
                
                # Create quantum state component
                quantum_component = amplitude * np.exp(1j * phase)
                
                # Project onto fingerprint space using interference matrix
                # This creates quantum interference between features
                quantum_state += self.interference_matrix[:, feature_idx] * quantum_component
            
            # Step 2: Apply quantum measurements (like measuring in different bases)
            for measurement_idx in range(self.n_measurements):
                basis = self.measurement_bases[measurement_idx]
                
                # Measure quantum state in this basis
                # This simulates quantum measurement: |<basis|state>|^2
                measurement = basis @ quantum_state
                
                # Extract measurement probabilities (like Born rule)
                # Real part: amplitude, Imaginary part: phase information
                probabilities = np.abs(measurement) ** 2
                
                # Normalize to create probability distribution
                prob_sum = probabilities.sum() + 1e-8
                probabilities = probabilities / prob_sum
                
                # Also capture phase information (quantum phase is important)
                phases = np.angle(measurement)
                
                # Combine probability and phase into fingerprint
                # Use square root of probabilities (like quantum amplitudes)
                fingerprint = np.sqrt(probabilities) * np.exp(1j * phases * 0.1)
                
                # Store real and imaginary parts (quantum state information)
                fingerprint_parts.extend(fingerprint.real)
                fingerprint_parts.extend(fingerprint.imag)
            
            # Step 3: Combine all measurements into final fingerprint
            fingerprint_combined = np.array(fingerprint_parts)
            
            # Normalize to prevent extreme values
            max_val = np.abs(fingerprint_combined).max()
            if max_val > 1e6:
                fingerprint_combined = fingerprint_combined / (max_val / 1e6)
            
            encoded_X[i] = fingerprint_combined
        
        # Validate output
        encoded_X = np.nan_to_num(encoded_X, nan=0.0, posinf=1.0, neginf=0.0)
        encoded_X = np.clip(encoded_X, -1e6, 1e6)
        
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CombinedQuantumEncoder:
    """
    Combine multiple quantum-inspired encodings for maximum privacy
    """
    
    def __init__(self, encoders=None):
        """
        encoders: List of encoder objects to apply sequentially
        """
        if encoders is None:
            # Default pipeline
            self.encoders = [
                QuantumSuperpositionEncoder(n_states=3, temperature=0.5),
                QuantumEntanglementEncoder(entanglement_strength=0.7),
                QuantumPhaseEncoder(n_phases=3, add_noise=True, noise_level=0.02),
            ]
        else:
            self.encoders = encoders
            
    def fit(self, X):
        """Fit all encoders in the pipeline"""
        current_X = X.copy()
        self.fitted_encoders = []
        
        for encoder in self.encoders:
            encoder.fit(current_X)
            current_X = encoder.transform(current_X)
            self.fitted_encoders.append(encoder)
        
        return self
    
    def transform(self, X):
        """Apply all encoders sequentially"""
        current_X = X.copy()
        
        for encoder in self.encoders:
            current_X = encoder.transform(current_X)
        
        return current_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


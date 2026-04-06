# Encoders Module

This module contains various encoding and mixing methods for privacy-preserving machine learning.

## Structure

- **`quantum_encoders.py`**: Quantum-inspired encoding methods
  - `QuantumSuperpositionEncoder`
  - `QuantumEntanglementEncoder`
  - `QuantumFourierEncoder`
  - `QuantumPhaseEncoder`
  - `QuantumWalkEncoder`
  - `QuantumFingerprintingEncoder`
  - `CombinedQuantumEncoder`

- **`classical_encoders.py`**: Classical encoding methods
  - `ClassicalXORMixer`
  - `RandomProjectionEncoder`

- **`irreversible_encoders.py`**: Irreversible encoding methods
  - `ChaoticSystemEncoder`
  - `OrthogonalSubspaceEncoder`
  - `FastPrivacyEncoder`
  - `RandomOneToManyEncoder` (deprecated for continuous data)

- **`pipelines.py`**: Pipeline utilities
  - `EncodeThenMixPipeline`

## Usage

```python
from quagua.encoders import (
    ChaoticSystemEncoder,
    QuantumWalkEncoder,
    FastPrivacyEncoder,
    EncodeThenMixPipeline,
)

# Single encoder
encoder = ChaoticSystemEncoder(map_type='logistic', n_iterations=5)
X_encoded = encoder.fit_transform(X)

# Pipeline
pipeline = EncodeThenMixPipeline(
    ChaoticSystemEncoder(map_type='logistic', n_iterations=5),
    QuantumWalkEncoder(n_steps=3)
)
X_processed = pipeline.fit_transform(X)
```

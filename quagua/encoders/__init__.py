"""
Encoder modules for quantum-inspired and irreversible encoding
"""

from .quantum_encoders import (
    QuantumSuperpositionEncoder,
    QuantumEntanglementEncoder,
    QuantumFourierEncoder,
    QuantumPhaseEncoder,
    QuantumWalkEncoder,
    QuantumFingerprintingEncoder,
    CombinedQuantumEncoder,
)

from .classical_encoders import (
    ClassicalXORMixer,
    RandomProjectionEncoder,
)

from .irreversible_encoders import (
    RandomOneToManyEncoder,
    ChaoticSystemEncoder,
    OrthogonalSubspaceEncoder,
    FastPrivacyEncoder,
)

from .random_walk_encoder import (
    RandomWalkFeatureEncoder,
)

from .pipelines import (
    EncodeThenMixPipeline,
)

__all__ = [
    # Quantum encoders
    'QuantumSuperpositionEncoder',
    'QuantumEntanglementEncoder',
    'QuantumFourierEncoder',
    'QuantumPhaseEncoder',
    'QuantumWalkEncoder',
    'QuantumFingerprintingEncoder',
    'CombinedQuantumEncoder',
    # Classical encoders
    'ClassicalXORMixer',
    'RandomProjectionEncoder',
    # Irreversible encoders
    'RandomOneToManyEncoder',
    'ChaoticSystemEncoder',
    'OrthogonalSubspaceEncoder',
    'FastPrivacyEncoder',
    # Random walk encoder
    'RandomWalkFeatureEncoder',
    # Pipelines
    'EncodeThenMixPipeline',
]

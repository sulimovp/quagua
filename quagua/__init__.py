"""
QuaGua: encode-then-mix pipelines and evaluation helpers for tabular data.

The package bundles irreversible encoders, mixers, optional quantum-inspired
components, and a harness that scores utility (downstream models) against
linear reconstruction-style attacks.
"""

__version__ = "0.1.0"

from . import encoders
from . import evaluation
from . import data_utils

__all__ = [
    "encoders",
    "evaluation",
    "data_utils",
]

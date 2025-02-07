"""
ReefConnect
~~~~~~~~~~

A Python package for calculating connectivity kernels in the Great Barrier Reef.
This package provides tools for:
- Calculating angles between reef positions
- Computing connectivity matrices
- Generating connectivity kernels
"""

from .angles import calculate_angles, compute_bearing
from .connectivity import create_connectivity_matrix
from .kernels import generate_kernels

__version__ = "0.1.0"
__all__ = [
    'calculate_angles',
    'compute_bearing',
    'create_connectivity_matrix',
    'generate_kernels',
]

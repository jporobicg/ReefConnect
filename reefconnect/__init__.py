"""
ReefConnect
~~~~~~~~~~

A Python package for calculating connectivity kernels in the Great Barrier Reef.
This package provides tools for:
- Calculating angles between reef positions
- Computing connectivity matrices
- Generating connectivity kernels
"""

from .angles import calculate_angle, calculate_direction_sector
from .utils import haversine
from .connectivity import ConnectivityCalculator
from .kernels import calculate_ds

__version__ = "0.1.0"
__all__ = [
    'calculate_angle',
    'calculate_direction_sector',
    'haversine',
    'ConnectivityCalculator',
    'calculate_ds'
]

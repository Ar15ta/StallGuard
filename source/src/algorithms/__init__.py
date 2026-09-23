"""Windowed signal-analysis methods."""

from .autocorrelation import autocorrelation_features
from .entropy import permutation_entropy_features
from .wavelet import wavelet_energy_features

__all__ = [
    "autocorrelation_features",
    "permutation_entropy_features",
    "wavelet_energy_features",
]

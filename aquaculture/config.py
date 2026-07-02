"""
Configuration for the AquacultureFeatureEngineer.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import numpy as np


@dataclass
class AquacultureConfig:
    """
    Configuration for the AquacultureFeatureEngineer.

    Parameters
    ----------
    simulate_mask : bool, default=True
        Whether to simulate cloud masking and window selection.
    random_state : int, np.random.Generator or None, default=None
        Random seed or generator for reproducible random operations.
    window_length_probs : tuple of float, default=(0.2, 0.5, 0.3)
        Probabilities for window lengths 4, 5, and 6 months.
        Must sum to 1.0.
    start_month_distribution : list of float or None, default=None
        Probability distribution for the start month (0-11). If None,
        a uniform distribution over valid start months (0 to 12 - window_length)
        is used for the selected window length.
    s2_monthly_dropout : list of float, default=[0.0]*12
        Monthly dropout probabilities for Sentinel-2 bands (10 bands).
        Each value should be between 0 and 1.
    include_raw_features : bool, default=True
        Whether to include raw VH and VV bands.
    include_temporal_statistics : bool, default=True
        Whether to include temporal statistics (mean, std, min, max, amplitude, slope).
    include_cross_sensor_features : bool, default=True
        Whether to include cross sensor features (ratios and products of SAR and optical indices).
    include_metadata : bool, default=True
        Whether to include metadata features (window length, start month, etc.).
    """

    simulate_mask: bool = True
    random_state: Optional[Union[int, np.random.Generator]] = None
    window_length_probs: Tuple[float, float, float] = (0.2, 0.5, 0.3)
    start_month_distribution: Optional[List[float]] = None
    s2_monthly_dropout: List[float] = field(default_factory=lambda: [0.001, 0.037, 0.003, 0.001, 0.001, 0.073, 0.001, 0.009, 0.003, 0.176, 0.007, 0.0])
    include_raw_features: bool = True
    include_temporal_statistics: bool = True
    include_cross_sensor_features: bool = True
    include_metadata: bool = True

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate window_length_probs
        if len(self.window_length_probs) != 3:
            raise ValueError("window_length_probs must be a tuple of length 3.")
        if not np.isclose(sum(self.window_length_probs), 1.0):
            raise ValueError("window_length_probs must sum to 1.0.")
        if any(p < 0 for p in self.window_length_probs):
            raise ValueError("window_length_probs must be non-negative.")

        # Validate start_month_distribution if provided
        if self.start_month_distribution is not None:
            if len(self.start_month_distribution) != 12:
                raise ValueError("start_month_distribution must be a list of length 12.")
            if not np.isclose(sum(self.start_month_distribution), 1.0):
                raise ValueError("start_month_distribution must sum to 1.0.")
            if any(p < 0 for p in self.start_month_distribution):
                raise ValueError("start_month_distribution must be non-negative.")

        # Validate s2_monthly_dropout
        if len(self.s2_monthly_dropout) != 12:
            raise ValueError("s2_monthly_dropout must be a list of length 12.")
        if any(p < 0 or p > 1 for p in self.s2_monthly_dropout):
            raise ValueError("s2_monthly_dropout values must be between 0 and 1.")
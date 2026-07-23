"""Cloud masking and contamination simulation for aquaculture remote sensing.

This module provides functions to simulate cloud contamination and missing data
patterns in Sentinel-2 optical bands while preserving SAR bands, as well as
functions to validate the masking procedures.

Typical usage
-------------
>>> import numpy as np
>>> from aquaculture.masking import create_s2_mask
>>> # Create a mask for 12 months, 100 samples, 10 S2 bands
>>> mask = create_s2_mask(
...     n_samples=100,
...     n_timesteps=12,
...     n_bands=10,
...     monthly_dropout=[0.1]*12,
...     random_state=42
... )
>>> mask.shape
(100, 12, 10)
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
from aquaculture.config import AquacultureConfig

def _validate_probabilities(probs: np.ndarray, name: str) -> None:
    """Validate that an array of probabilities sums to 1 and is non-negative.

    Parameters
    ----------
    probs : np.ndarray
        Array of probabilities.
    name : str
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If probabilities do not sum to 1 (within tolerance) or contain negatives.
    """
    if not np.allclose(np.sum(probs), 1.0):
        raise ValueError(f'{name} must sum to 1.0, got {np.sum(probs)}')
    if np.any(probs < 0):
        raise ValueError(f'{name} must contain non-negative values')

def select_window_length(rng: np.random.Generator, window_length_probs: tuple[float, float, float]) -> int:
    """Select a window length (4, 5, or 6 months) based on given probabilities.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    window_length_probs : tuple of float
        Probabilities for window lengths 4, 5, and 6 months.

    Returns
    -------
    window_length : int
        Selected window length (4, 5, or 6).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> select_window_length(rng, (1/3, 1/3, 1/3))
    5
    """
    _validate_probabilities(np.array(window_length_probs), 'window_length_probs')
    return int(rng.choice([4, 5, 6], p=window_length_probs))

def select_start_month(rng: np.random.Generator, window_length: int, start_month_distribution: Optional[np.ndarray]=None) -> int:
    """Select a start month for the observation window.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    window_length : int
        Length of the observation window (4, 5, or 6 months).
    start_month_distribution : np.ndarray or None, default=None
        Probability distribution for start month (0-11). If None, a uniform
        distribution over valid start months (0 to 12 - window_length) is used.

    Returns
    -------
    start_month : int
        Selected start month (0 = January, 11 = December).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> select_start_month(rng, 5)
    3
    """
    max_start = 12 - window_length
    if start_month_distribution is None:
        probs = np.ones(max_start + 1) / (max_start + 1)
    else:
        if len(start_month_distribution) != 12:
            raise ValueError('start_month_distribution must have length 12 (months Jan-Dec)')
        _validate_probabilities(np.array(start_month_distribution), 'start_month_distribution')
        probs = start_month_distribution[:max_start + 1]
        probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(max_start + 1), p=probs))

def create_s2_mask(n_samples: int, n_timesteps: int, n_bands: int, monthly_dropout: list[float], random_state: Optional[np.random.Generator]=None) -> np.ndarray:
    """Create a mask for Sentinel-2 bands simulating monthly cloud contamination.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_timesteps : int
        Number of time steps (should be 12 for monthly data).
    n_bands : int
        Number of Sentinel-2 bands (should be 10).
    monthly_dropout : list of float
        Dropout probability for each month (0-1). Length should be 12.
    random_state : np.random.Generator or None, default=None
        Random number generator. If None, a new generator is seeded from
        entropy.

    Returns
    -------
    mask : np.ndarray
        Boolean mask where True indicates the value should be kept (not masked),
        False indicates the value should be masked (set to -9999).
        Shape: (n_samples, n_timesteps, n_bands)

    Examples
    --------
    >>> mask = create_s2_mask(100, 12, 10, [0.1]*12, random_state=42)
    >>> mask.shape
    (100, 12, 10)
    >>> # Approximately 10% of values should be False (masked)
    >>> 1 - np.mean(mask)
    0.102...
    """
    if n_timesteps != 12:
        raise ValueError('n_timesteps must be 12 for monthly data')
    if n_bands != 10:
        raise ValueError('n_bands must be 10 for Sentinel-2 bands')
    if len(monthly_dropout) != 12:
        raise ValueError('monthly_dropout must have length 12')
    if any((p < 0 or p > 1 for p in monthly_dropout)):
        raise ValueError('monthly_dropout values must be between 0 and 1')
    rng = np.random.default_rng(random_state) if isinstance(random_state, int) else random_seed or np.random.default_rng()
    shape = (n_samples, n_timesteps, n_bands)
    rands = rng.random(shape)
    dropout_array = np.array(monthly_dropout).reshape((1, 12, 1))
    mask = rands >= dropout_array
    return mask

def apply_s2_masking(data: np.ndarray, mask: np.ndarray, fill_value: float=-9999.0) -> np.ndarray:
    """Apply Sentinel-2 mask to data, setting masked values to fill_value.

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape (n_samples, n_timesteps, n_bands).
        Assumes first two-bands 0-1: SAR (VH, VV) - should NOT be masked
        -bands 2-9: Sentinel-2 (Blue, Green, Red, RE1, RE2, RE3, NIR, NarrowNIR, SWIR1, SWIR2)
    mask : np.ndarray
        Boolean mask array with shape (n_samples, n_timesteps, n_bands) where
        True indicates keep value, False indicates mask (set to fill_value).
    fill_value : float, default=-9999.0
        Value to use for masked pixels.

    Returns
    -------
    masked_data : np.ndarray
        Data with masked values set to fill_value.

    Examples
    --------
    >>> data = np.ones((10, 12, 12))  # 10 samples, 12 months, 12 bands
    >>> mask = np.ones((10, 12, 12), dtype=bool)
    >>> mask[:, :, 2:] = False  # Mask all S2 bands
    >>> masked = apply_s2_masking(data, mask)
    >>> np.all(masked[:, :, :2] == 1.0)  # SAR unchanged
    True
    >>> np.all(masked[:, :, 2:] == -9999.0)  # S2 masked
    True
    """
    if data.shape != mask.shape:
        raise ValueError(f'Data shape {data.shape} must match mask shape {mask.shape}')
    masked_data = data.copy()
    masked_data[~mask] = fill_value
    return masked_data

def apply_competition_mask(data: np.ndarray, config: AquacultureConfig) -> tuple[np.ndarray, dict]:
    """Apply competition-style masking to simulate partial observations.

    This function selects a random contiguous window of months (4-6 months)
    for each sample and sets all bands outside that window to -9999 (later
    converted to NaN). Inside the window, SAR bands (VH, VV) are kept
    unchanged, while each S2 band is masked independently according to the
    month-specific dropout probability in s2_monthly_dropout. The
    resulting mask indicates which values were kept (True) vs. set to -9999
    (False). SAR bands are never masked by the dropout step but are set to
    -9999 outside the selected window, simulating a temporal
    observation window.

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape (n_samples, 12, 12).
        Band order: 0=VH, 1=VV, 2=Blue, 3=Green, 4=Red, 5=RE1, 6=RE2,
                  7=RE3, 8=NIR, 9=NarrowNIR, 10=SWIR1, 11=SWIR2.
    config : AquacultureConfig
        Configuration object containing masking parameters.

    Returns
    -------
    masked_data : np.ndarray
        Data after applying the competition mask, with the same shape as
        input. Values outside the selected window or masked by S2
        dropout are set to -9999.
    metadata : dict
        Dictionary with per-sample metadata:
        - window_length: ndarray of shape (n_samples,) with values 4,5,6
        - start_month: ndarray of shape (n_samples,) with values 0-11
        - end_month: ndarray of shape (n_samples,) with values 0-11
"""
    if data.ndim != 3 or data.shape[1] != 12 or data.shape[2] != 12:
        raise ValueError('Input data must have shape (n_samples, 12, 12)')
    n_samples = data.shape[0]
    rng = np.random.default_rng(config.random_state) if isinstance(config.random_state, int) else config.random_state if isinstance(config.random_state, np.random.Generator) else np.random.default_rng()
    window_lengths = np.empty(n_samples, dtype=int)
    start_months = np.empty(n_samples, dtype=int)
    end_months = np.empty(n_samples, dtype=int)
    masked_data = data.copy()
    for i in range(n_samples):
        window_length = select_window_length(rng, config.window_length_probs)
        start_month = select_start_month(rng, window_length, config.start_month_distribution)
        end_month = start_month + window_length - 1
        window_lengths[i] = window_length
        start_months[i] = start_month
        end_months[i] = end_month
        monthly_mask = np.zeros(12, dtype=bool)
        monthly_mask[start_month:end_month + 1] = True
        is_S2_band = np.array([False, False, True, True, True, True, True, True, True, True, True, True])
        is_outside_window = ~monthly_mask
        where_to_mask = np.tile(is_outside_window[:, np.newaxis], (1, 12)) & np.tile(is_S2_band[np.newaxis, :], (12, 1))
        masked_data[i][where_to_mask] = -9999.0
    metadata = {'window_length': window_lengths, 'start_month': start_months, 'end_month': end_months}
    return (masked_data, metadata)

def validate_masking(original_data: np.ndarray, masked_data: np.ndarray, mask_metadata: dict) -> dict:
    """Validate that masking was applied correctly.

    Parameters
    ----------
    original_data : np.ndarray
        Original data before masking (shape (n_samples, 12, 12)).
    masked_data : np.ndarray
        Data after masking (same shape).
    mask_metadata : dict
        Metadata from apply_competition_mask containing window_length,
        start_month, end_month arrays (one per sample).

    Returns
    -------
    validation : dict
        Dictionary with validation results:
        - sar_unchanged: bool, whether SAR bands are unchanged
        - s2_outside_window_masked: bool, whether S2 bands outside window are -9999
        - s2_inside_window_preserved: bool, whether S2 bands inside window are unchanged
        - per_sample_stats: dict with per-sample fractions

    Examples
    --------
    >>> # See apply_competition_mask example
    """
    if original_data.shape != masked_data.shape:
        raise ValueError('Input arrays must have the same shape')
    n_samples = original_data.shape[0]
    results = {'sar_unchanged': True, 's2_outside_window_masked': True, 's2_inside_window_preserved': True, 'per_sample_stats': {}}
    for i in range(n_samples):
        wl = mask_metadata['window_length'][i]
        sm = mask_metadata['start_month'][i]
        em = mask_metadata['end_month'][i]
        sar_equal = np.allclose(original_data[i, :, :2], masked_data[i, :, :2], equal_nan=True)
        if not sar_equal:
            results['sar_unchanged'] = False
        s2_orig = original_data[i, :, 2:]
        s2_masked = masked_data[i, :, 2:]
        outside_months = np.ones(12, dtype=bool)
        outside_months[sm:em + 1] = False
        outside_masked = np.isclose(s2_masked[outside_months], -9999.0)
        if not np.all(outside_masked):
            results['s2_outside_window_masked'] = False
        inside_months = np.zeros(12, dtype=bool)
        inside_months[sm:em + 1] = True
        inside_equal = np.allclose(s2_orig[inside_months], s2_masked[inside_months], equal_nan=True)
        if not inside_equal:
            results['s2_inside_window_preserved'] = False
        results['per_sample_stats'][i] = {'window_length': wl, 'start_month': sm, 'end_month': em, 'sar_equal': sar_equal, 's2_outside_masked_proportion': np.mean(np.isclose(s2_masked[outside_months], -9999.0)), 's2_inside_unchanged_proportion': np.mean(np.isclose(s2_orig[inside_months], s2_masked[inside_months], equal_nan=True))}
    return results
"""Temporal statistics for aquaculture remote sensing features.

This module provides functions to compute temporal statistics (mean, standard
deviation, minimum, maximum, amplitude, linear trend) from multi-temporal
satellite data. Functions handle NaN values appropriately for optical and SAR
bands as specified.

Typical usage
-------------
>>> import numpy as np
>>> from aquaculture.temporal import compute_temporal_stats
>>> # Shape: (n_samples, 12 time steps, n_features)
>>> data = np.random.rand(100, 12, 10)
>>> stats = compute_temporal_stats(data, sar_indices=[0,1], optical_indices=list(range(2,10)))
>>> stats.shape
(100, 60)  # 10 features * 6 statistics
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _ignore_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Return array with NaNs removed for 1D input.

    Parameters
    ----------
    arr : np.ndarray
        1D array possibly containing NaNs.

    Returns
    -------
    filtered : np.ndarray
        1D array with NaNs removed. If all values are NaN, returns empty array.
    """
    return arr[~np.isnan(arr)]


def _safe_mean(arr: np.ndarray, ignore_nans: bool) -> float:
    """Compute mean of array, optionally ignoring NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    ignore_nans : bool
        If True, NaN values are ignored in the calculation.
        If False, NaN values propagate (result is NaN if any NaN present).

    Returns
    -------
    mean : float
        Mean of the array.
    """
    if ignore_nans:
        return np.nanmean(arr)
    else:
        if np.any(np.isnan(arr)):
            return np.nan
        return np.mean(arr)


def _safe_std(arr: np.ndarray, ignore_nans: bool) -> float:
    """Compute standard deviation of array, optionally ignoring NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    ignore_nans : bool
        If True, NaN values are ignored in the calculation.
        If False, NaN values propagate (result is NaN if any NaN present).

    Returns
    -------
    std : float
        Standard deviation of the array.
    """
    if ignore_nans:
        return np.nanstd(arr, ddof=1)
    else:
        if np.any(np.isnan(arr)):
            return np.nan
        return np.std(arr, ddof=1)


def _safe_min(arr: np.ndarray, ignore_nans: bool) -> float:
    """Compute minimum of array, optionally ignoring NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    ignore_nans : bool
        If True, NaN values are ignored in the calculation.
        If False, NaN values propagate (result is NaN if any NaN present).

    Returns
    -------
    min : float
        Minimum of the array.
    """
    if ignore_nans:
        return np.nanmin(arr)
    else:
        if np.any(np.isnan(arr)):
            return np.nan
        return np.min(arr)


def _safe_max(arr: np.ndarray, ignore_nans: bool) -> float:
    """Compute maximum of array, optionally ignoring NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    ignore_nans : bool
        If True, NaN values are ignored in the calculation.
        If False, NaN values propagate (result is NaN if any NaN present).

    Returns
    -------
    max : float
        Maximum of the array.
    """
    if ignore_nans:
        return np.nanmax(arr)
    else:
        if np.any(np.isnan(arr)):
            return np.nan
        return np.max(arr)


def _linear_trend(
    arr: np.ndarray, ignore_nans: bool
) -> float:
    """Compute linear trend (slope) of array over equally spaced time points.

    Parameters
    ----------
    arr : np.ndarray
        1D array of values over time (assumed equally spaced).
    ignore_nans : bool
        If True, NaN values are ignored by fitting to non-NaN points.
        If False, if any NaN is present, returns NaN.

    Returns
    -------
    slope : float
        Slope of linear regression. Returns NaN if fewer than two
        valid points are available.
    """
    if ignore_nans:
        # Extract non-NaN values and their indices
        valid = ~np.isnan(arr)
        if np.sum(valid) < 2:
            return np.nan
        x = np.where(valid)[0].astype(float)
        y = arr[valid]
    else:
        # If any NaN present, return NaN (as per "use all available months"
        # but if any missing, we cannot compute trend? Actually requirement:
        # "SAR statistics always use all available months." For trend, if there
        # is a NaN, we cannot compute a linear fit. We'll follow the same
        # logic as mean/std: if any NaN, return NaN.
        if np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(len(arr), dtype=float)
        y = arr
    # Fit linear regression (degree 1)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])  # slope is first coefficient


def compute_temporal_stats(
    data: np.ndarray,
    sar_indices: Sequence[int],
    optical_indices: Sequence[int],
) -> np.ndarray:
    """Compute temporal statistics for each feature.

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape (n_samples, n_timesteps, n_features).
        Assumes time is axis 1.
    sar_indices : sequence of int
        Indices of SAR features (e.g., [0, 1] for VV, VH).
        Statistics for these features will ignore NaN values (compute over valid observations only).
    optical_indices : sequence of int
        Indices of optical features.
        Statistics for these features will ignore NaN values.

    Returns
    -------
    stats : np.ndarray
        Array of shape (n_samples, n_features * 6) where the six statistics
        for each feature are concatenated in the order:
        [mean, std, min, max, amplitude, slope].
        Amplitude is defined as max - min.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate 5 samples, 12 months, 3 features (2 SAR, 1 optical)
    >>> data = np.random.rand(5, 12, 3)
    >>> data[:, :, 2] = np.nan  # Introduce NaN in optical feature
    >>> stats = compute_temporal_stats(data, sar_indices=[0,1], optical_indices=[2])
    >>> stats.shape
    (5, 18)  # 3 features * 6 statistics
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3-dimensional (samples, time, features)")
    n_samples, n_timesteps, n_features = data.shape
    expected_indices = set(range(n_features))
    provided_indices = set(sar_indices) | set(optical_indices)
    if provided_indices != expected_indices:
        raise ValueError(
            f"Provided indices must cover all features 0-{n_features-1}. "
            f"Got {sorted(provided_indices)}"
        )
    if set(sar_indices) & set(optical_indices):
        raise ValueError("sar_indices and optical_indices must be disjoint")

    # Prepare output array
    n_stats = 6  # mean, std, min, max, amplitude, slope
    stats = np.zeros((n_samples, n_features * n_stats))

    for sample_idx in range(n_samples):
        for feat_idx in range(n_features):
            series = data[sample_idx, :, feat_idx]
            ignore_nans = feat_idx in optical_indices
            # Compute statistics
            mean_val = _safe_mean(series, ignore_nans)
            std_val = _safe_std(series, ignore_nans)
            min_val = _safe_min(series, ignore_nans)
            max_val = _safe_max(series, ignore_nans)
            amp_val = (
                max_val - min_val
                if not (np.isnan(min_val) or np.isnan(max_val))
                else np.nan
            )
            slope_val = _linear_trend(series, ignore_nans)
            # Store
            base_idx = feat_idx * n_stats
            stats[sample_idx, base_idx : base_idx + n_stats] = [
                mean_val,
                std_val,
                min_val,
                max_val,
                amp_val,
                slope_val,
            ]

    return stats
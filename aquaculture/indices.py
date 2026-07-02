"""Spectral index calculations for aquaculture remote sensing.

This module provides functions to compute common vegetation and water indices
from Sentinel-2 multispectral bands. All functions handle NaN values appropriately
and return NaN for invalid divisions (e.g., division by zero or when numerator
and denominator are both zero or NaN).

Typical usage
-------------
>>> import numpy as np
>>> from aquaculture.indices import ndvi
>>> nir = np.array([0.5, 0.6, np.nan])
>>> red = np.array([0.1, 0.2, 0.3])
>>> ndvi(nir, red)
array([0.666..., 0.5,        nan])
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int]


def _validate_inputs(
    a: ArrayLike, b: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to numpy arrays and broadcast them.

    Parameters
    ----------
    a, b : array_like
        Input arrays to be validated and broadcast.

    Returns
    -------
    a, b : np.ndarray
        Broadcasted arrays as float64.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    # Broadcast to common shape
    try:
        a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
    except ValueError as e:
        raise ValueError(f"Arrays could not be broadcast together: {e}")
    return a_arr, b_arr


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """Compute element-wise division handling NaN and infinities.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator array.
    denominator : np.ndarray
        Denominator array.

    Returns
    -------
    result : np.ndarray
        Element-wise division. Returns NaN where denominator is zero or
        where either input is NaN.
    """
    # If either input is NaN, result is NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        # Where denominator is zero or inputs are NaN, set to NaN
        invalid = (denominator == 0) | np.isnan(numerator) | np.isnan(denominator)
        result[invalid] = np.nan
    return result


def ndvi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    nir : array_like
        Near-infrared band.
    red : array_like
        Red band.

    Returns
    -------
    ndvi : np.ndarray
        NDGI values in the range [-1, 1]. Returns NaN where calculation is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> nir = np.array([0.5, 0.6, 0.0])
    >>> red = np.array([0.1, 0.2, 0.0])
    >>> ndvi(nir, red)
    array([0.666..., 0.5,        nan])
    """
    nir_arr, red_arr = _validate_inputs(nir, red)
    numerator = nir_arr - red_arr
    denominator = nir_arr + red_arr
    return safe_divide(numerator, denominator)


def ndwi(green: ArrayLike, nir: ArrayLike) -> np.ndarray:
    """Compute Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    Parameters
    ----------
    green : array_like
        Green band.
    nir : array_like
        Near-infrared band.

    Returns
    -------
    ndwi : np.ndarray
        NDWI values in the range [-1, 1]. Returns NaN where calculation is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> green = np.array([0.3, 0.4, 0.0])
    >>> nir = np.array([0.5, 0.6, 0.0])
    >>> ndwi(green, nir)
    array([-0.25, -0.2,        nan])
    """
    green_arr, nir_arr = _validate_inputs(green, nir)
    numerator = green_arr - nir_arr
    denominator = green_arr + nir_arr
    return safe_divide(numerator, denominator)


def mndwi(green: ArrayLike, swir1: ArrayLike) -> np.ndarray:
    """Compute Modified Normalized Difference Water Index (MNDWI).

    MNDWI = (Green - SWIR1) / (Green + SWIR1)

    Parameters
    ----------
    green : array_like
        Green band.
    swir1 : array_like
        Short-wave infrared 1 band.

    Returns
    -------
    mndwi : np.ndarray
        MNDWI values in the range [-1, 1]. Returns NaN where calculation is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> green = np.array([0.3, 0.4, 0.0])
    >>> swir1 = np.array([0.1, 0.2, 0.0])
    >>> mndwi(green, swir1)
    array([0.5, 0.333..., nan])
    """
    green_arr, swir1_arr = _validate_inputs(green, swir1)
    numerator = green_arr - swir1_arr
    denominator = green_arr + swir1_arr
    return safe_divide(numerator, denominator)


def ndmi(nir: ArrayLike, swir1: ArrayLike) -> np.ndarray:
    """Compute Normalized Difference Moisture Index (NDMI).

    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Parameters
    ----------
    nir : array_like
        Near-infrared band.
    swir1 : array_like
        Short-wave infrared 1 band.

    Returns
    -------
    ndmi : np.ndarray
        NDMI values in the range [-1, 1]. Returns NaN where calculation is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> nir = np.array([0.5, 0.6, 0.0])
    >>> swir1 = np.array([0.1, 0.2, 0.0])
    >>> ndmi(nir, swir1)
    array([0.666..., 0.5,        nan])
    """
    nir_arr, swir1_arr = _validate_inputs(nir, swir1)
    numerator = nir_arr - swir1_arr
    denominator = nir_arr + swir1_arr
    return safe_divide(numerator, denominator)


def ndre(nir: ArrayLike, red_edge: ArrayLike) -> np.ndarray:
    """Compute Normalized Difference Red Edge Index (NDRE).

    NDRE = (NIR - RedEdge) / (NIR + RedEdge)

    Parameters
    ----------
    nir : array_like
        Near-infrared band.
    red_edge : array_like
        Red-edge band (either RE2 or RE3).

    Returns
    -------
    ndre : np.ndarray
        NDRE values in the range [-1, 1]. Returns NaN where calculation is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> nir = np.array([0.5, 0.6, 0.0])
    >>> re = np.array([0.3, 0.4, 0.0])
    >>> ndre(nir, re)
    array([0.25, 0.2,        nan])
    """
    nir_arr, re_arr = _validate_inputs(nir, red_edge)
    numerator = nir_arr - re_arr
    denominator = nir_arr + re_arr
    return safe_divide(numerator, denominator)


def compute_spectral_indices(
    blue: ArrayLike,
    green: ArrayLike,
    red: ArrayLike,
    re1: ArrayLike,
    re2: ArrayLike,
    re3: ArrayLike,
    nir: ArrayLike,
    narrow_nir: ArrayLike,
    swir1: ArrayLike,
    swir2: ArrayLike,
) -> dict[str, np.ndarray]:
    """Compute all standard spectral indices for Sentinel-2 bands.

    Parameters
    ----------
    blue, green, red, re1, re2, re3, nir, narrow_nir, swir1, swir2 : array_like
        Input bands. Must be broadcastable to the same shape.

    Returns
    -------
    indices : dict
        Dictionary mapping index names to computed arrays.
        Keys: 'NDVI', 'NDWI', 'MNDWI', 'NDMI', 'NDRE2', 'NDRE3'

    Examples
    --------
    >>> import numpy as np
    >>> shape = (100, 100)
    >>> bands = [np.random.rand(*shape) for _ in range(10)]
    >>> indices = compute_spectral_indices(*bands)
    >>> sorted(indices.keys())
    ['MDWI', 'NDRE2', 'NDRE3', 'NDMI', 'NDVI', 'NDWI']
    """
    b, g, r, re1, re2, re3, n, nn, s1, s2 = [
        np.asarray(x, dtype=np.float64) for x in (blue, green, red, re1, re2, re3, nir, narrow_nir, swir1, swir2)
    ]

    # Broadcast all arrays to the same shape
    try:
        b, g, r, re1, re2, re3, n, nn, s1, s2 = np.broadcast_arrays(b, g, r, re1, re2, re3, n, nn, s1, s2)
    except ValueError as e:
        raise ValueError(f"Input bands could not be broadcast together: {e}")

    indices = {}
    indices["NDVI"] = ndvi(n, r)
    indices["NDWI"] = ndwi(g, n)
    indices["MNDWI"] = mndwi(g, s1)
    indices["NDMI"] = ndmi(n, s1)
    indices["NDRE2"] = ndre(n, re2)
    indices["NDRE3"] = ndre(n, re3)

    return indices
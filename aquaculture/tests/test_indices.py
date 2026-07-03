"""
Tests for the indices module.
"""
import numpy as np
from aquaculture.indices import (
    ndvi, ndwi, mndwi, ndmi, ndre, compute_spectral_indices
)


def test_ndvi():
    """Test NDVI calculation."""
    # Basic test
    nir = np.array([0.5, 0.6, 0.0])
    red = np.array([0.1, 0.2, 0.0])
    expected = np.array([0.0, 0.0, np.nan])  # Initialize with zeros
    expected[0] = (0.5 - 0.1) / (0.5 + 0.1)  # 0.4/0.6 = 0.666...
    expected[1] = (0.6 - 0.2) / (0.6 + 0.2)  # 0.4/0.8 = 0.5
    # Third element should be NaN because both inputs are 0

    result = ndvi(nir, red)
    assert np.allclose(result, expected, equal_nan=True)


def test_ndvi_with_nan():
    """Test NDVI with NaN inputs."""
    nir = np.array([0.5, np.nan, 0.6])
    red = np.array([0.1, 0.2, np.nan])
    result = ndvi(nir, red)
    assert np.isnan(result[1])  # nir is NaN
    assert np.isnan(result[2])  # red is NaN
    assert not np.isnan(result[0])  # both values are valid
    assert abs(result[0] - (0.5 - 0.1) / (0.5 + 0.1)) < 1e-6


def test_ndwi():
    """Test NDWI calculation."""
    green = np.array([0.3, 0.4, 0.0])
    nir = np.array([0.5, 0.6, 0.0])
    # NDWI = (green - nir) / (green + nir)
    expected = np.array([
        (0.3 - 0.5) / (0.3 + 0.5),  # -0.2/0.8 = -0.25
        (0.4 - 0.6) / (0.4 + 0.6),  # -0.2/1.0 = -0.2
        np.nan  # (0-0)/(0+0) = 0/0 = NaN
    ])
    result = ndwi(green, nir)
    assert np.allclose(result, expected, equal_nan=True)


def test_mndwi():
    """Test MNDWI calculation."""
    green = np.array([0.3, 0.4])
    swir1 = np.array([0.1, 0.2])
    # MNDWI = (green - swir1) / (ground + swir1)
    expected = np.array([
        (0.3 - 0.1) / (0.3 + 0.1),  # 0.2/0.4 = 0.5
        (0.4 - 0.2) / (0.4 + 0.2)   # 0.2/0.6 = 0.333...
    ])
    result = mndwi(green, swir1)
    assert np.allclose(result, expected)


def test_ndmi():
    """Test NDMI calculation."""
    nir = np.array([0.5, 0.6])
    swir1 = np.array([0.1, 0.2])
    # NDMI = (nir - swir1) / (nir + swir1)
    expected = np.array([
        (0.5 - 0.1) / (0.5 + 0.1),  # 0.4/0.6 = 0.666...
        (0.6 - 0.2) / (0.6 + 0.2)   # 0.4/0.8 = 0.5
    ])
    result = ndmi(nir, swir1)
    assert np.allclose(result, expected)


def test_ndre():
    """Test NDRE calculation."""
    nir = np.array([0.5, 0.6])
    red_edge = np.array([0.3, 0.4])
    # NDRE = (nir - red_edge) / (nir + red_edge)
    expected = np.array([
        (0.5 - 0.3) / (0.5 + 0.3),  # 0.2/0.8 = 0.25
        (0.6 - 0.4) / (0.6 + 0.4)   # 0.2/1.0 = 0.2
    ])
    result = ndre(nir, red_edge)
    assert np.allclose(result, expected)


def test_compute_spectral_indices():
    """Test computing all spectral indices."""
    # Create test data
    n_samples = 2
    n_time = 3
    n_bands = 10

    # Create bands in the order: Blue, Green, Red, RE1, RE2, RE3, NIR, NarrowNIR, SWIR1, SWIR2
    blue = np.random.rand(n_samples, n_time)
    green = np.random.rand(n_samples, n_time)
    red = np.random.rand(n_samples, n_time)
    re1 = np.random.rand(n_samples, n_time)
    re2 = np.random.rand(n_samples, n_time)
    re3 = np.random.rand(n_samples, n_time)
    nir = np.random.rand(n_samples, n_time)
    nir_narrow = np.random.rand(n_samples, n_time)
    swir1 = np.random.rand(n_samples, n_time)
    swir2 = np.random.rand(n_samples, n_time)

    # Compute indices
    indices = compute_spectral_indices(
        blue, green, red, re1, re2, re3, nir, nir_narrow, swir1, swir2
    )

    # Check that we got all expected indices
    expected_keys = {'NDVI', 'NDWI', 'MNDWI', 'NDMI', 'NDRE2', 'NDRE3'}
    assert set(indices.keys()) == expected_keys

    # Check shapes
    for key in indices:
        assert indices[key].shape == (n_samples, n_time)

    # Check specific calculations for one element
    # For sample 0, time 0
    b0, g0, r0, re1_0, re2_0, re3_0, n0, nn0, s1_0, s2_0 = [
        arr[0, 0] for arr in [blue, green, red, re1, re2, re3, nir, nir_narrow, swir1, swir2]
    ]

    # NDVI = (NIR - Red) / (NIR + Red)
    denominator = n0 + r0
    if denominator != 0:
        expected_ndvi = (n0 - r0) / denominator
    else:
        expected_ndvi = np.nan
    if np.isnan(expected_ndvi):
        assert np.isnan(indices['NDVI'][0, 0])
    else:
        assert abs(indices['NDVI'][0, 0] - expected_ndvi) < 1e-6

    # NDWI = (Green - NIR) / (Green + NIR)
    denominator = g0 + n0
    if denominator != 0:
        expected_ndwi = (g0 - n0) / denominator
    else:
        expected_ndwi = np.nan
    if np.isnan(expected_ndwi):
        assert np.isnan(indices['NDWI'][0, 0])
    else:
        assert abs(indices['NDWI'][0, 0] - expected_ndwi) < 1e-6


if __name__ == "__main__":
    test_ndvi()
    test_ndvi_with_nan()
    test_ndwi()
    test_mndwi()
    test_ndmi()
    test_ndre()
    test_compute_spectral_indices()
    print("All indices tests passed!")
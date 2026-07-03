"""Tests for the temporal module."""
import numpy as np
from aquaculture.temporal import (
    compute_temporal_stats,
    _ignore_nans_1d,
    _safe_mean,
    _safe_std,
    _safe_min,
    _safe_max,
    _linear_trend
)


def test_ignore_nans_1d():
    """Test _ignore_nans_1d function."""
    # Test with no NaNs
    arr = np.array([1.0, 2.0, 3.0])
    result = _ignore_nans_1d(arr)
    np.testing.assert_array_equal(result, arr)

    # Test with some NaNs
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result = _ignore_nans_1d(arr)
    expected = np.array([1.0, 3.0, 5.0])
    np.testing.assert_array_equal(result, expected)

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan])
    result = _ignore_nans_1d(arr)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

    # Test with empty array
    arr = np.array([])
    result = _ignore_nans_1d(arr)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_safe_mean():
    """Test _safe_mean function."""
    # Test with no NaNs
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _safe_mean(arr, False) == 3.0
    assert _safe_mean(arr, True) == 3.0

    # Test with NaNs
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.isnan(_safe_mean(arr, False))  # Should return NaN when not ignoring NaNs
    assert _safe_mean(arr, True) == 3.0      # Should ignore NaN: (1+2+4+5)/4 = 3.0

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_safe_mean(arr, False))
    assert np.isnan(_safe_mean(arr, True))

    # Test with single element
    arr = np.array([42.0])
    assert _safe_mean(arr, False) == 42.0
    assert _safe_mean(arr, True) == 42.0


def test_safe_std():
    """Test _safe_std function."""
    # Test with no NaNs
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.std(arr, ddof=1)
    assert _safe_std(arr, False) == expected
    assert _safe_std(arr, True) == expected

    # Test with NaNs
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.isnan(_safe_std(arr, False))  # Should return NaN when not ignoring NaNs
    # Should ignore NaN: std of [1,2,4,5]
    expected = np.std([1.0, 2.0, 4.0, 5.0], ddof=1)
    assert _safe_std(arr, True) == expected

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_safe_std(arr, False))
    assert np.isnan(_safe_std(arr, True))


def test_safe_min():
    """Test _safe_min function."""
    # Test with no NaNs
    arr = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    assert _safe_min(arr, False) == 1.0
    assert _safe_min(arr, True) == 1.0

    # Test with NaNs
    arr = np.array([3.0, np.nan, 4.0, 2.0, 5.0])
    assert np.isnan(_safe_min(arr, False))  # Should return NaN when not ignoring NaNs
    assert _safe_min(arr, True) == 2.0      # Should ignore NaN

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_safe_min(arr, False))
    assert np.isnan(_safe_min(arr, True))


def test_safe_max():
    """Test _safe_max function."""
    # Test with no NaNs
    arr = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    assert _safe_max(arr, False) == 5.0
    assert _safe_max(arr, True) == 5.0

    # Test with NaNs
    arr = np.array([3.0, np.nan, 4.0, 2.0, 5.0])
    assert np.isnan(_safe_max(arr, False))  # Should return NaN when not ignoring NaNs
    assert _safe_max(arr, True) == 5.0      # Should ignore NaN

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_safe_max(arr, False))
    assert np.isnan(_safe_max(arr, True))


def test_linear_trend():
    """Test _linear_trend function."""
    # Test with perfect linear relationship y = x
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(_linear_trend(arr, False) - 1.0) < 1e-6
    assert abs(_linear_trend(arr, True) - 1.0) < 1e-6

    # Test with different slope y = 2x
    arr = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    assert abs(_linear_trend(arr, False) - 2.0) < 1e-6
    assert abs(_linear_trend(arr, True) - 2.0) < 1e-6

    # Test with intercept y = 2x + 1
    arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    assert abs(_linear_trend(arr, False) - 2.0) < 1e-6
    assert abs(_linear_trend(arr, True) - 2.0) < 1e-6

    # Test with negative slope
    arr = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    assert abs(_linear_trend(arr, False) - (-1.0)) < 1e-6
    assert abs(_linear_trend(arr, True) - (-1.0)) < 1e-6

    # Test with NaNs - should return NaN when not ignoring NaNs
    arr = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    assert np.isnan(_linear_trend(arr, False))
    # Should work when ignoring NaNs: fit to [0,1,3,4] at x=[0,1,2,3]
    # Actually the x values should be [0,1,3,4] corresponding to the non-NaN positions
    # Using polyfit: y = 1.0*x + 0.0 (approximately)
    result = _linear_trend(arr, True)
    assert not np.isnan(result)
    assert abs(result - 1.0) < 1e-6

    # Test with insufficient points when ignoring NaNs (less than 2)
    arr = np.array([0.0, np.nan, np.nan, np.nan, np.nan])
    assert np.isnan(_linear_trend(arr, True))  # Only 1 point

    # Test with all NaNs
    arr = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    assert np.isnan(_linear_trend(arr, False))
    assert np.isnan(_linear_trend(arr, True))

    # Test with exactly 2 points (should work)
    arr = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
    # Points: (0,0) and (4,4) -> slope = 1.0
    result = _linear_trend(arr, True)
    assert not np.isnan(result)
    assert abs(result - 1.0) < 1e-6


def test_compute_temporal_stats():
    """Test the main compute_temporal_stats function."""
    # Create test data: 2 samples, 4 time steps, 3 features
    # Feature 0: SAR (should not ignore NaNs)
    # Feature 1: SAR (should not ignore NaNs)
    # Feature 2: Optical (should ignore NaNs)
    np.random.seed(42)
    data = np.zeros((2, 4, 3))

    # Sample 0
    # Feature 0 (SAR): [1, 2, 3, 4] -> mean=2.5, std=1.29, min=1, max=4, amp=3, slope=1.0
    data[0, :, 0] = [1, 2, 3, 4]
    # Feature 1 (SAR): [4, 3, 2, 1] -> mean=2.5, std=1.29, min=1, max=4, amp=3, slope=-1.0
    data[0, :, 1] = [4, 3, 2, 1]
    # Feature 2 (Optical): [1, NaN, 3, 4] -> ignoring NaN: [1,3,4] -> mean=2.67, std=1.53, min=1, max=4, amp=3, slope=1.5
    data[0, :, 2] = [1, np.nan, 3, 4]

    # Sample 1
    # Feature 0 (SAR): [2, 4, 6, 8] -> mean=5.0, std=2.58, min=2, max=8, amp=6, slope=2.0
    data[1, :, 0] = [2, 4, 6, 8]
    # Feature 1 (SAR): [8, 6, 4, 2] -> mean=5.0, std=2.58, min=2, max=8, amp=6, slope=-2.0
    data[1, :, 1] = [8, 6, 4, 2]
    # Feature 2 (Optical): [NaN, 2, NaN, 6] -> ignoring NaN: [2,6] -> mean=4.0, std=2.83, min=2, max=6, amp=4, slope=4.0
    data[1, :, 2] = [np.nan, 2, np.nan, 6]

    sar_indices = [0, 1]
    optical_indices = [2]

    stats = compute_temporal_stats(data, sar_indices, optical_indices)

    # Check shape: (n_samples, n_features * n_stats) = (2, 3 * 6) = (2, 18)
    assert stats.shape == (2, 18)

    # Check Sample 0, Feature 0 (SAR)
    # mean, std, min, max, amplitude, slope
    expected_0_0 = [2.5, 1.2909944487358056, 1.0, 4.0, 3.0, 1.0]
    actual_0_0 = stats[0, 0:6]
    np.testing.assert_allclose(actual_0_0, expected_0_0, rtol=1e-10)

    # Check Sample 0, Feature 1 (SAR)
    expected_0_1 = [2.5, 1.2909944487358056, 1.0, 4.0, 3.0, -1.0]
    actual_0_1 = stats[0, 6:12]
    np.testing.assert_allclose(actual_0_1, expected_0_1, rtol=1e-10)

    # Check Sample 0, Feature 2 (Optical)
    # For [1, NaN, 3, 4]:
    # mean = (1+3+4)/3 = 2.666...
    # std = sqrt(((1-2.67)^2 + (3-2.67)^2 + (4-2.67)^2)/2) = 1.5275...
    # min = 1
    # max = 4
    # amplitude = 4-1 = 3
    # slope: using points (0,1), (2,3), (3,4) -> fit y = 1x + 1
    expected_0_2 = [2.6666666666666665, 1.5275252316519465, 1.0, 4.0, 3.0, 1.0]
    actual_0_2 = stats[0, 12:18]
    np.testing.assert_allclose(actual_0_2, expected_0_2, rtol=1e-10)

    # Check Sample 1, Feature 0 (SAR)
    expected_1_0 = [5.0, 2.581988897471611, 2.0, 8.0, 6.0, 2.0]
    actual_1_0 = stats[1, 0:6]
    np.testing.assert_allclose(actual_1_0, expected_1_0, rtol=1e-10)

    # Check Sample 1, Feature 1 (SAR)
    expected_1_1 = [5.0, 2.581988897471611, 2.0, 8.0, 6.0, -2.0]
    actual_1_1 = stats[1, 6:12]
    np.testing.assert_allclose(actual_1_1, expected_1_1, rtol=1e-10)

    # Check Sample 1, Feature 2 (Optical)
    # For [NaN, 2, NaN, 6]:
    # mean = (2+6)/2 = 4.0
    # std = sqrt(((2-4)^2 + (6-4)^2)/1) = sqrt(8) = 2.828...
    # min = 2
    # max = 6
    # amplitude = 6-2 = 4
    # slope: using points (1,2) and (3,6) -> slope = (6-2)/(3-1) = 4/2 = 2.0
    expected_1_2 = [4.0, 2.8284271247461903, 2.0, 6.0, 4.0, 2.0]
    actual_1_2 = stats[1, 12:18]
    np.testing.assert_allclose(actual_1_2, expected_1_2, rtol=1e-10)

    # Test with all NaNs in an optical feature
    data_all_nan = np.zeros((1, 4, 3))
    data_all_nan[0, :, 2] = [np.nan, np.nan, np.nan, np.nan]
    stats_all_nan = compute_temporal_stats(data_all_nan, [0, 1], [2])
    # For optical feature with all NaNs, all stats should be NaN
    assert np.all(np.isnan(stats_all_nan[0, 12:18]))

    # Test with all NaNs in a SAR feature
    data_all_nan_sar = np.zeros((1, 4, 3))
    data_all_nan_sar[0, :, 0] = [np.nan, np.nan, np.nan, np.nan]
    stats_all_nan_sar = compute_temporal_stats(data_all_nan_sar, [0], [1, 2])
    # For SAR feature with all NaNs, all stats should be NaN
    assert np.all(np.isnan(stats_all_nan_sar[0, 0:6]))

    # Test edge case: all valid data (no NaNs)
    data_clean = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])  # Shape (1, 4, 3)
    stats_clean = compute_temporal_stats(data_clean, [0], [1, 2])
    # Feature 0: [1,4,7,10] -> mean=5.5, std=sqrt(15), min=1, max=10, amp=9, slope=3.0
    expected_0_0 = [5.5, 3.872983346207417, 1.0, 10.0, 9.0, 3.0]
    actual_0_0 = stats_clean[0, 0:6]
    np.testing.assert_allclose(actual_0_0, expected_0_0, rtol=1e-10)
    # Feature 1: [2,5,8,11] -> mean=6.5, std=sqrt(15), min=2, max=11, amp=9, slope=3.0
    expected_0_1 = [6.5, 3.872983346207417, 2.0, 11.0, 9.0, 3.0]
    actual_0_1 = stats_clean[0, 6:12]
    np.testing.assert_allclose(actual_0_1, expected_0_1, rtol=1e-10)
    # Feature 2: [3,6,9,12] -> mean=7.5, std=sqrt(15), min=3, max=12, amp=9, slope=3.0
    expected_0_2 = [7.5, 3.872983346207417, 3.0, 12.0, 9.0, 3.0]
    actual_0_2 = stats_clean[0, 12:18]
    np.testing.assert_allclose(actual_0_2, expected_0_2, rtol=1e-10)


if __name__ == "__main__":
    test_ignore_nans_1d()
    test_safe_mean()
    test_safe_std()
    test_safe_min()
    test_safe_max()
    test_linear_trend()
    test_compute_temporal_stats()
    print("All temporal tests passed!")
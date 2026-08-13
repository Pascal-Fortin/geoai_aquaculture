"""
Tests for the AquacultureFeatureEngineer class
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the system path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from aquaculture.config import AquacultureConfig
from aquaculture.feature_engineering import AquacultureFeatureEngineer

def test_conditional_features():
    """Test conditional/threshold-based features functionality."""
    # Create sample data: 3 samples, 12 months, 12 bands
    np.random.seed(42)
    X = np.random.rand(3, 12, 12).astype(np.float64)

    # Introduce some missing values (-9999) to simulate real data
    X[0, 0, 2] = -9999.0  # Missing blue band in first sample, first month
    X[1, 5, 5] = -9999.0  # Missing RE1 in second sample, sixth month

    # Create feature engineer with conditional features enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # No masking for deterministic test
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,  # Need temporal stats for conditional features
        include_cross_sensor_features=True,
        include_metadata=False,
        include_conditional_features=True,
        conditional_feature_specs=[
            {
                "base_feature": "NDWI_max",
                "thresholds": [0.0],
                "outputs": [0, 1]  # NDWI_max < 0 -> 0, NDWI_max >= 0 -> 1
            },
            {
                "base_feature": "MNDWI_std",
                "thresholds": [0.1, 0.3],
                "outputs": [-1, 0, 1]  # <0.1 -> -1, [0.1,0.3) -> 0, >=0.3 -> 1
            }
        ]
    )

    # Fit and transform
    fe.fit(X)
    X_transformed = fe.transform(X, training=False)

    # Check that we have the expected conditional features
    feature_names = list(fe.get_feature_names_out())

    # Should have the conditional features in the output
    assert "NDWI_max_cond_0" in feature_names, "Expected NDWI_max_cond_0 feature not found"
    assert "MNDWI_std_cond_1" in feature_names, "Expected MNDWI_std_cond_1 feature not found"

    # Check that they're in the conditional group
    groups = fe.feature_groups()
    conditional_features = groups['conditional']
    assert len(conditional_features) == 2, f"Expected 2 conditional features, got {len(conditional_features)}"

    # Check that the conditional features have the right naming
    conditional_names = [feature_names[i] for i in conditional_features]
    assert "NDWI_max_cond_0" in conditional_names
    assert "MNDWI_std_cond_1" in conditional_names

    # Get the actual values to verify thresholding logic
    ndwi_max_idx = feature_names.index("NDWI_max_cond_0")
    mndwi_std_idx = feature_names.index("MNDWI_std_cond_1")

    ndwi_max_cond_values = X_transformed.iloc[:, ndwi_max_idx].values
    mndwi_std_cond_values = X_transformed.iloc[:, mndwi_std_idx].values

    # Get the base temporal statistics values to verify against
    # First we need to find where the temporal statistics are
    # NDWI_max and MNDWI_std are the base temporal feature names
    ndwi_max_base_idx = feature_names.index("NDWI_max")
    mndwi_std_base_idx = feature_names.index("MNDWI_std")

    ndwi_max_base_values = X_transformed.iloc[:, ndwi_max_base_idx].values
    mndwi_std_base_values = X_transformed.iloc[:, mndwi_std_base_idx].values

    # Verify thresholding logic for NDWI_max (single threshold at 0.0)
    # Values < 0.0 should map to 0, values >= 0.0 should map to 1
    expected_ndwi_max = np.where(ndwi_max_base_values < 0.0, 0, 1)
    np.testing.assert_array_equal(ndwi_max_cond_values, expected_ndwi_max,
                                  "NDWI_max conditional feature thresholding incorrect")

    # Verify thresholding logic for MNDWI_std (two thresholds at 0.1 and 0.3)
    # < 0.1 -> -1, [0.1, 0.3) -> 0, >= 0.3 -> 1
    expected_mndwi_std = np.where(mndwi_std_base_values < 0.1, -1,
                                  np.where(mndwi_std_base_values < 0.3, 0, 1))
    np.testing.assert_array_equal(mndwi_std_cond_values, expected_mndwi_std,
                                  "MNDWI_std conditional feature thresholding incorrect")

    print("Conditional features test passed!")


def test_conditional_features_disabled():
    """Test that when conditional features are disabled, no conditional features are created."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(2, 12, 12).astype(np.float64)

    # Create feature engineer with conditional features disabled (default)
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=False,
        include_conditional_features=False  # Explicitly disabled (default)
    )

    fe.fit(X)
    X_transformed = fe.transform(X, training=False)

    # Check that we have NO conditional features
    feature_names = list(fe.get_feature_names_out())
    conditional_like_features = [f for f in feature_names if '_cond_' in f]
    assert len(conditional_like_features) == 0, f"Found unexpected conditional-like features: {conditional_like_features}"

    # Check that conditional group is empty
    groups = fe.feature_groups()
    conditional_features = groups['conditional']
    assert len(conditional_features) == 0, f"Expected 0 conditional features, got {len(conditional_features)}"

    print("Conditional features disabled test passed!")


def test_conditional_features_same_base_different_thresholds():
    """Test that the same base feature can be used multiple times with different thresholds."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(2, 12, 12).astype(np.float64)

    # Create feature engineer with conditional features enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=False,
        include_conditional_features=True,
        conditional_feature_specs=[
            {
                "base_feature": "NDWI_max",
                "thresholds": [0.0],
                "outputs": [0, 1]  # First spec: NDWI_max < 0 -> 0, NDWI_max >= 0 -> 1
            },
            {
                "base_feature": "NDWI_max",  # Same base feature
                "thresholds": [-0.2, 0.2],   # Second spec: different thresholds
                "outputs": [-1, 0, 1]        # <-0.2 -> -1, [-0.2,0.2) -> 0, >=0.2 -> 1
            }
        ]
    )

    # Fit and transform
    fe.fit(X)
    X_transformed = fe.transform(X, training=False)

    # Check that we have BOTH conditional features
    feature_names = list(fe.get_feature_names_out())

    # Should have both conditional features with different indices
    assert "NDWI_max_cond_0" in feature_names, "Expected NDWI_max_cond_0 feature not found"
    assert "NDWI_max_cond_1" in feature_names, "Expected NDWI_max_cond_1 feature not found"

    # Check that they're in the conditional group
    groups = fe.feature_groups()
    conditional_features = groups['conditional']
    assert len(conditional_features) == 2, f"Expected 2 conditional features, got {len(conditional_features)}"

    # Check that the conditional features have the right naming
    conditional_names = [feature_names[i] for i in conditional_features]
    assert "NDWI_max_cond_0" in conditional_names
    assert "NDWI_max_cond_1" in conditional_names

    # Get the actual values
    ndwi_max_cond_0_idx = feature_names.index("NDWI_max_cond_0")
    ndwi_max_cond_1_idx = feature_names.index("NDWI_max_cond_1")

    ndwi_max_cond_0_values = X_transformed.iloc[:, ndwi_max_cond_0_idx].values
    ndwi_max_cond_1_values = X_transformed.iloc[:, ndwi_max_cond_1_idx].values

    # Get the base temporal statistics values
    ndwi_max_base_idx = feature_names.index("NDWI_max")  # using max statistic
    ndwi_max_base_values = X_transformed.iloc[:, ndwi_max_base_idx].values

    # Verify first thresholding logic (threshold at 0.0)
    # Values < 0.0 should map to 0, values >= 0.0 should map to 1
    expected_cond_0 = np.where(ndwi_max_base_values < 0.0, 0, 1)
    np.testing.assert_array_equal(ndwi_max_cond_0_values, expected_cond_0,
                                  "First NDWI_max conditional feature thresholding incorrect")

    # Verify second thresholding logic (thresholds at -0.2 and 0.2)
    # < -0.2 -> -1, [-0.2, 0.2) -> 0, >= 0.2 -> 1
    expected_cond_1 = np.where(ndwi_max_base_values < -0.2, -1,
                               np.where(ndwi_max_base_values < 0.2, 0, 1))
    np.testing.assert_array_equal(ndwi_max_cond_1_values, expected_cond_1,
                                  "Second NDWI_max conditional feature thresholding incorrect")

    print("Conditional features same base different thresholds test passed!")


def test_conditional_features_invalid_specs():
    """Test that invalid conditional feature specifications are handled gracefully."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(2, 12, 12).astype(np.float64)

    # Create feature engineer with conditional features enabled but invalid specs
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=False,
        include_conditional_features=True,
        conditional_feature_specs=[
            {
                # Missing base_feature - should be skipped
                "thresholds": [0.0],
                "outputs": [0, 1]
            },
            {
                "base_feature": "NONEXISTENT_FEATURE",  # Doesn't exist - should be skipped
                "thresholds": [0.0],
                "outputs": [0, 1]
            },
            {
                "base_feature": "NDWI_max_invalid",  # Invalid format (too many underscores) - should be skipped
                "thresholds": [0.0],
                "outputs": [0, 1]
            },
            {
                "base_feature": "NDWI_max",  # Valid spec
                "thresholds": [0.0],
                "outputs": [0, 1]
            },
            {
                "base_feature": "NDWI_mean",  # Valid spec but wrong statistic type
                "thresholds": [0.0],
                "outputs": [0, 1],
                # Note: This would be parsed as base="NDWI", stat="mean_invalid" which is invalid
            }
        ]
    )

    # Fit and transform - should not crash
    fe.fit(X)
    X_transformed = fe.transform(X, training=False)

    # Check that we have ONLY the valid conditional feature
    feature_names = list(fe.get_feature_names_out())

    # Should have two conditional features from the valid specs (specs 3 and 4)
    conditional_like_features = [f for f in feature_names if '_cond_' in f]
    assert len(conditional_like_features) == 2, f"Expected 2 conditional features from valid specs, got {len(conditional_like_features)}: {conditional_like_features}"

    # Should be NDWI_max_cond_3 and NDWI_mean_cond_4 (the valid specs)
    assert "NDWI_max_cond_3" in feature_names, "Expected NDWI_max_cond_3 feature not found"
    assert "NDWI_mean_cond_4" in feature_names, "Expected NDWI_mean_cond_4 feature not found"

    # Check that conditional group has exactly two features
    groups = fe.feature_groups()
    conditional_features = groups['conditional']
    assert len(conditional_features) == 2, f"Expected 2 conditional features, got {len(conditional_features)}"

    print("Conditional features invalid specs test passed!")


if __name__ == "__main__":
    # Run the conditional features tests
    test_conditional_features()
    test_conditional_features_disabled()
    test_conditional_features_same_base_different_thresholds()
    test_conditional_features_invalid_specs()
    print("All conditional features tests passed!")
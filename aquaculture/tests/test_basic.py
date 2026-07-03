"""
Simple test for the AquacultureFeatureEngineer.
"""
import numpy as np
import pandas as pd
from aquaculture import AquacultureFeatureEngineer


def test_basic_functionality():
    """Test basic functionality of the feature engineer."""
    # Create sample data: 5 samples, 12 months, 12 bands
    np.random.seed(42)
    X = np.random.rand(5, 12, 12).astype(np.float64)

    # Introduce some missing values (-9999) to simulate real data
    X[0, 0, 2] = -9999.0  # Missing blue band in first sample, first month
    X[2, 5, 5] = -9999.0  # Missing RE1 in second sample, sixth month

    # Create feature engineer
    fe = AquacultureFeatureEngineer(
        simulate_mask=True,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=True,
        include_metadata=True
    )

    # Fit the transformer
    fe.fit(X)

    # Transform training data (with simulation)
    X_train = fe.transform(X, training=True)
    print(f"Training data shape: {X_train.shape}")
    feature_names = fe.get_feature_names_out()
    print(f"Number of feature names: {len(feature_names)}")
    print(f"All feature names: {list(feature_names)}")

    # Debug: let's see what's in feature_names_out_
    if fe.feature_names_out_ is not None:
        print(f"Number of feature_names_out_: {len(fe.feature_names_out_)}")
        print(f"All feature_names_out_: {list(fe.feature_names_out_)}")

    # Check that we have feature names
    assert fe.feature_names_out_ is not None

    # Print the actual lengths for debugging
    print(f"Expected features from feature_names_out_: {len(fe.feature_names_out_)}")
    print(f"Actual features in X_train: {X_train.shape[1]}")

    # This is the assertion that's failing
    assert len(fe.feature_names_out_) == X_train.shape[1], f"Feature count mismatch: {len(fe.feature_names_out_)} names vs {X_train.shape[1]} columns"

    # Transform test data (no simulation)
    X_test = fe.transform(X, training=False)
    print(f"Test data shape: {X_test.shape}")

    # Check that we didn't produce all NaN values
    assert not np.all(np.isnan(X_train.values)), "All values are NaN in training data"
    assert not np.all(np.isnan(X_test.values)), "All values are NaN in test data"

    # Print summary
    print("\nSummary:")
    print(fe.summary())

    # Check that we should create them groups = fe = feature_groups
    print("\nFeature groups:")
    groups = fe.feature_groups()
    for key, indices in groups.items():
        print(f"  {key}: {len(indices)} features")

    print("\nTest passed!")


def test_feature_groups():
    """Test the feature_groups method."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(3, 12, 12).astype(np.float64)

    # Create transformer
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # Simplify for testing
        random_state=42
    )

    # Fit and transform
    fe.fit(X)
    X_transformed = fe.transform(X, training=False)

    # Get feature groups
    groups = fe.feature_groups()
    print(f"Feature groups: { {k: len(v) for k, v in groups.items()} }")

    # Check that we have some features in each expected
    total_assigned = sum(len(v) for v in groups.values())
    assert total_assigned == X_transformed.shape[1], f"Expected {X_transformed.shape[1]} features, got {total_assigned}"

    print("Feature groups test passed!")


def test_config():
    """Test configuration options."""
    # Create sample data
    X = np.random.rand(2, 12, 12).astype(np.float64)

    # Test with minimal features
    fe_minimal = AquacultureFeatureEngineer(
        simulate_mask=False,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=False,
        include_cross_sensor_features=False,
        include_metadata=False
    )
    fe_minimal.fit(X)
    X_minimal = fe_minimal.transform(X, training=False)
    print(f"Minimal features shape: {X_minimal.shape}")

    # Test with maximal features
    fe_maximal = AquacultureFeatureEngineer(
        simulate_mask=False,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=True
    )
    fe_maximal.fit(X)
    X_maximal = fe_maximal.transform(X, training=False)
    print(f"Maximal features shape: {X_maximal.shape}")

    # Maximal should have more features than minimal
    assert X_maximal.shape[1] > X_minimal.shape[1]

    print("Config test passed!")


if __name__ == "__main__":
    print("Testing AquacultureFeatureEngineer...")
    test_basic_functionality()
    test_feature_groups()
    test_config()
    print("All tests passed!")
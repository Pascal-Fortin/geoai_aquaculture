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
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=True
    )

    # Fit the transformer
    fe.fit(X)

    # Transform training data (with simulation)
    X_train = fe.transform(X, training=True)
    print(f"Training data shape: {X_train.shape}")
    feature_names = fe.get_feature_names_out()
    print(f"Number of feature names: {len(feature_names)}")
    print(f"Feature names (first 10): {list(feature_names)[:10]}")  # Show first 10

    # Debug: let's see what's in feature_names_out_
    if fe.feature_names_out_ is not None:
        print(f"Number of feature_names_out_: {len(fe.feature_names_out_)}")
        print(f"First 10 feature_names_out_: {list(fe.feature_names_out_)[:10]}")

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




def test_cloudy_month_within_window():
    """Test that a cloudy month inside the window is ignored for optical feature statistics."""
    # Create deterministic data: one sample, constant values for bands
    np.random.seed(123)
    n_samples = 1
    # Initialize all bands to some baseline value
    X = np.ones((n_samples, 12, 12), dtype=np.float64) * 0.5  # placeholder

    # Set specific bands for NDVI calculation: NIR (band 4) and Red (band 9) according to band order:
    # Band order: 0=VH,1=VV,2=Blue,3=Green,4=NIR,5=Nira,6=RE1,7=RE2,8=RE3,9=Red,10=SWIR1,11=SWIR2
    # Set NIR = 0.2, Red = 0.09 to get NDVI = 0.3793103448275862
    X[:, :, 4] = 0.2   # NIR
    X[:, :, 9] = 0.09  # Red
    # Set other S2 bands to some constants (doesn't matter much for NDVI)
    X[:, :, 2] = 0.1   # Blue
    X[:, :, 3] = 0.5   # Green
    X[:, :, 5] = 0.05  # Nira
    X[:, :, 6] = 0.06  # RE1
    X[:, :, 7] = 0.07  # RE2
    X[:, :, 8] = 0.8   # RE3
    X[:, :, 10] = 0.10 # SWIR1
    X[:, :, 11] = 0.11 # SWIR2

    # SAR bands (VH, VV) we can set to constants but we will disable SAR features
    X[:, :, 0] = 0.3   # VH
    X[:, :, 1] = 0.4   # VV

    # Configure feature engineer: simulate mask True, window length 4 months starting at month 0 (Jan), 
    # and make month 1 (Feb) always cloudy for S2 bands.
    fe = AquacultureFeatureEngineer(
        simulate_mask=True,
        random_state=42,
        window_length_probs=(1.0, 0.0, 0.0),   # always 4 months
        start_month_distribution=[1.0] + [0.0]*11,  # always start at month 0 (Jan)
        s2_monthly_dropout=[0.0, 1.0] + [0.0]*10,  # month 1 (Feb) always cloudy (dropout=1.0), others never cloudy
        include_optical=True,
        include_sar=False,          # disable SAR to avoid NaN propagation issues
        include_temporal_statistics=True,
        include_cross_sensor_features=False,
        include_metadata=False
    )

    fe.fit(X)
    # Transform with training=True to apply masking simulation
    X_trans = fe.transform(X, training=True)

    # Get feature names
    feature_names = fe.get_feature_names_out()
    # Find index of NDVI_mean (since we have temporal statistics)
    try:
        ndvi_mean_idx = list(feature_names).index("NDVI_mean")
    except ValueError:
        raise AssertionError("NDVI_mean not found in feature names")

    ndvi_mean_val = X_trans.iloc[0, ndvi_mean_idx]
    # Expected NDVI mean over months 0,2,3 (Jan, Mar, Apr) each NDVI = 0.3793103448275862 => mean = 0.3793103448275862
    expected = 0.3793103448275862
    tolerance = 1e-6
    assert abs(ndvi_mean_val - expected) < tolerance, f"NDVI mean expected {expected}, got {ndvi_mean_val}"

    # Also check that NDVI_std is near zero (since constant)
    try:
        ndvi_std_idx = list(feature_names).index("NDVI_std")
    except ValueError:
        raise AssertionError("NDVI_std not found")
    ndvi_std_val = X_trans.iloc[0, ndvi_std_idx]
    assert abs(ndvi_std_val - 0.0) < tolerance, f"NDVI std expected ~0, got {ndvi_std_val}"

    # Optionally check that NDVI_min and max also 0.6
    try:
        ndvi_min_idx = list(feature_names).index("NDVI_min")
        ndvi_max_idx = list(feature_names).index("NDVI_max")
    except ValueError:
        raise AssertionError("NDVI min/max not found")
    ndvi_min_val = X_trans.iloc[0, ndvi_min_idx]
    ndvi_max_val = X_trans.iloc[0, ndvi_max_idx]
    assert abs(ndvi_min_val - expected) < tolerance, f"NDVI min expected {expected}, got {ndvi_min_val}"
    assert abs(ndvi_max_val - expected) < tolerance, f"NDVI max expected {expected}, got {ndvi_max_val}"

    print("Cloudy month within window test passed!")


if __name__ == "__main__":
    print("Testing AquacultureFeatureEngineer...")
    test_basic_functionality()
    test_feature_groups()
    test_config()
    test_cloudy_month_within_window()
    print("All tests passed!")

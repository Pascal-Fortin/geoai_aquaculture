"""
Tests for the AquacultureFeatureEngineer.
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


def test_feature_order_matches_transform():
    """Test that feature names generated by _build_feature_names match the order of features produced by transform."""
    # Create a simple test dataset
    # Shape: (n_samples, 12 time steps, 12 bands)
    np.random.seed(42)
    X = np.random.rand(5, 12, 12).astype(np.float64)

    # Introduce some NaN values to simulate missing data
    X[0, 3, 2] = np.nan  # Make one S2 band NaN for first sample
    X[1, 7, 5] = np.nan  # Make another S2 band NaN for second sample

    # Create feature engineer with all features enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # No masking for predictable order
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=True,
        include_metadata=True
    )

    # Fit the transformer (this builds feature names)
    fe.fit(X)

    # Get the feature names
    feature_names = fe.get_feature_names_out()

    # Transform the data to get feature values
    df = fe.transform(X, training=False)

    # Check that number of feature names matches number of columns
    assert len(feature_names) == df.shape[1], f"Feature names count ({len(feature_names)}) doesn't match DataFrame columns ({df.shape[1]})"

    # Check that feature names match DataFrame column names
    for i in range(len(feature_names)):
        assert feature_names[i] == df.columns[i], f"Feature name at index {i} doesn't match: '{feature_names[i]}' vs '{df.columns[i]}'"

    # Additional verification: Check the order of a few specific features
    # Without temporal stats and metadata for simpler verification
    fe_simple = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=False,  # Disable for easier checking
        include_metadata=False
    )

    fe_simple.fit(X)
    df_simple = fe_simple.transform(X, training=False)
    feature_names_simple = fe_simple.get_feature_names_out()

    # Check that we have the expected number of monthly features
    # 11 optical + 4 SAR + 8 cross = 23 features per month
    # 12 months * 23 features = 276 total features
    expected_monthly_features = 12 * (11 + 4 + 8)  # optical + SAR + cross
    assert len(feature_names_simple) == expected_monthly_features, f"Expected {expected_monthly_features} monthly features, got {len(feature_names_simple)}"

    # Check that the first features are January (01) optical features
    expected_first_features = [
        "NDVI_01", "NDWI_01", "MNDWI_01", "NDMI_01", "NDRE2_01", "NDRE3_01",
        "green_01", "nir_01", "nira_01", "swir1_01", "swir2_01"  # 11 optical features
    ]

    for i, expected_feature in enumerate(expected_first_features):
        assert feature_names_simple[i] == expected_feature, f"At position {i}, expected '{expected_feature}', got '{feature_names_simple[i]}'"

    # Check that after optical features come SAR features for January
    sar_start_idx = 11  # After 11 optical features
    expected_sar_features = ["VH_01", "VV_01", "VH_VV_ratio_01", "VH_VV_diff_01"]

    for i, expected_feature in enumerate(expected_sar_features):
        idx = sar_start_idx + i
        assert feature_names_simple[idx] == expected_feature, f"At position {idx}, expected '{expected_feature}', got '{feature_names_simple[idx]}'"

    # Check that after SAR features come cross features for January
    cross_start_idx = sar_start_idx + 4  # After 11 optical + 4 SAR = 15
    expected_cross_features = [
        "VH_NDWI_ratio_01", "VV_NDWI_ratio_01", "VH_NDVI_ratio_01", "VV_NDVI_ratio_01",
        "VH_NDWI_mul_01", "VV_NDWI_mul_01", "VH_NDVI_mul_01", "VV_NDVI_mul_01"
    ]

    for i, expected_feature in enumerate(expected_cross_features):
        idx = cross_start_idx + i
        assert feature_names_simple[idx] == expected_feature, f"At position {idx}, expected '{expected_feature}', got '{feature_names_simple[idx]}'"

    # Check that February features start after all January features
    # January has 23 features (11+4+8), so February should start at index 23
    feb_start_idx = 23
    expected_feb_first = "NDVI_02"
    assert feature_names_simple[feb_start_idx] == expected_feb_first, f"February should start at index {feb_start_idx} with '{expected_feb_first}', got '{feature_names_simple[feb_start_idx]}'"


def test_feature_order_with_temporal_stats():
    """Test feature order when temporal statistics are enabled."""
    # Create a simple test dataset
    np.random.seed(42)
    X = np.random.rand(5, 12, 12).astype(np.float64)

    # Introduce some NaN values to simulate missing data
    X[0, 3, 2] = np.nan  # Make one S2 band NaN for first sample
    X[1, 7, 5] = np.nan  # Make another S2 band NaN for second sample

    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=True,
        include_metadata=False
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    feature_names = fe.get_feature_names_out()

    # Check basic consistency
    assert len(feature_names) == df.shape[1]

    # With temporal stats, we should have:
    # Monthly features: 12 months * 23 features = 276
    # Temporal stats: 23 features * 6 stats = 138
    # Total: 276 + 138 = 414
    # Explanation: For each of the 23 base feature types:
    #   - 12 monthly values (one per month)
    #   - 6 temporal statistics (computed over the 12 months)
    #   - Total per base feature: 12 + 6 = 18
    #   - Total: 23 * 18 = 414
    n_base_features_per_month = 11 + 4 + 8  # optical + SAR + cross
    n_months = 12
    n_temporal_stats = 6
    expected_total = n_base_features_per_month * (n_months + n_temporal_stats)
    assert len(feature_names) == expected_total, f"Expected {expected_total} features with temporal stats, got {len(feature_names)}"

    # Check that monthly features come first, then temporal stats
    # First n_base_features_per_month * n_months should be monthly features
    monthly_count = n_base_features_per_month * n_months
    for i in range(monthly_count):
        # These should NOT end with _mean, _std, etc.
        stat_suffixes = ['_mean', '_std', '_min', '_max', '_amplitude', '_slope']
        is_temporal = any(feature_names[i].endswith(suffix) for suffix in stat_suffixes)
        assert not is_temporal, f"Feature at index {i} ('{feature_names[i]}') should be monthly (no stat suffix) but appears to be temporal"

    # Next features should be temporal statistics
    for i in range(monthly_count, len(feature_names)):
        # These SHOULD end with _mean, _std, etc.
        stat_suffixes = ['_mean', '_std', '_min', '_max', '_amplitude', '_slope']
        is_temporal = any(feature_names[i].endswith(suffix) for suffix in stat_suffixes)
        assert is_temporal, f"Feature at index {i} ('{feature_names[i]}') should be temporal (have stat suffix) but doesn't"


def test_feature_order_consistency_across_configs():
    """Test that feature order is consistent for equivalent configurations."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(5, 12, 12).astype(np.float64)

    # Create two feature engineers with the same settings
    fe1 = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=False,
        include_metadata=False
    )

    fe2 = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=False,
        include_metadata=False
    )

    # Fit both
    fe1.fit(X)
    fe2.fit(X)

    # Get feature names
    names1 = fe1.get_feature_names_out()
    names2 = fe2.get_feature_names_out()

    # Should be identical
    assert len(names1) == len(names2), "Feature arrays should have same length"

    for i, (name1, name2) in enumerate(zip(names1, names2)):
        assert name1 == name2, f"Feature names differ at index {i}: '{name1}' vs '{name2}'"


def test_cross_sensor_division_by_zero():
    """Test that cross-sensor ratios produce NaN when denominator is zero."""
    # Create a deterministic dataset: 1 sample, 12 months, 12 bands
    np.random.seed(42)
    X = np.ones((1, 12, 12), dtype=np.float64) * 0.1  # default small value to avoid zeros elsewhere

    # Band order: 0:VH,1:VV,2:blue,3:green,4:nir,5:nira,6:re1,7:re2,8:re3,9:red,10:swir1,11:swir2
    # Set NIR (band 4) and Red (band 9) equal to get NDVI = 0
    X[:, :, 4] = 0.5  # NIR
    X[:, :, 9] = 0.5  # Red
    # NDVI = (0.5-0.5)/(0.5+0.5) = 0/1 = 0

    # Set Green (band 3) and SWIR1 (band 10) equal to get NDWI = 0
    X[:, :, 3] = 0.5  # Green
    X[:, :, 10] = 0.5 # SWIR1
    # NDWI = (0.5-0.5)/(0.5+0.5) = 0/1 = 0

    # Set SAR bands to non-zero to avoid zero division there
    X[:, :, 0] = 0.6  # VH
    X[:, :, 1] = 0.7  # VV

    # Other bands keep default 0.1 (non-zero)

    # Create feature engineer with cross-sensor features enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # No masking for deterministic test
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=True,
        include_temporal_statistics=False,  # Disable for simpler checking
        include_metadata=False
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    feature_names = fe.get_feature_names_out()

    # Based on our verification, the ordering is:
    # Each month has 23 features: 11 optical + 4 SAR + 8 cross
    # Within cross features (indices 15-22 per month):
    #   Indices 15-18: ratio features (VH_NDWI_ratio, VV_NDWI_ratio, VH_NDVI_ratio, VV_NDVI_ratio)
    #   Indices 19-22: multiplication features (VH_NDWI_mul, VV_NDWI_mul, VH_NDVI_mul, VV_NDVI_mul)
    n_features_per_month = 23
    n_optical = 11
    n_sar = 4
    n_cross = 8
    n_cross_ratio = 4  # first 4 of cross features are ratios
    n_months = 12

    # For each month, check that the four ratio features are NaN where denominator is zero
    for month in range(n_months):
        base_idx = month * n_features_per_month  # start of this month's features
        ratio_start = base_idx + n_optical + n_sar  # index of first ratio feature
        # Expected ratio names for this month
        month_str = f"{month+1:02d}"
        expected_ratio_names = [
            f"VH_NDWI_ratio_{month_str}",
            f"VV_NDWI_ratio_{month_str}",
            f"VH_NDVI_ratio_{month_str}",
            f"VV_NDVI_ratio_{month_str}"
        ]
        # Verify names match
        for i, expected_name in enumerate(expected_ratio_names):
            actual_name = feature_names[ratio_start + i]
            assert actual_name == expected_name, f"Expected {expected_name} at index {ratio_start+i}, got {actual_name}"

        # Get the values for these four features
        ratio_values = df.iloc[0, ratio_start:ratio_start + n_cross_ratio].values
        # Since NDWI and NDVI are zero for all months (we set them constant), denominators are zero -> should be NaN
        assert np.all(np.isnan(ratio_values)), f"Expected NaN for ratio features in month {month+1}, got {ratio_values}"

    # Additionally, verify that multiplication features (last four of cross) are zero (since VH*0 etc)
    for month in range(n_months):
        base_idx = month * n_features_per_month
        mul_start = base_idx + n_optical + n_sar + n_cross_ratio  # index of first multiplication feature
        expected_mul_names = [
            f"VH_NDWI_mul_{month+1:02d}",
            f"VV_NDWI_mul_{month+1:02d}",
            f"VH_NDVI_mul_{month+1:02d}",
            f"VV_NDVI_mul_{month+1:02d}"
        ]
        for i, expected_name in enumerate(expected_mul_names):
            actual_name = feature_names[mul_start + i]
            assert actual_name == expected_name, f"Expected {expected_name} at index {mul_start+i}, got {actual_name}"
        mul_values = df.iloc[0, mul_start:mul_start + 4].values
        # Since NDWI and NDVI are zero, products should be zero (VH*0 = 0, VV*0 = 0)
        assert np.all(mul_values == 0.0), f"Expected zero for multiplication features in month {month+1}, got {mul_values}"


def test_observation_window_from_sar_data():
    """Test that window length, start month, and end month are correctly derived from SAR data when simulate_mask=False."""
    # Create test data with known SAR availability pattern
    np.random.seed(42)
    n_samples = 3
    X = np.full((n_samples, 12, 12), -9999.0, dtype=np.float64)  # Start with all missing

    # Band order: 0:VH,1:VV,2:blue,3:green,4:nir,5:nira,6:re1,7:re2,8:re3,9:red,10:swir1,11:swir2

    # Sample 0: SAR available months 2-5 (Mar-Jun) -> window_length=4, start=2, end=5
    X[0, 2:6, 0] = 0.5  # VH available Mar-Jun
    X[0, 2:6, 1] = 0.6  # VV available Mar-Jun

    # Sample 1: SAR available months 0-3 (Jan-Apr) -> window_length=4, start=0, end=3
    X[1, 0:4, 0] = 0.5  # VH available Jan-Apr
    X[1, 0:4, 1] = 0.6  # VV available Jan-Apr

    # Sample 2: SAR available months 8-11 (Sep-Dec) -> window_length=4, start=8, end=11
    X[2, 8:12, 0] = 0.5  # VH available Sep-Dec
    X[2, 8:12, 1] = 0.6  # VV available Sep-Dec

    # Add some dummy S2 data to avoid issues in other calculations (but we're testing SAR-based window)
    X[:, :, 2:] = 0.1  # Set S2 bands to small non-zero values

    # Create feature engineer with metadata enabled to check window values
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # Important: use actual data, not simulation
        random_state=42,
        include_optical=False,  # Simplify test
        include_sar=True,
        include_cross_sensor_features=False,
        include_temporal_statistics=False,
        include_metadata=True  # Enable to get window_length, start_month, end_month features
    )

    fe.fit(X)
    df = fe.transform(X, training=False)  # training=False to use actual SAR data

    # Get feature names to locate metadata features
    feature_names = list(fe.get_feature_names_out())

    # Find indices of metadata features
    try:
        window_length_idx = feature_names.index("window_length")
        start_month_idx = feature_names.index("start_month")
        end_month_idx = feature_names.index("end_month")
    except ValueError as e:
        raise AssertionError(f"Could not find metadata features: {e}")

    # Extract the metadata values for each sample
    window_lengths = df.iloc[:, window_length_idx].values
    start_months = df.iloc[:, start_month_idx].values
    end_months = df.iloc[:, end_month_idx].values

    # Expected values based on our test data construction
    expected_window_lengths = np.array([4, 4, 4])
    expected_start_months = np.array([2, 0, 8])
    expected_end_months = np.array([5, 3, 11])

    # Check that we got the expected values
    np.testing.assert_array_equal(window_lengths, expected_window_lengths,
                                  "Window lengths do not match expected values")
    np.testing.assert_array_equal(start_months, expected_start_months,
                                  "Start months do not match expected values")
    np.testing.assert_array_equal(end_months, expected_end_months,
                                  "End months do not match expected values")


def test_fraction_optical_calculation():
    """Test that fraction_optical is correctly computed as n_optical_obs / window_lengths."""
    # Create test data: 3 samples, 12 months, 12 bands
    np.random.seed(123)
    n_samples = 3
    X = np.full((n_samples, 12, 12), -9999.0, dtype=np.float64)  # Start with all missing

    # Band order: 0:VH,1:VV,2:blue,3:green,4:nir,5:nira,6:re1,7:re2,8:re3,9:red,10:swir1,11:swir2

    # Sample 0: SAR available months 0-2 (Jan-Mar) -> window_length=3
    #   Make S2 data available in months 0 and 2 (Jan and Mar) -> n_optical_obs=2 -> expected fraction = 2/3
    X[0, 0:3, 0] = 0.5  # VH available Jan-Mar
    X[0, 0:3, 1] = 0.6  # VV available Jan-Mar
    # S2 data: make Jan and Mar have some valid data, Feb missing
    # Month 0 (Jan): set S2 bands to 0.1
    X[0, 0, 2:12] = 0.1
    # Month 1 (Feb): leave as -9999.0 (all S2 bands missing)
    # Month 2 (Mar): set S2 bands to 0.1
    X[0, 2, 2:12] = 0.1

    # Sample 1: SAR available months 3-8 (Apr-Sep) -> window_length=6
    #   Make S2 data available in months 3,5,6,8 (Apr,Jun,Jul,Sep) -> n_optical_obs=4 -> expected fraction = 4/6 = 2/3
    X[1, 3:9, 0] = 0.5  # VH available Apr-Sep
    X[1, 3:9, 1] = 0.6  # VV available Apr-Sep
    # S2 data: set months 3,5,6,8 to 0.1, others to -9999.0
    for m in [3,5,6,8]:
        X[1, m, 2:12] = 0.1
    # months 4,7 remain -9999.0

    # Sample 2: SAR available months 9-11 (Oct-Dec) -> window_length=3
    #   No S2 data available (all months missing) -> n_optical_obs=0 -> expected fraction = 0/3 = 0
    X[2, 9:12, 0] = 0.5  # VH available Oct-Dec
    X[2, 9:12, 1] = 0.6  # VV available Oct-Dec
    # leave all S2 bands as -9999.0

    # Create feature engineer with metadata enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # Use actual SAR data to determine windows
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_cross_sensor_features=False,
        include_temporal_statistics=False,
        include_metadata=True  # Enable to get fraction_optical
    )

    fe.fit(X)
    df = fe.transform(X, training=False)  # training=False to use actual SAR/S2 data

    # Get feature names to locate fraction_optical
    feature_names = list(fe.get_feature_names_out())
    try:
        fraction_idx = feature_names.index("fraction_optical")
    except ValueError as e:
        raise AssertionError(f"Could not find fraction_optical feature: {e}")

    # Extract the fraction values for each sample
    fractions = df.iloc[:, fraction_idx].values

    # Expected fractions
    expected = np.array([2/3, 4/6, 0/3])  # [0.666..., 0.666..., 0.0]

    # Check that we got the expected values (within tolerance)
    np.testing.assert_allclose(fractions, expected, rtol=1e-6,
                               err_msg="fraction_optical values do not match expected")


if __name__ == "__main__":
    print("Testing AquacultureFeatureEngineer...")
    test_basic_functionality()
    test_feature_groups()
    test_config()
    test_cloudy_month_within_window()
    test_feature_order_matches_transform()
    test_feature_order_with_temporal_stats()
    test_feature_order_consistency_across_configs()
    test_cross_sensor_division_by_zero()
    test_observation_window_from_sar_data()
    test_fraction_optical_calculation()
    print("All tests passed!")

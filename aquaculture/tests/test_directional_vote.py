"""
Tests for the directional vote features.
"""
import numpy as np
import pandas as pd
from aquaculture import AquacultureFeatureEngineer


def test_directional_vote_basic():
    """Test basic directional vote calculation."""
    # Create deterministic data: 1 sample, 12 months, 12 bands
    # We'll set values to get specific signs for the indices
    np.random.seed(42)
    X = np.ones((1, 12, 12), dtype=np.float64) * 0.1  # default small value

    # Band order: 0:VH,1:VV,2:blue,3:green,4:nir,5:nira,6:re1,7:re2,8:re3,9:red,10:swir1,11:swir2

    # Set values to control the signs of NDWI, MNDWI, NDRE2, NDVI
    # NDWI = (green - nir) / (green + nir)
    # MNDWI = (green - swir1) / (green + swir1)
    # NDRE2 = (nir - re2) / (nir + re2)
    # NDVI = (nir - red) / (nir + red)

    # Pattern for 3 months: [V=4, V=0, V=-4] repeating
    # Month 0: V = 4 (all positive)
    # Month 1: V = 0 (two +, two -)
    # Month 2: V = -4 (all negative)
    pattern_months = [
        # Month 0: V = 4
        (0.8, 0.2, 0.3, 0.9, 0.1, 0.5, 0.1),  # green, blue, nir, re2, re3, red, swir1
        # Month 1: V = 0
        (0.3, 0.2, 0.6, 0.1, 0.1, 0.8, 0.1),  # green, blue, nir, re2, re3, red, swir1
        # Month 2: V = -4
        (0.1, 0.2, 0.6, 0.3, 0.1, 0.4, 0.8)   # green, blue, nir, re2, re3, red, swir1
    ]

    # Apply the pattern to all 12 months
    for month in range(12):
        pattern_idx = month % 3
        green_val, blue_val, nir_val, re2_val, re3_val, red_val, swir1_val = pattern_months[pattern_idx]

        X[0, month, 3] = green_val  # green
        X[0, month, 2] = blue_val   # blue
        X[0, month, 4] = nir_val    # nir
        X[0, month, 7] = re2_val    # re2
        X[0, month, 8] = re3_val    # re3
        X[0, month, 9] = red_val    # red
        X[0, month, 10] = swir1_val # swir1

    # Set SAR bands to non-zero to avoid issues
    X[:, :, 0] = 0.3  # VH
    X[:, :, 1] = 0.4  # VV

    # Create feature engineer with directional vote enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # No masking for deterministic test
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=False,  # Disable for simpler checking
        include_cross_sensor_features=False,
        include_metadata=False,
        include_directional_vote=True
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    feature_names = list(fe.get_feature_names_out())

    # Find the directional vote features
    try:
        frac_pos_idx = feature_names.index("directional_vote_fraction_positive")
        frac_ge_2_idx = feature_names.index("directional_vote_fraction_ge_2")
        frac_eq_4_idx = feature_names.index("directional_vote_fraction_eq_4")
        mean_idx = feature_names.index("directional_vote_mean")
        min_idx = feature_names.index("directional_vote_min")
        max_idx = feature_names.index("directional_vote_max")
    except ValueError as e:
        raise AssertionError(f"Could not find directional vote features: {e}")

    # Get the values
    frac_positive = df.iloc[0, frac_pos_idx]
    frac_ge_2 = df.iloc[0, frac_ge_2_idx]
    frac_eq_4 = df.iloc[0, frac_eq_4_idx]
    mean_vote = df.iloc[0, mean_idx]
    min_vote = df.iloc[0, min_idx]
    max_vote = df.iloc[0, max_idx]

    # With our 12 months following the pattern [V=4, V=0, V=-4] repeated 4 times:
    # Months 0,3,6,9: V = 4 (all positive)
    # Months 1,4,7,10: V = 0 (two +, two -)
    # Months 2,5,8,11: V = -4 (all negative)
    #
    # Fraction positive: 4/12 = 1/3 (months with V > 0)
    # Fraction >= 2: 4/12 = 1/3 (months with V >= 2)
    # Fraction == 4: 4/12 = 1/3 (months with V == 4)
    # Mean: (4*4 + 0*4 + (-4)*4) / 12 = 0
    # Min: -4
    # Max: 4

    expected_frac_positive = 1/3
    expected_frac_ge_2 = 1/3
    expected_frac_eq_4 = 1/3
    expected_mean = 0.0
    expected_min = -4.0
    expected_max = 4.0

    tolerance = 1e-6
    assert abs(frac_positive - expected_frac_positive) < tolerance, \
        f"Fraction positive expected {expected_frac_positive}, got {frac_positive}"
    assert abs(frac_ge_2 - expected_frac_ge_2) < tolerance, \
        f"Fraction >=2 expected {expected_frac_ge_2}, got {frac_ge_2}"
    assert abs(frac_eq_4 - expected_frac_eq_4) < tolerance, \
        f"Fraction ==4 expected {expected_frac_eq_4}, got {frac_eq_4}"
    assert abs(mean_vote - expected_mean) < tolerance, \
        f"Mean vote expected {expected_mean}, got {mean_vote}"
    assert abs(min_vote - expected_min) < tolerance, \
        f"Min vote expected {expected_min}, got {min_vote}"
    assert abs(max_vote - expected_max) < tolerance, \
        f"Max vote expected {expected_max}, got {max_vote}"

    print("Basic directional vote test passed!")


def test_directional_vote_all_positive():
    """Test directional vote when all months have V=4."""
    # Create data where all months give V=4
    np.random.seed(42)
    n_samples = 2
    n_months = 12
    X = np.ones((n_samples, n_months, 12), dtype=np.float64) * 0.1

    # Set values for V=4 in all months: MNDWI>0, NDWI>0, NDRE2<0, NDVI<0
    # This means: green > nir, green > swir1, nir < re2, nir < red
    X[:, :, 3] = 0.8  # green
    X[:, :, 2] = 0.2  # blue
    X[:, :, 4] = 0.3  # nir
    X[:, :, 7] = 0.9  # re2 (large)
    X[:, :, 8] = 0.1  # re3 (small)
    X[:, :, 9] = 0.5  # red
    X[:, :, 10] = 0.1 # swir1 (small)

    # Set SAR bands
    X[:, :, 0] = 0.3  # VH
    X[:, :, 1] = 0.4  # VV

    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=False,
        include_temporal_statistics=False,
        include_cross_sensor_features=False,
        include_metadata=False,
        include_directional_vote=True
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    feature_names = list(fe.get_feature_names_out())

    # Get directional vote features
    frac_pos_idx = feature_names.index("directional_vote_fraction_positive")
    frac_ge_2_idx = feature_names.index("directional_vote_fraction_ge_2")
    frac_eq_4_idx = feature_names.index("directional_vote_fraction_eq_4")
    mean_idx = feature_names.index("directional_vote_mean")
    min_idx = feature_names.index("directional_vote_min")
    max_idx = feature_names.index("directional_vote_max")

    # All months should have V=4
    # Fraction positive: 12/12 = 1.0
    # Fraction >= 2: 12/12 = 1.0
    # Fraction == 4: 12/12 = 1.0
    # Mean: 4.0
    # Min: 4.0
    # Max: 4.0

    frac_positive = df.iloc[0, frac_pos_idx]
    frac_ge_2 = df.iloc[0, frac_ge_2_idx]
    frac_eq_4 = df.iloc[0, frac_eq_4_idx]
    mean_vote = df.iloc[0, mean_idx]
    min_vote = df.iloc[0, min_idx]
    max_vote = df.iloc[0, max_idx]

    assert abs(frac_positive - 1.0) < 1e-6
    assert abs(frac_ge_2 - 1.0) < 1e-6
    assert abs(frac_eq_4 - 1.0) < 1e-6
    assert abs(mean_vote - 4.0) < 1e-6
    assert abs(min_vote - 4.0) < 1e-6
    assert abs(max_vote - 4.0) < 1e-6

    # Test second sample too
    frac_positive_1 = df.iloc[1, frac_pos_idx]
    assert abs(frac_positive_1 - 1.0) < 1e-6

    print("All positive directional vote test passed!")


def test_directional_vote_disabled():
    """Test that directional vote features are not included when disabled."""
    np.random.seed(42)
    X = np.random.rand(2, 12, 12).astype(np.float64)

    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=False,
        include_directional_vote=False  # Explicitly disabled
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    feature_names = list(fe.get_feature_names_out())

    # Check that directional vote features are NOT present
    directional_vote_features = [
        "directional_vote_fraction_positive",
        "directional_vote_fraction_ge_2",
        "directional_vote_fraction_eq_4",
        "directional_vote_mean",
        "directional_vote_min",
        "directional_vote_max",
    ]

    for feature in directional_vote_features:
        assert feature not in feature_names, f"Unexpected feature found: {feature}"

    print("Directional vote disabled test passed!")


def test_directional_vote_feature_groups():
    """Test that directional vote features are in the correct group."""
    np.random.seed(42)
    X = np.random.rand(2, 12, 12).astype(np.float64)

    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_cross_sensor_features=True,
        include_metadata=True,
        include_directional_vote=True
    )

    fe.fit(X)
    df = fe.transform(X, training=False)
    groups = fe.feature_groups()

    # Should have 6 directional vote features
    assert len(groups['directional_vote']) == 6, \
        f"Expected 6 directional vote features, got {len(groups['directional_vote'])}"

    # Feature names should match
    directional_vote_names = [
        "directional_vote_fraction_positive",
        "directional_vote_fraction_ge_2",
        "directional_vote_fraction_eq_4",
        "directional_vote_mean",
        "directional_vote_min",
        "directional_vote_max",
    ]

    feature_names = fe.get_feature_names_out()
    for i, idx in enumerate(groups['directional_vote']):
        assert feature_names[idx] == directional_vote_names[i], \
            f"Directional vote feature {i} mismatch: expected {directional_vote_names[i]}, got {feature_names[idx]}"

    print("Directional vote feature groups test passed!")


if __name__ == "__main__":
    test_directional_vote_basic()
    test_directional_vote_all_positive()
    test_directional_vote_disabled()
    test_directional_vote_feature_groups()
    print("All directional vote tests passed!")
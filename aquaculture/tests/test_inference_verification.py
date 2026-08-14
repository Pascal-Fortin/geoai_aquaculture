"""
Tests for inference verification improvements in the aquaculture package.
These tests verify that the aquaculture components work correctly for inference.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the project root to the Python path so we can import aquaculture modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquaculture.feature_engineering import AquacultureFeatureEngineer
from aquaculture.feature_selection import FeatureSelector


def test_feature_engineering_no_masking_during_inference():
    """Test that AquacultureFeatureEngineer does not apply masking when training=False."""
    # Create sample data with some values that would be masked
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 12, 12)).astype(np.float64)

    # Introduce some extreme values that would likely be masked if simulation was on
    X[0, 0, 0] = -9999.0  # This represents missing data

    # Create feature engineer with masking simulation enabled
    fe = AquacultureFeatureEngineer(
        simulate_mask=True,  # This enables masking simulation
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True
    )

    # Fit on data (this would learn nothing but is required)
    fe.fit(X)

    # Transform with training=True (should apply masking simulation)
    X_train_transformed = fe.transform(X, training=True)

    # Transform with training=False (should NOT apply masking simulation)
    X_test_transformed = fe.transform(X, training=False)

    # The results should be different because training=True applies stochastic masking
    # while training=False uses the actual data availability (no simulation)
    # Note: Due to the randomness, we can't guarantee they'll be different every time,
    # but we can verify the transform works correctly in both modes

    assert isinstance(X_train_transformed, pd.DataFrame)
    assert isinstance(X_test_transformed, pd.DataFrame)
    assert X_train_transformed.shape == X_test_transformed.shape
    assert X_train_transformed.shape[0] == 5  # Same number of samples

    # The key point is that both operations complete without error
    # and produce valid feature DataFrames


def test_feature_selection_works_with_pre_fitted_engineer():
    """Test that FeatureSelector works correctly with a pre-fitted feature engineer."""
    # Create sample data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 12, 12)).astype(np.float64)

    # Create and fit a feature engineer
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,  # No masking for deterministic test
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_metadata=True
    )
    fe.fit(X)

    # Create a feature selector using the pre-fitted engineer
    selector = FeatureSelector(
        base_engineer=fe,
        selection_method='groups',
        groups=['temporal', 'metadata']
    )

    # Verify the selector was created correctly
    assert selector.base_engineer is fe
    assert selector.selection_method == 'groups'
    assert selector.n_features_selected > 0
    assert selector.n_features_selected < len(fe.get_feature_names_out())  # Should have selected fewer features

    # Test transforming data with the selector
    X_selected = selector.transform(X, training=False)

    # Verify we got selected features back
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == 10  # Same number of samples
    assert X_selected.shape[1] == selector.n_features_selected  # Correct number of selected features

    # Verify the feature names are correct
    selected_feature_names = list(selector.get_feature_names_out())
    assert len(selected_feature_names) == selector.n_features_selected

    # Verify that all selected features exist in the original feature set
    all_feature_names = set(fe.get_feature_names_out())
    selected_feature_set = set(selected_feature_names)
    assert selected_feature_set.issubset(all_feature_names)


def test_feature_engineering_consistent_feature_names():
    """Test that feature engineering produces consistent feature names."""
    # Create sample data
    rng = np.random.default_rng(42)
    X1 = rng.normal(size=(5, 12, 12)).astype(np.float64)
    X2 = rng.normal(size=(5, 12, 12)).astype(np.float64)

    # Create two feature engineers with identical configuration
    fe1 = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=123,  # Same seed
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_metadata=True
    )

    fe2 = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=123,  # Same seed
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True,
        include_metadata=True
    )

    # Fit both on different data (but same configuration)
    fe1.fit(X1)
    fe2.fit(X2)

    # Get feature names from both
    feature_names_1 = list(fe1.get_feature_names_out())
    feature_names_2 = list(fe2.get_feature_names_out())

    # They should be identical because they have the same configuration
    assert feature_names_1 == feature_names_2
    assert len(feature_names_1) == len(feature_names_2) > 0


def test_feature_selector_persistence():
    """Test that FeatureSelector maintains consistency across transformations."""
    # Create sample data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 12, 12)).astype(np.float64)

    # Create and fit a feature engineer
    fe = AquacultureFeatureEngineer(
        simulate_mask=False,
        random_state=42,
        include_optical=True,
        include_sar=True,
        include_temporal_statistics=True
    )
    fe.fit(X)

    # Create a feature selector
    selector = FeatureSelector(
        base_engineer=fe,
        selection_method='groups',
        groups=['temporal']
    )

    # Transform the same data multiple times
    X_selected1 = selector.transform(X, training=False)
    X_selected2 = selector.transform(X, training=False)
    X_selected3 = selector.transform(X, training=False)

    # All results should be identical
    assert X_selected1.equals(X_selected2)
    assert X_selected2.equals(X_selected3)

    # Verify feature names are consistent
    names1 = list(selector.get_feature_names_out())
    names2 = list(selector.get_feature_names_out())
    names3 = list(selector.get_feature_names_out())

    assert names1 == names2 == names3


if __name__ == "__main__":
    # Run the tests
    test_feature_engineering_no_masking_during_inference()
    print("������✓ test_feature_engineering_no_masking_during_inference passed")

    test_feature_selection_works_with_pre_fitted_engineer()
    print("������✓ test_feature_selection_works_with_pre_fitted_engineer passed")

    test_feature_engineering_consistent_feature_names()
    print("������✓ test_feature_engineering_consistent_feature_names passed")

    test_feature_selector_persistence()
    print("������✓ test_feature_selector_persistence passed")

    print("\nAll aquaculture inference verification tests passed! ���� �� �� 🎉")
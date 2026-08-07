"""
Tests for the FeatureSelector utility
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
import re

# Add the project root to the Python path so we can import aquaculture modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aquaculture.feature_selection import FeatureSelector, select_temporal_features, select_metadata_features
from aquaculture.feature_engineering import AquacultureFeatureEngineer


class TestFeatureSelector:
    """Test the FeatureSelector utility class"""

    def setup_method(self):
        """Set up test data and fitted feature engineer for each test"""
        # Create simple test data
        np.random.seed(42)
        self.X = np.random.rand(10, 12, 12).astype(np.float64)

        # Create and fit a feature engineer
        self.fe = AquacultureFeatureEngineer(
            simulate_mask=False,
            include_optical=True,
            include_sar=True,
            include_cross_sensor_features=True,
            include_temporal_statistics=True,
            include_metadata=True
        )
        self.fe.fit(self.X)

        # Verify we have features to work with
        assert len(self.fe.get_feature_names_out()) > 0
        assert self.fe.feature_groups()  # Should have group breakdown

    def test_initialization_with_groups(self):
        """Test initialization with group-based selection"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal', 'metadata']
        )

        # Should have selected some features
        assert selector.n_features_selected > 0
        assert selector.n_features_selected < len(self.fe.get_feature_names_out())

        # Should have the right feature names
        selected_names = selector.get_feature_names_out()
        assert len(selected_names) == selector.n_features_selected

        # Check that selected features are actually temporal or metadata
        groups = self.fe.feature_groups()
        temporal_indices = set(groups.get('temporal', []))
        metadata_indices = set(groups.get('metadata', []))
        selected_indices = set(selector._selected_indices)

        # All selected indices should be in either temporal or metadata
        for idx in selected_indices:
            assert idx in temporal_indices or idx in metadata_indices

    def test_initialization_with_names(self):
        """Test initialization with specific feature names"""
        all_names = self.fe.get_feature_names_out()
        # Pick a few actual feature names
        test_names = all_names[:3].tolist() if len(all_names) >= 3 else all_names.tolist()

        selector = FeatureSelector(
            self.fe,
            selection_method='names',
            names=test_names
        )

        assert selector.n_features_selected == len(test_names)
        selected_names = selector.get_feature_names_out()
        assert set(selected_names) == set(test_names)

    def test_initialization_with_patterns(self):
        """Test initialization with feature name patterns"""
        # Select features ending with '_mean'
        selector = FeatureSelector(
            self.fe,
            selection_method='patterns',
            patterns=['_mean$']
        )

        assert selector.n_features_selected > 0
        selected_names = selector.get_feature_names_out()

        # All selected names should end with '_mean'
        for name in selected_names:
            assert name.endswith('_mean')

    def test_initialization_with_indices(self):
        """Test initialization with feature indices"""
        n_total = len(self.fe.get_feature_names_out())
        test_indices = [0, 1, 2] if n_total >= 3 else list(range(n_total))

        selector = FeatureSelector(
            self.fe,
            selection_method='indices',
            indices=test_indices
        )

        assert selector.n_features_selected == len(test_indices)
        selected_names = selector.get_feature_names_out()
        all_names = self.fe.get_feature_names_out()
        assert set(selected_names) == set(all_names[i] for i in test_indices)

    def test_transform_returns_correct_shape(self):
        """Test that transform returns DataFrame with correct shape"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        # Transform some test data
        X_test = self.X[:5]  # First 5 samples
        X_selected = selector.transform(X_test, training=False)

        # Should be DataFrame with same number of rows, selected columns
        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X_test.shape[0]
        assert X_selected.shape[1] == selector.n_features_selected

        # Column names should match selected features
        assert list(X_selected.columns) == selector.get_feature_names_out().tolist()

    def test_transform_with_training_flag(self):
        """Test that transform respects training flag"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        X_test = self.X[:3]

        # Both should work without error
        X_train_transform = selector.transform(X_test, training=True)
        X_test_transform = selector.transform(X_test, training=False)

        # Should have same shape
        assert X_train_transform.shape == X_test_transform.shape
        assert X_train_transform.shape[1] == selector.n_features_selected

    def test_get_feature_names_out_returns_correct(self):
        """Test that get_feature_names_out returns correct names"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        selected_names = selector.get_feature_names_out()
        assert isinstance(selected_names, np.ndarray)
        # String arrays have specific dtypes like '<U23', not generic 'O'
        assert selected_names.dtype.kind in ('U', 'S', 'O')  # Unicode, byte string, or object
        assert len(selected_names) == selector.n_features_selected

    def test_n_features_selected_property(self):
        """Test n_features_selected property"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        assert selector.n_features_selected == len(selector.get_feature_names_out())
        assert selector.n_features_selected >= 0

    def test_feature_groups_in_selection_property(self):
        """Test feature_groups_in_selection property"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal', 'metadata']
        )

        group_counts = selector.feature_groups_in_selection
        assert isinstance(group_counts, dict)

        # Should have entries for temporal and metadata (if they have features)
        total_from_groups = sum(group_counts.values())
        assert total_from_groups == selector.n_features_selected

    def test_get_selection_summary(self):
        """Test get_selection_summary method"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        summary = selector.get_selection_summary()
        assert isinstance(summary, str)
        assert str(selector.n_features_selected) in summary
        assert 'temporal' in summary.lower()

    def test_empty_selection(self):
        """Test behavior when no features are selected"""
        selector = FeatureSelector(
            self.fe,
            selection_method='names',
            names=['nonexistent_feature_12345']  # This feature doesn't exist
        )

        assert selector.n_features_selected == 0
        assert len(selector.get_feature_names_out()) == 0

        # Transform should return empty DataFrame with correct index
        X_test = self.X[:3]
        X_selected = selector.transform(X_test, training=False)
        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X_test.shape[0]
        assert X_selected.shape[1] == 0

    def test_select_all_features(self):
        """Test selecting all features"""
        all_names = self.fe.get_feature_names_out()
        selector = FeatureSelector(
            self.fe,
            selection_method='names',
            names=all_names.tolist()
        )

        assert selector.n_features_selected == len(all_names)
        selected_names = selector.get_feature_names_out()
        assert set(selected_names) == set(all_names)

        # Transform should return same as base engineer (approximately)
        X_test = self.X[:3]
        X_selected = selector.transform(X_test, training=False)
        X_all = self.fe.transform(X_test, training=False)

        # Should have same shape and values
        assert X_selected.shape == X_all.shape
        np.testing.assert_array_equal(X_selected.values, X_all.values)

    def test_select_temporal_features_convenience(self):
        """Test the select_temporal_features convenience function"""
        selector = select_temporal_features(self.fe)

        assert selector.selection_method == 'groups'
        assert selector.selection_kwargs.get('groups') == ['temporal']
        assert selector.n_features_selected > 0

        # Verify all selected features are actually temporal
        groups = self.fe.feature_groups()
        temporal_indices = set(groups.get('temporal', []))
        selected_indices = set(selector._selected_indices)

        for idx in selected_indices:
            assert idx in temporal_indices

    def test_select_metadata_features_convenience(self):
        """Test the select_metadata_features convenience function"""
        selector = select_metadata_features(self.fe)

        assert selector.selection_method == 'groups'
        assert selector.selection_kwargs.get('groups') == ['metadata']
        assert selector.n_features_selected > 0

        # Verify all selected features are actually metadata
        groups = self.fe.feature_groups()
        metadata_indices = set(groups.get('metadata', []))
        selected_indices = set(selector._selected_indices)

        for idx in selected_indices:
            assert idx in metadata_indices

    def test_combine_method_include_only(self):
        """Test combine method with only include criteria"""
        selector = FeatureSelector(
            self.fe,
            selection_method='combine',
            include={'method': 'groups', 'groups': ['temporal']}
        )

        # Should be equivalent to selecting just temporal groups
        expected_selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        assert selector.n_features_selected == expected_selector.n_features_selected
        assert set(selector.get_feature_names_out()) == set(expected_selector.get_feature_names_out())

    def test_combine_method_include_exclude(self):
        """Test combine method with both include and exclude"""
        # Start with all temporal features
        # Then exclude a few specific ones
        all_temporal = select_temporal_features(self.fe)
        if all_temporal.n_features_selected >= 3:
            # Get first 3 temporal feature names to exclude
            temporal_names = all_temporal.get_feature_names_out()
            names_to_exclude = temporal_names[:3].tolist()

            selector = FeatureSelector(
                self.fe,
                selection_method='combine',
                include={'method': 'groups', 'groups': ['temporal']},
                exclude={'method': 'names', 'names': names_to_exclude}
            )

            # Should have fewer than all temporal features
            assert selector.n_features_selected < all_temporal.n_features_selected
            assert selector.n_features_selected >= 0

            # Excluded names should not be in selection
            selected_names = set(selector.get_feature_names_out())
            excluded_names = set(names_to_exclude)
            assert len(selected_names.intersection(excluded_names)) == 0

            # All selected should still be temporal
            groups = self.fe.feature_groups()
            temporal_indices = set(groups.get('temporal', []))
            selected_indices = set(selector._selected_indices)
            for idx in selected_indices:
                assert idx in temporal_indices

    def test_combine_method_include_exclude_patterns(self):
        """Test combine method with include/exclude using patterns"""
        # Include all features, then exclude optical band prefixes
        selector = FeatureSelector(
            self.fe,
            selection_method='combine',
            include={'method': 'patterns', 'patterns': ['.*']},  # Include all
            exclude={'method': 'patterns', 'patterns': [
                'green_.*',   # Features starting with green_
                'nir_.*',     # Features starting with nir_
                'nira_.*',    # Features starting with nira_
                'swir1_.*',   # Features starting with swir1_
                'swir2_.*'    # Features starting with swir2_
            ]}
        )

        # Should have selected some features
        assert selector.n_features_selected > 0
        assert selector.n_features_selected < len(self.fe.get_feature_names_out())

        # Get selected feature names
        selected_names = set(selector.get_feature_names_out())
        all_names = set(self.fe.get_feature_names_out())

        # Excluded prefixes should not be in selection
        excluded_prefixes = ['green_', 'nir_', 'nira_', 'swir1_', 'swir2_']
        for prefix in excluded_prefixes:
            # Check that no selected name starts with this prefix
            for name in selected_names:
                assert not name.startswith(prefix), f"Feature {name} should be excluded (starts with {prefix})"

        # All excluded features should not be in selection
        excluded_features = set()
        for name in all_names:
            for prefix in excluded_prefixes:
                if name.startswith(prefix):
                    excluded_features.add(name)
                    break

        # Selected and excluded sets should be disjoint
        assert len(selected_names.intersection(excluded_features)) == 0

        # Selected + excluded should be less than or equal to all features
        # (there may be features that are neither selected nor excluded)
        assert len(selected_names.union(excluded_features)) <= len(all_names)

    def test_custom_selection_function(self):
        """Test custom selection function"""
        # Select features with 'mean' in the name
        selector = FeatureSelector(
            self.fe,
            selection_method='custom',
            function=lambda names: np.array(['mean' in name for name in names])
        )

        assert selector.n_features_selected > 0
        selected_names = selector.get_feature_names_out()

        # All selected names should contain 'mean'
        for name in selected_names:
            assert 'mean' in name

    def test_selector_is_not_fitted(self):
        """Test that selector doesn't require fitting (but base engineer must be fitted)"""
        # Selector itself doesn't have fit/transform semantics in the ML sense
        # It just wraps the base engineer
        selector = FeatureSelector(self.fe, 'groups', groups=['temporal'])

        # Calling fit on selector should just return self (no-op)
        result = selector.fit(self.X)
        assert result is selector

    def test_error_on_unfitted_base_engineer(self):
        """Test that we get appropriate error if base engineer isn't fitted"""
        # Create unfitted engineer
        unfitted_fe = AquacultureFeatureEngineer(simulate_mask=False)

        with pytest.raises(ValueError, match="Base engineer must be fitted"):
            FeatureSelector(unfitted_fe, 'groups', groups=['temporal'])

    def test_invalid_selection_method(self):
        """Test that invalid selection method raises error"""
        with pytest.raises(ValueError, match="Unknown selection method"):
            FeatureSelector(self.fe, 'invalid_method')

    def test_custom_method_missing_function(self):
        """Test that custom method requires function parameter"""
        with pytest.raises(ValueError, match="Custom method requires 'function' parameter"):
            FeatureSelector(self.fe, 'custom')

    def test_selection_preserves_order(self):
        """Test that selection preserves original feature order"""
        selector = FeatureSelector(
            self.fe,
            selection_method='names',
            names=['VH_01', 'NDVI_01', 'temperature_mean']  # Mix of early, middle, late if they exist
        )

        # Filter to only names that actually exist
        all_names = self.fe.get_feature_names_out()
        existing_names = [name for name in ['VH_01', 'NDVI_01', 'temperature_mean'] if name in all_names]

        if len(existing_names) > 1:
            selector = FeatureSelector(
                self.fe,
                selection_method='names',
                names=existing_names
            )

            selected_names = selector.get_feature_names_out()
            # Should be in same order as in original feature list
            original_order = [name for name in all_names if name in existing_names]
            assert selected_names.tolist() == original_order

    def test_selector_reuse(self):
        """Test that selector can be reused on different data"""
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        # Transform different datasets
        X1 = self.X[:3]
        X2 = self.X[3:6]

        result1 = selector.transform(X1, training=False)
        result2 = selector.transform(X2, training=False)

        # Should have same number of features
        assert result1.shape[1] == result2.shape[1] == selector.n_features_selected

        # Should have correct number of rows
        assert result1.shape[0] == X1.shape[0]
        assert result2.shape[0] == X2.shape[0]

    def test_wildcard_match_method(self):
        """Test the _wildcard_match helper method directly"""
        # Create a FeatureSelector to access _wildcard_match (method doesn't matter for accessing helper)
        selector = FeatureSelector(
            self.fe,
            selection_method='groups',
            groups=['temporal']
        )

        # Test cases: (pattern, text, expected_result, description)
        test_cases = [
            # (*_01 patterns)
            ('*_01', 'green_01', True, "*_01 should match green_01"),
            ('*_01', 'green_02', False, "*_01 should not match green_02"),
            ('*_01', 'abc_01', True, "*_01 should match abc_01"),
            ('*_01', '01', False, "*_01 should not match 01 (need something before _)"),
            # (green_* patterns)
            ('green_*', 'green_01', True, "green_* should match green_01"),
            ('green_*', 'green_abc', True, "green_* should match green_abc"),
            ('green_*', 'green_', True, "green_* should match green_ (zero characters after)"),
            ('green_*', 'agr_01', False, "green_* should not match agr_01"),
            # (? patterns - single character)
            ('a?c', 'abc', True, "a?c should match abc"),
            ('a?c', 'ac', False, "a?c should not match ac (too short)"),
            ('a?c', 'abbc', False, "a?c should not match abbc (too long)"),
            ('a?c', 'axc', True, "a?c should match axc"),
            # Mixed patterns
            ('*_mean*', 'green_mean_01', True, "*_mean* should match green_mean_01"),
            ('*_mean*', 'green_01', False, "*_mean* should not match green_01"),
            # Edge cases
            ('*', 'anything', True, "* should match anything"),
            ('*', '', True, "* should match empty string"),
            ('?', 'a', True, "? should match single character"),
            ('?', 'ab', False, "? should not match two characters"),
            ('?', '', False, "? should not match empty string"),
        ]

        all_passed = True
        for pattern, text, expected, description in test_cases:
            try:
                result = selector._wildcard_match(text, pattern)
                if result == expected:
                    pass  # Test passed
                else:
                    print(f"FAIL: {description}")
                    print(f"      Pattern: '{pattern}', Text: '{text}', Expected: {expected}, Got: {result}")
                    all_passed = False
            except Exception as e:
                print(f"ERROR: {description}")
                print(f"      Pattern: '{pattern}', Text: '{text}', Error: {e}")
                all_passed = False

        # Assert all tests passed
        assert all_passed, "One or more _wildcard_match tests failed"

    def test_patterns_method_with_wildcard_patterns(self):
        """Test the patterns method with wildcard patterns that trigger fallback path"""
        # Test pattern '*_01' (should match features ending with _01)
        selector = FeatureSelector(
            self.fe,
            selection_method='patterns',
            patterns=['*_01']  # Wildcard pattern: any sequence followed by _01
        )

        # Should have selected some features
        assert selector.n_features_selected > 0
        selected_names = set(selector.get_feature_names_out())

        # All selected names should end with '_01'
        for name in selected_names:
            assert name.endswith('_01'), f"Feature {name} should end with '_01'"

        # Verify we actually got the expected matches by checking against manual regex
        import re
        # Manual wildcard conversion: *_01 -> .*_01
        manual_pattern = '.*_01'
        manual_regex = re.compile(manual_pattern)
        expected_matches = {name for name in self.fe.get_feature_names_out() if manual_regex.search(name)}

        assert selected_names == expected_matches, f"Selected names {selected_names} should equal expected matches {expected_matches}"

    def test_patterns_method_with_mixed_patterns(self):
        """Test the patterns method with both regex and wildcard patterns"""
        # Mix of regex pattern (ending with _mean) and wildcard pattern (starting with green_)
        selector = FeatureSelector(
            self.fe,
            selection_method='patterns',
            patterns=[
                '_mean$',   # Regex: features ending with _mean
                'green_*'   # Wildcard: features starting with green_
            ]
        )

        # Should have selected some features
        assert selector.n_features_selected > 0
        selected_names = set(selector.get_feature_names_out())

        # All selected names should either end with _mean OR start with green_
        for name in selected_names:
            assert name.endswith('_mean') or name.startswith('green_'), \
                f"Feature {name} should either end with '_mean' or start with 'green_'"

        # Verify we got the expected matches
        import re
        mean_regex = re.compile('_mean$')
        green_pattern = 'green_.*'  # wildcard conversion
        green_regex = re.compile(green_pattern)

        expected_names = set()
        for name in self.fe.get_feature_names_out():
            if mean_regex.search(name) or green_regex.search(name):
                expected_names.add(name)

        assert selected_names == expected_names, f"Selected names {selected_names} should equal expected matches {expected_names}"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
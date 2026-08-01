"""
Tests for SHAP functionality in the evaluate module.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from src.evaluate import get_shap_values, get_shap_feature_importance, save_shap_analysis


def test_get_shap_feature_importance():
    """Test SHAP feature importance calculation."""
    # Create mock SHAP values
    shap_values = np.array([
        [0.1, -0.2, 0.3],
        [-0.1, 0.2, -0.3],
        [0.2, -0.1, 0.4]
    ])
    feature_names = ['feature_a', 'feature_b', 'feature_c']

    # Calculate importance
    importance_df = get_shap_feature_importance(shap_values, feature_names)

    # Check that we get a DataFrame with correct columns
    assert isinstance(importance_df, pd.DataFrame)
    assert list(importance_df.columns) == ['feature', 'importance']
    assert len(importance_df) == 3

    # Check that features are sorted by importance descending
    # feature_a: [0.1, -0.1, 0.2] -> abs: [0.1, 0.1, 0.2] -> mean: 0.133
    # feature_b: [-0.2, 0.2, -0.1] -> abs: [0.2, 0.2, 0.1] -> mean: 0.167
    # feature_c: [0.3, -0.3, 0.4] -> abs: [0.3, 0.3, 0.4] -> mean: 0.333
    expected_order = ['feature_c', 'feature_b', 'feature_a']
    assert list(importance_df['feature']) == expected_order

    # Check importance values (mean absolute SHAP values)
    expected_importance = [0.3333333333333333, 0.16666666666666666, 0.13333333333333333]  # approx
    np.testing.assert_allclose(importance_df['importance'].values, expected_importance, rtol=1e-10)


def test_get_shap_feature_importance_single_feature():
    """Test SHAP feature importance with single feature."""
    shap_values = np.array([[0.5], [-0.3], [0.2]])
    feature_names = ['single_feature']

    importance_df = get_shap_feature_importance(shap_values, feature_names)

    assert len(importance_df) == 1
    assert importance_df.iloc[0]['feature'] == 'single_feature'
    assert abs(importance_df.iloc[0]['importance'] - 0.3333333333333333) < 1e-10


@patch('shap.TreeExplainer')
def test_get_shap_values_tree_model(mock_tree_explainer):
    """Test SHAP value computation for tree-based models."""
    # Setup mock
    mock_explainer_instance = Mock()
    mock_explainer_instance.shap_values.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_tree_explainer.return_value = mock_explainer_instance

    # Create mock model
    model = Mock()
    model.__class__.__name__ = 'LGBMClassifier'  # Tree-based model

    # Test data
    X = np.array([[1, 2], [3, 4]])
    feature_names = ['feat1', 'feat2']

    # Call function
    result = get_shap_values(model, X, feature_names, sample_size=10)

    # Verify
    mock_tree_explainer.assert_called_once_with(model)
    # For binary classification, should return positive class shap values
    expected = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(result, expected)


@patch('shap.TreeExplainer')
def test_get_shap_values_returns_list(mock_tree_explainer):
    """Test SHAP value computation when shap_values returns a list (multi-class)."""
    # Setup mock
    mock_explainer_instance = Mock()
    # Return list of arrays (one per class) - 3 classes for multi-class
    mock_explainer_instance.shap_values.return_value = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),  # Class 0
        np.array([[0.5, 0.6], [0.7, 0.8]]),  # Class 1
        np.array([[0.9, 0.1], [0.2, 0.3]])   # Class 2
    ]
    mock_tree_explainer.return_value = mock_explainer_instance

    # Create mock model
    model = Mock()
    model.__class__.__name__ = 'LGBMClassifier'

    # Test data
    X = np.array([[1, 2], [3, 4]])
    feature_names = ['feat1', 'feat2']

    # Call function
    result = get_shap_values(model, X, feature_names, sample_size=10)

    # Should return the mean absolute across classes for feature importance
    # Mean of abs([[0.1,0.2],[0.3,0.4]]), abs([[0.5,0.6],[0.7,0.8]]), abs([[0.9,0.1],[0.2,0.3]])
    # = Mean of [[0.1,0.2],[0.3,0.4]], [[0.5,0.6],[0.7,0.8]], [[0.9,0.1],[0.2,0.3]]
    # = [[0.5,0.3], [0.4,0.5]]
    expected = np.array([[0.5, 0.3], [0.4, 0.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_get_shap_values_import_error():
    """Test SHAP value computation when shap is not installed."""
    # Temporarily remove shap from sys.modules
    import sys
    original_shap = sys.modules.get('shap')

    # Remove shap from sys.modules if it exists
    if 'shap' in sys.modules:
        del sys.modules['shap']

    # Also prevent importing it by mocking the import
    with patch.dict(sys.modules, {'shap': None}):
        try:
            model = Mock()
            X = np.array([[1, 2]])
            feature_names = ['feat1', 'feat2']

            with pytest.raises(ImportError, match="SHAP is not available"):
                get_shap_values(model, X, feature_names)
        finally:
            # Restore shap if it was originally there
            if original_shap is not None:
                sys.modules['shap'] = original_shap


@patch('src.plotting.plot_shap_summary')
@patch('src.plotting.plot_shap_dependence')
def test_save_shap_analysis(mock_plot_dependence, mock_plot_summary):
    """Test saving SHAP analysis."""
    # Create test data
    shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = np.array([[1, 2], [3, 4]])
    feature_names = ['feat1', 'feat2']
    experiment_dir = "/tmp/test_experiment"

    # Call function
    save_shap_analysis(shap_values, X, feature_names, experiment_dir,
                      plot_types=["dot"], max_display=2)

    # Verify that the plotting functions were called
    mock_plot_summary.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
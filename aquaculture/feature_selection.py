"""
Feature selection utilities for the aquaculture feature engineering pipeline.
Provides flexible ways to select subsets of features for modeling while
preserving the full feature calculation for analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import re
from typing import List, Callable, Union, Dict, Any, Optional


class FeatureSelector:
    """
    Flexible feature selector that works with any fitted feature engineer.
    Allows precise control over which features to keep for modeling.

    This selector works with a fitted AquacultureFeatureEngineer (or any
    similar feature engineer with feature_names_out_ and feature_groups() methods)
    to select arbitrary subsets of features for modeling without modifying
    the original feature calculation code.

    Examples
    --------
    >>> # Keep only temporal and metadata features
    >>> selector = FeatureSelector(
    ...     base_engineer=fitted_fe,
    ...     selection_method='groups',
    ...     groups=['temporal', 'metadata']
    ... )
    >>> X_selected = selector.transform(X_data)

    >>> # Keep only features matching specific patterns
    >>> selector = FeatureSelector(
    ...     base_engineer=fitted_fe,
    ...     selection_method='patterns',
    ...     patterns=['_mean$', '_std$']  # mean and std features only
    ... )

    >>> # Complex selection: temporal + metadata + specific optical bands
    >>> selector = FeatureSelector(
    ...     base_engineer=fitted_fe,
    ...     selection_method='combine',
    ...     include={'method': 'groups', 'groups': ['temporal', 'metadata']},
    ...     exclude={'method': 'names', 'names': ['green_01', 'nir_01']}
    ... )
    """

    def __init__(self, base_engineer, selection_method: str = 'groups', **selection_kwargs):
        """
        Initialize selector with a fitted feature engineer.

        Parameters
        ----------
        base_engineer : fitted feature engineer
            Must be already fitted (has feature_names_out_ and feature_groups() methods)
        selection_method : str
            How to select features: 'groups', 'names', 'patterns', 'indices', 'custom', or 'combine'
        **selection_kwargs :
            Selection criteria depending on method:
            - groups: list of group names to keep (e.g., ['temporal', 'metadata'])
            - names: list of specific feature names to keep
            - patterns: list of regex/wildcard patterns to match feature names
            - indices: list of feature indices to keep
            - custom: function that takes feature names and returns boolean mask
            - combine: dict with keys 'include' and 'exclude', each containing above criteria
        """
        self.base_engineer = base_engineer
        self.selection_method = selection_method
        self.selection_kwargs = selection_kwargs
        self._selected_indices = None
        self._selected_names = None

        # Validate that engineer is fitted
        if not hasattr(base_engineer, 'feature_names_out_') or base_engineer.feature_names_out_ is None:
            raise ValueError("Base engineer must be fitted before creating selector")

        self._compute_selection()

    def _compute_selection(self):
        """Compute which feature indices to keep based on selection method."""
        all_names = np.array(self.base_engineer.get_feature_names_out())
        n_features = len(all_names)

        if self.selection_method == 'groups':
            # Select by feature groups
            groups_to_keep = self.selection_kwargs.get('groups', [])
            groups = self.base_engineer.feature_groups()

            selected_indices = []
            for group in groups_to_keep:
                if group in groups:
                    selected_indices.extend(groups[group])
            self._selected_indices = np.sort(selected_indices)

        elif self.selection_method == 'names':
            # Select by specific feature names
            names_to_keep = self.selection_kwargs.get('names', [])
            name_to_idx = {name: i for i, name in enumerate(all_names)}
            self._selected_indices = np.sort([
                name_to_idx[name] for name in names_to_keep
                if name in name_to_idx
            ])

        elif self.selection_method == 'patterns':
            # Select by patterns (regex or wildcard)
            patterns = self.selection_kwargs.get('patterns', [])
            flags = self.selection_kwargs.get('flags', 0)

            selected_indices = []
            for pattern in patterns:
                try:
                    # Try as regex first
                    regex = re.compile(pattern, flags)
                    matches = [i for i, name in enumerate(all_names) if regex.search(name)]
                except re.error:
                    # Fall back to wildcard matching
                    matches = [i for i, name in enumerate(all_names)
                             if self._wildcard_match(name, pattern)]
                selected_indices.extend(matches)
            self._selected_indices = np.sort(selected_indices)

        elif self.selection_method == 'indices':
            # Select by direct indices
            indices = self.selection_kwargs.get('indices', [])
            self._selected_indices = np.sort([i for i in indices if 0 <= i < len(all_names)])

        elif self.selection_method == 'custom':
            # Select by custom function
            selector_func = self.selection_kwargs.get('function')
            if selector_func is None:
                raise ValueError("Custom method requires 'function' parameter")
            mask = selector_func(all_names)
            self._selected_indices = np.where(mask)[0]

        elif self.selection_method == 'combine':
            # Combine multiple selection methods with include/exclude logic
            include_criteria = self.selection_kwargs.get('include', {})
            exclude_criteria = self.selection_kwargs.get('exclude', {})

            # Start with all features
            include_mask = np.ones(len(all_names), dtype=bool)
            exclude_mask = np.zeros(len(all_names), dtype=bool)

            # Process include criteria
            if include_criteria:
                include_selector = FeatureSelector(
                    self.base_engineer,
                    selection_method=include_criteria.get('method', 'groups'),
                    **{k: v for k, v in include_criteria.items() if k != 'method'}
                )
                include_mask = np.zeros(len(all_names), dtype=bool)
                include_mask[include_selector._selected_indices] = True

            # Process exclude criteria
            if exclude_criteria:
                exclude_selector = FeatureSelector(
                    self.base_engineer,
                    selection_method=exclude_criteria.get('method', 'groups'),
                    **{k: v for k, v in exclude_criteria.items() if k != 'method'}
                )
                exclude_mask = np.zeros(len(all_names), dtype=bool)
                exclude_mask[exclude_selector._selected_indices] = True

            # Final selection: included AND not excluded
            final_mask = include_mask & (~exclude_mask)
            self._selected_indices = np.where(final_mask)[0]
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        self._selected_names = all_names[self._selected_indices] if len(self._selected_indices) > 0 else np.array([])

    def _wildcard_match(self, text: str, pattern: str) -> bool:
        """Simple wildcard matching (* matches any sequence, ? matches any single character)"""
        # Escape regex special chars except * and ?
        pattern = re.escape(pattern)
        pattern = pattern.replace(r'\*', '.*').replace(r'\?', '.')
        return bool(re.match(pattern, text))

    def fit(self, X, y=None):
        """Selector doesn't need fitting - just pass through"""
        return self

    def transform(self, X, training: bool = False) -> pd.DataFrame:
        """Transform data and return only selected features

        Parameters
        ----------
        X : array-like
            Input data to transform
        training : bool, default=False
            Whether to apply training transformations (e.g., masking simulation)

        Returns
        -------
        X_selected : pd.DataFrame
            DataFrame containing only the selected features
        """
        # Get all features from base engineer
        X_all = self.base_engineer.transform(X, training=training)

        # Return only selected features
        if len(self._selected_indices) == 0:
            # Return empty DataFrame with same index
            return pd.DataFrame(index=X_all.index)
        elif len(self._selected_indices) == len(self.base_engineer.get_feature_names_out()):
            # Return all features (optimization)
            return X_all
        else:
            return X_all.iloc[:, self._selected_indices]

    def get_feature_names_out(self) -> np.ndarray:
        """Get names of selected features

        Returns
        -------
        feature_names : ndarray
            Names of the selected features
        """
        return self._selected_names.copy()

    @property
    def n_features_selected(self) -> int:
        """Number of selected features"""
        return len(self._selected_indices)

    @property
    def feature_groups_in_selection(self) -> Dict[str, int]:
        """Get breakdown of which groups are represented in selection

        Returns
        -------
        group_counts : dict
            Dictionary mapping group names to count of features from that group
        """
        groups = self.base_engineer.feature_groups()
        group_counts = {}
        for group_name, indices in groups.items():
            overlap = np.intersect1d(self._selected_indices, indices)
            if len(overlap) > 0:
                group_counts[group_name] = len(overlap)
        return group_counts

    def get_selection_summary(self) -> str:
        """Get a human-readable summary of the selection

        Returns
        -------
        summary : str
            Description of what features were selected
        """
        if self.n_features_selected == 0:
            return "No features selected"

        groups = self.feature_groups_in_selection
        if not groups:
            return f"{self.n_features_selected} features selected (custom selection)"

        group_desc = ", ".join([f"{count} {group}" for group, count in groups.items()])
        return f"{self.n_features_selected} features selected: {group_desc}"


# Utility functions for common selection patterns
def select_temporal_features(base_engineer) -> FeatureSelector:
    """Convenience function to select only temporal features"""
    return FeatureSelector(base_engineer, 'groups', groups=['temporal'])


def select_metadata_features(base_engineer) -> FeatureSelector:
    """Convenience function to select only metadata features"""
    return FeatureSelector(base_engineer, 'groups', groups=['metadata'])


def select_temporal_and_metadata(base_engineer) -> FeatureSelector:
    """Convenience function to select temporal and metadata features"""
    return FeatureSelector(base_engineer, 'groups', groups=['temporal', 'metadata'])


def select_features_by_pattern(base_engineer, patterns: List[str],
                              flags: int = 0) -> FeatureSelector:
    """Convenience function to select features by regex patterns"""
    return FeatureSelector(base_engineer, 'patterns', patterns=patterns, flags=flags)


def select_feature_indices(base_engineer, indices: List[int]) -> FeatureSelector:
    """Convenience function to select features by index"""
    return FeatureSelector(base_engineer, 'indices', indices=indices)
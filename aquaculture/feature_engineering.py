"""Feature engineering transformer for aquaculture remote sensing.

This module provides the AquacultureFeatureEngineer class, a scikit-learn
compatible transformer that processes multi-temporal Sentinel-1 SAR and
Sentinel-2 multispectral imagery to generate features suitable for
machine learning models like LightGBM, CatBoost, and XGBoost.

Typical usage
-------------
>>> import numpy as np
>>> from aquaculture.feature_engineering import AquacultureFeatureEngineer
>>> # Create transformer with default configuration
>>> fe = AquacultureFeatureEngineer(simulate_mask=True, random_state=42)
>>> # Fit on training data (learns nothing, but required by sklearn API)
>>> fe.fit(train_data)
>>> # Transform training data (with masking simulation)
>>> X_train = fe.transform(train_data, training=True)
>>> # Transform test data (no masking simulation)
>>> X_test = fe.transform(test_data, training=False)
>>> print(X_train.head())
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import AquacultureConfig
from . import indices, masking, temporal


class AquacultureFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generate features for aquaculture pond classification from Sentinel-1/2 time series.

    Parameters
    ----------
    simulate_mask : bool, default=True
        Whether to simulate cloud masking and window selection during training.
        If False, all months are treated as observed.
    random_state : int, np.random.Generator or None, default=None
        Random seed or generator for reproducible random operations.
    window_length_probs : tuple of float, default=(1/3, 1/3, 1/3)
        Probabilities for window lengths 4, 5, and 6 months.
    start_month_distribution : list of float or None, default=None
        Probability distribution for the start month (0-11). If None,
        a uniform distribution over valid start months is used.
    s2_monthly_dropout : list of float, default=[0.0]*12
        Monthly dropout probabilities for Sentinel-2 bands (10 bands).
    include_optical : bool, default=True
        Whether to include optical features (spectral indices + green, nir, nira, swir1, swir2).
    include_sar : bool, default=True
        Whether to include SAR features (VH, VV, VH_VV_ratio, VH_VV_diff).
    include_cross_sensor_features : bool, default=True
        Whether to include cross sensor features (ratios and products of SAR and optical indices).
    include_temporal_statistics : bool, default=True
        Whether to include temporal statistics (mean, std, min, max, amplitude, slope).
    include_metadata : bool, default=True
        Whether to include metadata features (window length, start month, etc.).
    """

    def __init__(
        self,
        simulate_mask: bool = True,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        window_length_probs: Tuple[float, float, float] = (1/3, 1/3, 1/3),
        start_month_distribution: Optional[List[float]] = None,
        s2_monthly_dropout: List[float] = None,
        include_optical: bool = True,
        include_sar: bool = True,
        include_cross_sensor_features: bool = True,
        include_temporal_statistics: bool = True,
        include_metadata: bool = True,
    ):
        self.simulate_mask = simulate_mask
        self.random_state = random_state
        self.window_length_probs = window_length_probs
        self.start_month_distribution = start_month_distribution
        self.s2_monthly_dropout = s2_monthly_dropout if s2_monthly_dropout is not None else [0.0] * 12
        self.include_optical = include_optical
        self.include_sar = include_sar
        self.include_cross_sensor_features = include_cross_sensor_features
        self.include_temporal_statistics = include_temporal_statistics
        self.include_metadata = include_metadata

        # Internal random generator
        self._rng: Optional[np.random.Generator] = None

        # Feature names will be set during fit
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None

        # We'll define the feature groups and their names here for consistency
        # Note: The order must match the band order in the data (0:VH, 1:VV, 2:blue, 3:green, 4:nir, 5:nira, 6:re1, 7:re2, 8:re3, 9:red, 10:swir1, 11:swir2)
        self._optical_feature_names = [
            "NDVI",
            "NDWI",
            "MNDWI",
            "NDMI",
            "NDRE2",
            "NDRE3",
            "green",
            "nir",
            "nira",
            "swir1",
            "swir2",
        ]

        self._sar_feature_names = [
            "VH",
            "VV",
            "VH_VV_ratio",
            "VH_VV_diff",
        ]

        self._cross_sensor_feature_names = [
            "VH_NDWI_ratio",
            "VV_NDWI_ratio",
            "VH_NDVI_ratio",
            "VV_NDVI_ratio",
            "VH_NDWI_mul",
            "VV_NDWI_mul",
            "VH_NDVI_mul",
            "VV_NDVI_mul",
        ]

        self._stat_names = ["mean", "std", "min", "max", "amplitude", "slope"]

        self._metadata_names = [
            "window_length",
            "start_month",
            "end_month",
            "n_optical_obs",
            "fraction_optical",
        ] + [f"optical_obs_{m+1:02d}" for m in range(12)]

    def _check_random_state(self) -> np.random.Generator:
        """Ensure we have a random number generator."""
        if self._rng is None:
            if isinstance(self.random_state, np.random.Generator):
                self._rng = self.random_state
            else:
                self._rng = np.random.default_rng(self.random_state)
        return self._rng

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AquacultureFeatureEngineer":
        """Fit the transformer. This method does not learn any parameters.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12).
        y : np.ndarray or None, default=None
            Ignored. Present for compatibility with sklearn API.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X = self._validate_input(X)
        # Store input feature names (though we know them)
        self.feature_names_in_ = [f"band_{i:02d}" for i in range(X.shape[2])]
        # Build feature names for output
        self._build_feature_names()
        return self

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input shape and type.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        X_validated : np.ndarray
            Validated input as float64 array.

        Raises
        ------
        ValueError
            If input shape is not (n_samples, 12, 12).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"Input must be 3-dimensional (n_samples, 12, 12), got shape {X.shape}"
            )
        if X.shape[1] != 12 or X.shape[2] != 12:
            raise ValueError(
                f"Expected 12 time steps and 12 bands, got {X.shape[1]} time steps and {X.shape[2]} bands"
            )
        return X.astype(np.float64)

    def _build_feature_names(self) -> None:
        """Build output feature names based on configuration."""
        feature_names = []

        # Optical features (spectral indices + green, nir, nira, swir1, swir2)
        # These are always included as base features when include_optical=True
        if self.include_optical:
            for month in range(1, 13):
                for fname in self._optical_feature_names:
                    feature_names.append(f"{fname}_{month:02d}")

        # SAR features (VH, VV, VH_VV_ratio, VH_VV_diff)
        # These are always included as base features when include_sar=True
        if self.include_sar:
            for month in range(1, 13):
                for fname in self._sar_feature_names:
                    feature_names.append(f"{fname}_{month:02d}")

        # Cross sensor features (only if both optical and SAR are included AND we want cross features)
        if self.include_cross_sensor_features and self.include_optical and self.include_sar:
            for month in range(1, 13):
                for fname in self._cross_sensor_feature_names:
                    feature_names.append(f"{fname}_{month:02d}")

        # Temporal statistics (computed on all available base features)
        if self.include_temporal_statistics:
            # We need to know what features we have for temporal stats
            # This includes all base features: optical, SAR, and cross (if enabled)
            temp_feature_names = []
            # Optical features
            if self.include_optical:
                temp_feature_names.extend(self._optical_feature_names)
            # SAR features
            if self.include_sar:
                temp_feature_names.extend(self._sar_feature_names)
            # Cross sensor features (only if both base types are present and we want them)
            if self.include_cross_sensor_features and self.include_optical and self.include_sar:
                temp_feature_names.extend(self._cross_sensor_feature_names)

            # Now add temporal statistics for each feature
            for fname in temp_feature_names:
                for stat in self._stat_names:
                    feature_names.append(f"{fname}_{stat}")

        # Metadata features
        if self.include_metadata:
            feature_names.extend(self._metadata_names)

        self.feature_names_out_ = feature_names

    def transform(self, X: np.ndarray, training: bool = True) -> pd.DataFrame:
        """Transform input data to feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12).
        training : bool, default=True
            Whether to simulate masking (if simulate_mask=True). If False,
            no masking simulation is performed (useful for test data).

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame of shape (n_samples, n_features) with feature names as columns.
        """
        # Validate input
        X = self._validate_input(X)
        n_samples = X.shape[0]

        # Initialize random generator
        rng = self._check_random_state()

        # We'll collect features in a list and then concatenate horizontally
        feature_arrays = []

        # Step 1: Apply masking if simulating and training
        if self.simulate_mask and training:
            # Determine window length and start month per sample
            window_lengths = np.zeros(n_samples, dtype=int)
            start_months = np.zeros(n_samples, dtype=int)
            end_months = np.zeros(n_samples, dtype=int)
            # We'll also create a mask for S2 bands per sample, per month, per band
            s2_mask = np.ones((n_samples, 12, 10), dtype=bool)  # True means keep
            for i in range(n_samples):
                wl = select_window_length(rng, self.window_length_probs)
                sm = select_start_month(rng, wl, self.start_month_distribution)
                em = sm + wl - 1
                window_lengths[i] = wl
                start_months[i] = sm
                end_months[i] = em

                # FIRST: Set ALL bands (SAR and S2) to -9999 for months OUTSIDE the window
                # Months outside window: [0, sm) and (em, 11]
                X_masked = X.copy()  # We'll move this outside the loop for efficiency
                for m in range(12):
                    if m < sm or m > em:
                        # Set ALL bands (0-11) to -9999 for months outside window
                        X_masked[i, m, :] = -9999.0

                # SECOND: For months INSIDE the window, apply cloud masking to S2 bands only
                # Create monthly dropout mask for this sample
                # Shape (12, 10) for months and bands
                monthly_dropout_arr = np.array(self.s2_monthly_dropout).reshape((1, 12, 1))
                # Generate random uniform for each month and band
                rands = rng.random((12, 10))
                # Mask where random >= dropout (keep) else False (mask)
                mask_12_10 = rands >= monthly_dropout_arr
                s2_mask[i] = mask_12_10
                # Apply mask to S2 bands (indices 2-11) ONLY for months inside window
                for m in range(sm, em + 1):
                    # Where mask is False, set S2 bands to -9999
                    X_masked[i, m, 2:12][~s2_mask[i, m]] = -9999.0
        else:
            # No masking simulation: use all months as observed
            window_lengths = np.full(n_samples, 12, dtype=int)
            start_months = np.zeros(n_samples, dtype=int)
            end_months = np.full(n_samples, 11, dtype=int)
            X_masked = X.copy()
            # No masking, so no values set to -9999
            # We'll create a dummy mask of all True for consistency
            s2_mask = np.ones((n_samples, 12, 10), dtype=bool)

        # Convert masking sentinel values (-9999) to NaN for proper handling in downstream computations
        X_masked[X_masked == -9999.0] = np.nan

        # Step 2: Extract optical features (if included)
        if self.include_optical:
            # Extract S2 bands: shape (n_samples, 12, 10)
            # Order: Blue, Green, NIR, NarrowNIR, RE1, RE2, RE3, Red, SWIR1, SWIR2
            s2_bands = X_masked[:, :, 2:12]
            # Separate the bands for easier indexing
            b = s2_bands[:, :, 0]  # Blue
            g = s2_bands[:, :, 1]  # Green
            nir = s2_bands[:, :, 2]  # NIR
            nnir = s2_bands[:, :, 3]  # NarrowNIR
            re1 = s2_bands[:, :, 4]  # RE1
            re2 = s2_bands[:, :, 5]  # RE2
            re3 = s2_bands[:, :, 6]  # RE3
            r = s2_bands[:, :, 7]   # Red
            s1 = s2_bands[:, :, 8]  # SWIR1
            s2 = s2_bands[:, :, 9]  # SWIR2

            # Compute indices (vectorized over samples and months)
            ndvi = indices.ndvi(nir, r)
            ndwi = indices.ndwi(g, nir)
            mndwi = indices.mndwi(g, s1)
            ndmi = indices.ndmi(nir, s1)
            ndre2 = indices.ndre(nir, re2)
            ndre3 = indices.ndre(nir, re3)

            # Stack to shape (n_samples, 12, 6)
            indices_stack = np.stack(
                [ndvi, ndwi, mndwi, ndmi, ndre2, ndre3], axis=2
            )
            feature_arrays.append(indices_stack)

            # Extract specific optical bands (green, nir, nira, swir1, swir2)
            # Note: We skip blue, red, re1, re2, re3 as requested
            optical_bands = np.stack([g, nir, nnir, s1, s2], axis=2)  # (n_samples, 12, 5)
            feature_arrays.append(optical_bands)

        # Step 3: Extract SAR features (VH, VV, ratio, difference) (if included)
        if self.include_sar:
            vh = X_masked[:, :, 0]  # shape (n_samples, 12)
            vv = X_masked[:, :, 1]  # shape (n_samples, 12)
            # Avoid division by zero: where vv == 0, set ratio to NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                vh_vv_ratio = np.true_divide(vh, vv)
                vh_vv_ratio[vv == 0] = np.nan
            vh_vv_diff = vh - vv
            # Stack to shape (n_samples, 12, 4)
            sar_stack = np.stack([vh, vv, vh_vv_ratio, vh_vv_diff], axis=2)
            feature_arrays.append(sar_stack)

        # Step 4: Compute cross sensor features (if both optical and SAR are included)
        if self.include_cross_sensor_features and self.include_optical and self.include_sar:
            # We need VH, VV, NDWI, NDVI
            # We already have vh, vv from SAR processing above
            # And we have ndwi, ndvi from optical processing above
            # Avoid division by zero: where denominator is 0, set to NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                vh_ndwi_ratio = np.true_divide(vh, ndwi)
                vv_ndwi_ratio = np.true_divide(vv, ndwi)
                vh_ndvi_ratio = np.true_divide(vh, ndvi)
                vv_ndvi_ratio = np.true_divide(vv, ndvi)
                # Set NaN where denominator is zero or denominator is NaN
                # The above already handles division by zero, but if denominator is NaN, result is NaN.
            vh_ndwi_mul = vh * ndwi
            vv_ndwi_mul = vv * ndwi
            vh_ndvi_mul = vh * ndvi
            vv_ndvi_mul = vv * ndvi
            # Stack to shape (n_samples, 12, 8)
            cross_stack = np.stack(
                [
                    vh_ndwi_ratio,
                    vv_ndwi_ratio,
                    vh_ndvi_ratio,
                    vv_ndvi_ratio,
                    vh_ndwi_mul,
                    vv_ndwi_mul,
                    vh_ndvi_mul,
                    vv_ndvi_mul,
                ],
                axis=2,
            )
            feature_arrays.append(cross_stack)

        # Step 5: Combine all monthly features
        if feature_arrays:
            monthly_features = np.concatenate(feature_arrays, axis=2)  # (n_samples, 12, n_monthly)
        else:
            monthly_features = np.empty((n_samples, 12, 0))

        # Step 6: Compute temporal statistics (if included)
        if self.include_temporal_statistics and monthly_features.shape[2] > 0:
            # Build ignore_nans mask for each monthly feature
            ignore_nans = []
            # Optical features (if included)
            if self.include_optical:
                # Spectral indices (6): all True (ignore NaNs)
                # Specific optical bands (green, nir, nira, swir1, swir2) (5): all True (ignore NaNs)
                ignore_nans.extend([True] * 11)  # 6 + 5 = 11 optical features
            # SAR features (if included)
            if self.include_sar:
                # VH, VV, VH_VV_ratio, VH_VV_diff (4): all False (don't ignore NaNs)
                ignore_nans.extend([False] * 4)
            # Cross sensor features (if included)
            if self.include_cross_sensor_features and self.include_optical and self.include_sar:
                # 8 cross features: all True (ignore NaNs)
                ignore_nans.extend([True] * 8)

            ignore_nans = np.array(ignore_nans, dtype=bool)

            # Determine which features are SAR vs optical for statistics calculation
            optical_indices = np.where(ignore_nans)[0]
            sar_indices = np.where(~ignore_nans)[0]

            # Compute temporal statistics for all features at once
            temporal_features = temporal.compute_temporal_stats(
                monthly_features,
                sar_indices=sar_indices,
                optical_indices=optical_indices
            )   # shape: (n_samples, n_monthly * 6)

            # Collapse monthly features to 2D for concatenation
            monthly_features_2d = monthly_features.reshape(n_samples, -1)

            # Build feature_arrays list
            feature_arrays = [monthly_features_2d, temporal_features]
        else:
            # If not computing temporal statistics, we still need to use the monthly features
            # Reshape to 2D for concatenation with other features
            if monthly_features.size > 0:
                monthly_features = monthly_features.reshape(n_samples, -1)
                feature_arrays = [monthly_features]
            else:
                feature_arrays = []

        # Step 7: Compute metadata features
        if self.include_metadata:
            # We have per-sample window_lengths, start_months, end_months
            # Compute n_optical_obs and fraction_optical
            # We'll use the s2_mask: True means the band is kept (not masked)
            # For each sample and month, we have a mask of length 10 (S2 bands).
            # We consider a month to have an optical observation if any of the 10 bands is kept.
            optical_obs_per_month = np.any(s2_mask, axis=2)  # shape (n_samples, 12)
            n_optical_obs = np.sum(optical_obs_per_month, axis=1)  # shape (n_samples,)
            fraction_optical = n_optical_obs / 12.0

            # Stack metadata features
            # Shape (n_samples, 5 + 12) = (n_samples, 17)
            meta_features = np.column_stack(
                (
                    window_lengths,
                    start_months,
                    end_months,
                    n_optical_obs,
                    fraction_optical,
                    # Now the 12 binary flags
                    optical_obs_per_month.astype(int),
                )
            )
            feature_arrays.append(meta_features)

        # Now combine all feature arrays horizontally
        if feature_arrays:
            X_combined = np.concatenate(feature_arrays, axis=1)
        else:
            X_combined = np.empty((n_samples, 0))

        # Create DataFrame
        if X_combined.size > 0:
            # Use the feature names we built in _build_feature_names
            # But we need to make sure they match the actual columns
            # For safety, we'll generate names based on what we actually computed
            df = pd.DataFrame(X_combined)
        else:
            df = pd.DataFrame(index=range(n_samples))

        # Set the column names to our built feature names
        # But only if we have the right number of columns
        if len(self.feature_names_out_) == df.shape[1]:
            df.columns = self.feature_names_out_
        else:
            # If there's a mismatch, we'll generate generic names
            # This should not happen in a correct implementation
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Feature names.
        """
        if self.feature_names_out_ is None:
            self._build_feature_names()
        return np.array(self.feature_names_out_)

    def summary(self) -> dict:
        """Return a summary of the transformer's configuration.

        Returns
        -------
        summary : dict
            Summary of parameters and feature counts.
        """
        return {
            "simulate_mask": self.simulate_mask,
            "random_state": self.random_state,
            "n_features_out": len(self.get_feature_names_out())
                if self.feature_names_out_ is not None
                else None,
        }

    def feature_groups(self) -> dict:
        """Return a dictionary mapping feature group names to lists of feature indices.

        Returns
        -------
        groups : dict
            Dictionary with keys 'sar', 'optical', 'cross', 'temporal', 'metadata', 'other'
            mapping to lists of column indices in the transformed data.
        """
        if self.feature_names_out_ is None:
            self._build_feature_names()

        feature_names = self.feature_names_out_
        groups = {
            'sar': [],
            'optical': [],
            'cross': [],
            'temporal': [],
            'metadata': [],
            'other': []
        }

        # Precompute sets for faster lookup
        sar_set = {"VH", "VV", "VH_VV_ratio", "VH_VV_diff"}
        optical_set = {"NDVI", "NDWI", "MNDWI", "NDMI", "NDRE2", "NDRE3",
                      "green", "nir", "nira", "swir1", "swir2"}
        cross_set = {"VH_NDWI_ratio", "VV_NDWI_ratio", "VH_NDVI_ratio", "VV_NDVI_ratio",
                    "VH_NDWI_mul", "VV_NDWI_mul", "VH_NDVI_mul", "VV_NDVI_mul"}
        stat_set = set(self._stat_names)
        metadata_set = set(self._metadata_names)

        for i, name in enumerate(feature_names):
            # Metadata features (exact match)
            if name in metadata_set:
                groups['metadata'].append(i)
            # Temporal statistics: ends with _{stat}
            elif any(name.endswith(f"_{stat}") for stat in stat_set):
                groups['temporal'].append(i)
            # Monthly features: ends with _{MM} where MM is 01-12
            elif len(name) >= 5 and name[-3] == '_' and name[-2:].isdigit() and 1 <= int(name[-2:]) <= 12:
                prefix = name[:-3]  # Remove the _{MM} suffix
                month = int(name[-2:])
                if prefix in sar_set:
                    groups['sar'].append(i)
                elif prefix in optical_set:
                    groups['optical'].append(i)
                elif prefix in cross_set:
                    groups['cross'].append(i)
                else:
                    groups['other'].append(i)
            # If we get here, it's an unknown feature type (shouldn't happen with current implementation)
            else:
                groups['other'].append(i)

        return groups

    def plot_missingness(self) -> None:
        """Plot missingness pattern from the last transformation.

        This method would require storing the mask from transform.
        For now, it raises NotImplementedError.
        """
        raise NotImplementedError("plot_missingness not implemented")

    def validate_masking(self) -> dict:
        """Validate the masking simulation by comparing simulated vs expected.

        Returns
        -------
        validation : dict
            Dictionary with validation metrics.
        """
        # TODO: implement
        return {}


# Helper functions to avoid circular imports
def select_window_length(
    rng: np.random.Generator, window_length_probs: Tuple[float, float, float]
) -> int:
    """Select a window length based on probabilities."""
    _validate_probabilities(np.array(window_length_probs), "window_length_probs")
    return int(rng.choice([4, 5, 6], p=window_length_probs))


def select_start_month(
    rng: np.random.Generator,
    window_length: int,
    start_month_distribution: Optional[List[float]] = None,
) -> int:
    """Select a start month for the observation window."""
    max_start = 12 - window_length
    if start_month_distribution is None:
        probs = np.ones(max_start + 1) / (max_start + 1)
    else:
        if len(start_month_distribution) != 12:
            raise ValueError("start_month_distribution must have length 12")
        if not np.isclose(sum(start_month_distribution), 1.0):
            raise ValueError("start_month_distribution must sum to 1.0")
        probs = np.array(start_month_distribution[: max_start + 1])
        probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(max_start + 1), p=probs))


def _validate_probabilities(probs: np.ndarray, name: str) -> None:
    """Validate that an array of probabilities sums to 1 and is non-negative."""
    if not np.allclose(np.sum(probs), 1.0):
        raise ValueError(f"{name} must sum to 1.0, got {np.sum(probs)}")
    if np.any(probs < 0):
        raise ValueError(f"{name} must contain non-negative values")
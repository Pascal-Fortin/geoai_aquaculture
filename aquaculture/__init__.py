"""Aquaculture Feature Engineering Package.

This package provides a scikit-learn compatible transformer for feature engineering
on aquaculture remote sensing data, specifically designed for Sentinel-1 SAR and
Sentinel-2 multispectral imagery time series.

Typical usage
-------------
>>> from aquaculture.feature_engineering import AquacultureFeatureEngineer
>>> fe = AquacultureFeatureEngineer(simulate_mask=True, random_state=42)
>>> fe.fit(train_data)
>>> X_train = fe.transform(train_data, training=True)
>>> X_test = fe.transform(test_data, training=False)
"""

from .config import AquacultureConfig
from .feature_engineering import AquacultureFeatureEngineer

__all__ = [
    "AquacultureConfig",
    "AquacultureFeatureEngineer",
]
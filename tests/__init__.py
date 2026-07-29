"""
Test package for aquaculture machine learning framework.
"""

# Import test modules to make them available when importing the test package
from . import test_basic
from . import test_logging
from . import test_trainer
from . import test_optuna_utils
from . import test_trainer_init

__all__ = [
    'test_basic',
    'test_logging',
    'test_trainer',
    'test_optuna_utils',
    'test_trainer_init'
]
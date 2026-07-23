#!/usr/bin/env python3
"""
Master test script to run all tests for the aquaculture machine learning framework.
"""

import unittest
import sys
import os
from pathlib import Path

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent
    # Add the src directory to the path so tests can import modules
    sys.path.insert(0, str(project_root / "src"))

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = str(project_root / "tests")
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
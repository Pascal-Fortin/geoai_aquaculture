"""
Tests for the logging functionality.
"""

import tempfile
import sys
import os
import logging
from pathlib import Path
import unittest

# Add the project root to the Python path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.trainer import Trainer


class TestLogging(unittest.TestCase):
    """Test logging configuration and functionality."""

    def setUp(self):
        """Reset logging state before each test."""
        # Shutdown any existing handlers
        logging.shutdown()
        # Clear handlers from root logger
        root = logging.getLogger()
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        # Reset root logger level to default (WARNING)
        root.setLevel(logging.WARNING)

    def tearDown(self):
        """Clean up after each test."""
        self.setUp()

    def test_logging_enabled(self):
        """Test that logging works when enabled."""
        # Create a temporary directory for our test
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiments"

            # Create a config with file logging enabled
            config = TrainingConfig(
                experiment_dir=str(exp_dir),
                enable_file_logging=True,
                log_level_file="INFO"
            )

            # Create a trainer instance
            trainer = Trainer(config)

            # Trigger the experiment directory creation (which should set up logging)
            exp_dir_path = trainer._create_experiment_directory()
            self.assertTrue(exp_dir_path.exists(), "Experiment directory should be created")

            # Check if logs directory exists
            log_dir = exp_dir_path / "logs"
            self.assertTrue(log_dir.exists(), "Logs directory should be created")

            # Check if training.log exists
            log_file = log_dir / "training.log"
            self.assertTrue(log_file.exists(), "Log file should be created when logging is enabled")

            # Read and check the log file contents
            with open(log_file, 'r') as f:
                content = f.read()

            # Check if our expected log messages are present
            self.assertIn("File logging enabled:", content, "File logging setup message should be in log")
            self.assertIn("Created experiment directory:", content, "Experiment directory creation message should be in log")

    def test_logging_disabled(self):
        """Test that no log file is created when logging is disabled."""
        # Create a temporary directory for our test
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiments"

            # Create a config with file logging DISABLED
            config = TrainingConfig(
                experiment_dir=str(exp_dir),
                enable_file_logging=False,  # DISABLED
                log_level_file="INFO"
            )

            # Create a trainer instance
            trainer = Trainer(config)

            # Trigger the experiment directory creation
            exp_dir_path = trainer._create_experiment_directory()
            self.assertTrue(exp_dir_path.exists(), "Experiment directory should be created")

            # Check if logs directory exists
            log_dir = exp_dir_path / "logs"
            self.assertTrue(log_dir.exists(), "Logs directory should be created")

            # Check if training.log exists (it should NOT)
            log_file = log_dir / "training.log"
            self.assertFalse(log_file.exists(), "No log file should be created when logging is disabled")

    def test_log_level_config(self):
        """Test that log level configuration works."""
        # Create a temporary directory for our test
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiments"

            # Create a config with file logging enabled and DEBUG level
            config = TrainingConfig(
                experiment_dir=str(exp_dir),
                enable_file_logging=True,
                log_level_file="DEBUG"  # Set to DEBUG
            )

            # Create a trainer instance
            trainer = Trainer(config)

            # Trigger the experiment directory creation
            exp_dir_path = trainer._create_experiment_directory()
            self.assertTrue(exp_dir_path.exists(), "Experiment directory should be created")

            # Check if logs directory exists
            log_dir = exp_dir_path / "logs"
            self.assertTrue(log_dir.exists(), "Logs directory should be created")

            # Check if training.log exists
            log_file = log_dir / "training.log"
            self.assertTrue(log_file.exists(), "Log file should be created")

            # Read and check the log file contents
            with open(log_file, 'r') as f:
                content = f.read()

            # At minimum, we should see our setup messages (which are INFO level)
            self.assertIn("File logging enabled:", content, "File logging setup message should be in log")
            self.assertIn("Created experiment directory:", content, "Experiment directory creation message should be in log")
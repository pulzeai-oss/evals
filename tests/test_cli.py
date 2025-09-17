"""
Basic tests for the evaluation CLI.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_cli import EvalCLI, create_parser
from evaluators import get_evaluator
from utils import ConfigLoader


class TestConfigLoader:
    """Test the configuration loader."""

    def test_config_loader_initialization(self):
        """Test that ConfigLoader initializes without errors."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            assert config is not None
            assert "OPENAI_API_KEY" in config

    def test_config_validation(self):
        """Test configuration validation."""
        with patch.dict(os.environ, {}, clear=True):
            config_loader = ConfigLoader()
            errors = config_loader.validate_config()
            assert "api_keys" in errors


class TestEvaluatorFactory:
    """Test the evaluator factory."""

    def test_get_evaluator_valid_benchmarks(self):
        """Test that we can get evaluators for all valid benchmarks."""
        config = {"OPENAI_API_KEY": "test_key"}

        valid_benchmarks = ["financebench", "mmlu", "pulze", "marketing"]

        for benchmark in valid_benchmarks:
            evaluator = get_evaluator(benchmark, config)
            assert evaluator is not None
            assert evaluator.benchmark_name == benchmark

    def test_get_evaluator_invalid_benchmark(self):
        """Test that invalid benchmark raises ValueError."""
        config = {"OPENAI_API_KEY": "test_key"}

        with pytest.raises(ValueError, match="Unsupported benchmark"):
            get_evaluator("invalid_benchmark", config)


class TestCLIParser:
    """Test the CLI argument parser."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None

    def test_run_command_parsing(self):
        """Test parsing of run command."""
        parser = create_parser()

        # Use a benchmark that exists in the current directory structure
        # If no benchmarks are discovered, we'll use a mock approach
        from eval_cli import _discover_benchmarks_from_folders_static

        available_benchmarks = _discover_benchmarks_from_folders_static()

        if available_benchmarks:
            benchmark = available_benchmarks[0]
            args = parser.parse_args(["run", "--benchmark", benchmark, "--model", "gpt-4", "--subject", "marketing"])

            assert args.command == "run"
            assert args.benchmark == benchmark
            assert args.model == "gpt-4"
            assert args.subject == "marketing"
        else:
            # If no benchmarks are available, test that parser creation doesn't crash
            assert parser is not None

    def test_leaderboard_command_parsing(self):
        """Test parsing of leaderboard command."""
        parser = create_parser()

        # Use a benchmark that exists in the current directory structure
        from eval_cli import _discover_benchmarks_from_folders_static

        available_benchmarks = _discover_benchmarks_from_folders_static()

        if available_benchmarks:
            benchmark = available_benchmarks[0]
            args = parser.parse_args(["leaderboard", "--benchmark", benchmark, "--export", "html"])

            assert args.command == "leaderboard"
            assert args.benchmark == benchmark
            assert args.export == "html"
        else:
            # Test leaderboard with --all flag which doesn't require specific benchmark
            args = parser.parse_args(["leaderboard", "--all"])
            assert args.command == "leaderboard"
            assert args.all is True

    def test_list_command_parsing(self):
        """Test parsing of list command."""
        parser = create_parser()

        args = parser.parse_args(["list"])

        assert args.command == "list"


class TestEvalCLI:
    """Test the main CLI class."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_cli_initialization(self):
        """Test that CLI initializes with valid config."""
        cli = EvalCLI(validate_config=False)  # Skip validation for testing
        assert cli.config_loader is not None
        assert cli.results_manager is not None
        assert cli.leaderboard_generator is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_initialization_invalid_config(self):
        """Test that CLI exits with invalid config."""
        with pytest.raises(SystemExit):
            EvalCLI(validate_config=True)  # Enable validation to test SystemExit


if __name__ == "__main__":
    pytest.main([__file__])

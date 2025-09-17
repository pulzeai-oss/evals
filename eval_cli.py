#!/usr/bin/env python3
"""
Unified CLI for running evaluations across multiple benchmarks.

A comprehensive command-line interface for running evaluations across multiple benchmarks
(FinanceBench, MMLU, Pulze-v0.1, Marketing) and viewing results in leaderboard format.

Usage:
    python eval_cli.py run --benchmark financebench --model pulze/llama-3.1-70b-instruct --rater gpt-4
    python eval_cli.py run --benchmark mmlu --subjects marketing --model openai/gpt-4
    python eval_cli.py leaderboard --benchmark marketing
    python eval_cli.py list
"""

import argparse
import json
import os
import sys

# Import our modular components
from evaluators import get_evaluator
from utils import ConfigLoader, LeaderboardGenerator, ResultsManager


class EvalCLI:
    """Main CLI class for unified evaluation system."""

    def __init__(self, validate_config=True):
        """Initialize the CLI with configuration and managers."""
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_config()

        # Validate configuration if requested (default True for production, False for testing)
        if validate_config:
            errors = self.config_loader.validate_config()
            if errors:
                print("Configuration errors found:")
                for key, error in errors.items():
                    print(f"  {key}: {error}")
                print("\nPlease check your environment variables.")
                sys.exit(1)

        self.results_manager = ResultsManager(self.config.get("RESULTS_DIR", "results"))
        self.leaderboard_generator = LeaderboardGenerator(self.results_manager)

    def _validate_config_for_run(self):
        """Validate configuration for run command (requires API keys)."""
        errors = self.config_loader.validate_config()
        if errors:
            print("Configuration errors found:")
            for key, error in errors.items():
                print(f"  {key}: {error}")
            print("\nPlease check your environment variables.")
            sys.exit(1)

    def run_evaluation(self, args):
        """Run evaluation on specified benchmark."""
        # Validate configuration (requires API keys for run command)
        self._validate_config_for_run()

        try:
            # Get evaluator for the benchmark
            evaluator = get_evaluator(args.benchmark, self.config)

            print(f"Running evaluation on {args.benchmark}")
            if args.subject:
                print(f"Subject: {args.subject}")

            # Run the evaluation
            results = evaluator.evaluate(
                model=args.model,
                template=args.template or self.config.get("DEFAULT_TEMPLATE", "default"),
                rater_model=args.rater or self.config.get("DEFAULT_RATER_MODEL", "gpt-4"),
                subject=args.subject,
            )

            # Save results
            output_file = self.results_manager.save_results(
                results=results,
                benchmark=args.benchmark,
                model=args.model,
                template=args.template or self.config.get("DEFAULT_TEMPLATE", "default"),
                rater_model=args.rater or self.config.get("DEFAULT_RATER_MODEL", "gpt-4"),
                subject=args.subject,
            )

            # Print summary
            total_items = len(results)
            successful_items = len([r for r in results if "error" not in r])
            avg_score = sum(r.get("score", 0.0) for r in results) / total_items if total_items > 0 else 0.0

            print("\nEvaluation Summary:")
            print(f"  Total items: {total_items}")
            print(f"  Successful: {successful_items}")
            print(f"  Average score: {avg_score:.4f}")
            print(f"  Results saved to: {output_file}")

        except Exception as e:
            print(f"Error running evaluation: {e}")
            sys.exit(1)

    def show_leaderboard(self, args):
        """Display leaderboard with evaluation results."""
        try:
            if args.benchmark:
                # Single benchmark leaderboard
                self.leaderboard_generator.print_benchmark_leaderboard(
                    benchmark=args.benchmark, sort_by=args.sort_by, top_n=args.top_n
                )

                # Export if requested
                if args.export:
                    if args.export == "html":
                        output_file = f"{args.benchmark}_leaderboard.html"
                        self.leaderboard_generator.export_leaderboard_html(
                            benchmark=args.benchmark, output_file=output_file, sort_by=args.sort_by
                        )
                    elif args.export == "csv":
                        output_file = f"{args.benchmark}_results.csv"
                        self.results_manager.export_results_csv(args.benchmark, output_file)

            elif args.all:
                # Cross-benchmark leaderboard - use discovered benchmarks
                discovered_benchmarks = self._discover_benchmarks_from_folders()
                benchmark_names = list(discovered_benchmarks.keys())
                self.leaderboard_generator.print_cross_benchmark_leaderboard(
                    benchmarks=benchmark_names, top_n=args.top_n
                )
            else:
                print("Please specify --benchmark or --all")

        except Exception as e:
            print(f"Error generating leaderboard: {e}")
            sys.exit(1)

    def _discover_benchmarks_from_folders(self):
        """Discover benchmarks from folder structure and benchmark.json files."""
        benchmarks = {}

        # Look for benchmark folders in current directory
        current_dir = os.getcwd()
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)

            # Skip if not a directory or if it's a system/hidden directory
            if (
                not os.path.isdir(item_path)
                or item.startswith(".")
                or item in ["utils", "evaluators", "tests", "results"]
            ):
                continue

            # Check if it has benchmark.json
            benchmark_json_path = os.path.join(item_path, "benchmark.json")
            if os.path.exists(benchmark_json_path):
                try:
                    with open(benchmark_json_path, "r") as f:
                        benchmark_data = json.load(f)

                    benchmark_id = benchmark_data.get("id", item)
                    benchmark_name = benchmark_data.get("name", item.title())
                    benchmark_description = benchmark_data.get("description", f"{benchmark_name} benchmark")

                    benchmarks[benchmark_id] = benchmark_description
                except (json.JSONDecodeError, IOError):
                    # If benchmark.json is invalid, fall back to folder name
                    benchmarks[item] = f"{item.title()} benchmark (auto-discovered)"
            else:
                # Check if it looks like a benchmark folder (has data/ or results/ subdirectory)
                has_data = os.path.exists(os.path.join(item_path, "data"))
                has_results = os.path.exists(os.path.join(item_path, "results"))

                if has_data or has_results:
                    benchmarks[item] = f"{item.title()} benchmark (auto-discovered)"

        return benchmarks

    def _get_available_benchmarks(self):
        """Get all available benchmarks (both with evaluators and with results)."""
        # Discover benchmarks from folders
        discovered_benchmarks = self._discover_benchmarks_from_folders()

        # Also check for benchmarks with results in the results directory
        all_results = self.results_manager.get_all_results()
        results_benchmarks = set(all_results.keys())

        # Combine discovered and results benchmarks
        all_benchmarks = discovered_benchmarks.copy()
        for benchmark in results_benchmarks:
            if benchmark not in all_benchmarks:
                all_benchmarks[benchmark] = "Auto-discovered benchmark with results"

        return all_benchmarks

    def list_benchmarks(self):
        """List available benchmarks and their subjects."""
        benchmarks = self._get_available_benchmarks()

        print("Available Benchmarks:")
        print("=" * 60)

        for benchmark, description in benchmarks.items():
            try:
                evaluator = get_evaluator(benchmark, self.config)
                subjects = evaluator.get_available_subjects()

                print(f"\n{benchmark.upper()}")
                print(f"  Description: {description}")
                print(f"  Subjects ({len(subjects)}):")

                # Show first 10 subjects
                for subject in subjects[:10]:
                    print(f"    - {subject}")
                if len(subjects) > 10:
                    print(f"    ... and {len(subjects) - 10} more")

            except Exception as e:
                print(f"\n{benchmark.upper()}")
                print(f"  Description: {description}")

                # Check if we have results even if no evaluator
                all_results = self.results_manager.get_all_results(benchmark)
                if benchmark in all_results and all_results[benchmark]:
                    print(f"  Results available: {len(all_results[benchmark])} files")
                    print("  Note: Evaluator not available, but results can be viewed in leaderboard")
                else:
                    print(f"  Error loading subjects: {e}")

    def show_config(self):
        """Show current configuration."""
        self.config_loader.print_config_summary()


def _discover_benchmarks_from_folders_static():
    """Static version of benchmark discovery for parser creation."""
    benchmarks = {}

    try:
        # Look for benchmark folders in current directory
        current_dir = os.getcwd()
        if not os.path.exists(current_dir):
            return []

        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)

            # Skip if not a directory or if it's a system/hidden directory
            if (
                not os.path.isdir(item_path)
                or item.startswith(".")
                or item in ["utils", "evaluators", "tests", "results"]
            ):
                continue

            # Check if it has benchmark.json
            benchmark_json_path = os.path.join(item_path, "benchmark.json")
            if os.path.exists(benchmark_json_path):
                try:
                    with open(benchmark_json_path, "r") as f:
                        benchmark_data = json.load(f)

                    benchmark_id = benchmark_data.get("id", item)
                    benchmarks[benchmark_id] = True
                except (json.JSONDecodeError, IOError):
                    # If benchmark.json is invalid, fall back to folder name
                    benchmarks[item] = True
            else:
                # Check if it looks like a benchmark folder (has data/ or results/ subdirectory)
                has_data = os.path.exists(os.path.join(item_path, "data"))
                has_results = os.path.exists(os.path.join(item_path, "results"))

                if has_data or has_results:
                    benchmarks[item] = True

        return list(benchmarks.keys())
    except (OSError, PermissionError):
        # If we can't read the directory, return empty list
        return []


def create_parser():
    """Create argument parser for CLI."""
    # Discover available benchmarks for choices
    available_benchmarks = _discover_benchmarks_from_folders_static()

    parser = argparse.ArgumentParser(
        description="Unified CLI for running evaluations across multiple benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FinanceBench evaluation
  python eval_cli.py run --benchmark financebench --model pulze/llama-3.1-70b-instruct --rater gpt-4

  # Run MMLU marketing evaluation
  python eval_cli.py run --benchmark mmlu --subject marketing --model openai/gpt-4

  # Run Pulze evaluation with specific template
  python eval_cli.py run --benchmark pulze-v0.1 --subject writing_marketing_materials --model pulze/llama-3.1-70b-instruct --template pulze_multi_dimensional_evaluation

  # Show marketing benchmark leaderboard
  python eval_cli.py leaderboard --benchmark marketing

  # Show cross-benchmark leaderboard
  python eval_cli.py leaderboard --all

  # Export leaderboard to HTML
  python eval_cli.py leaderboard --benchmark mmlu --export html

  # List available benchmarks
  python eval_cli.py list

  # Show configuration
  python eval_cli.py config
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument(
        "--benchmark",
        required=True,
        choices=available_benchmarks,
        help="Benchmark to evaluate",
    )
    run_parser.add_argument("--subject", help="Specific subject to evaluate")
    run_parser.add_argument("--model", required=True, help="Model to evaluate")
    run_parser.add_argument("--rater", help="Rater model (default from config)")
    run_parser.add_argument("--template", help="Evaluation template to use")

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    leaderboard_parser.add_argument(
        "--benchmark",
        choices=available_benchmarks,
        help="Show leaderboard for specific benchmark",
    )
    leaderboard_parser.add_argument("--all", action="store_true", help="Show cross-benchmark leaderboard")
    leaderboard_parser.add_argument(
        "--sort-by", choices=["average", "total", "count"], default="average", help="Sort criteria"
    )
    leaderboard_parser.add_argument("--top-n", type=int, help="Show only top N models")
    leaderboard_parser.add_argument("--export", choices=["html", "csv"], help="Export leaderboard to file")

    # List command
    subparsers.add_parser("list", help="List available benchmarks and subjects")

    # Config command
    subparsers.add_parser("config", help="Show current configuration")

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = EvalCLI()

    if args.command == "run":
        cli.run_evaluation(args)
    elif args.command == "leaderboard":
        cli.show_leaderboard(args)
    elif args.command == "list":
        cli.list_benchmarks()
    elif args.command == "config":
        cli.show_config()


if __name__ == "__main__":
    main()

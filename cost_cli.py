#!/usr/bin/env python3
"""
Cost Estimation CLI for Benchmark Evaluations
Estimates token usage and costs for running benchmarks with different AI models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import tabulate
from utils.cost_estimator import CostEstimator


def format_cost(cost: float) -> str:
    """Format cost for display"""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    """Format token count for display"""
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.2f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.1f}K"
    else:
        return str(tokens)


def print_cost_summary(result: dict, detailed: bool = False):
    """Print cost estimation summary"""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== Cost Estimation Summary ===")
    print(f"Benchmark: {result['benchmark']}")
    print(f"Model: {result['model']}")
    print(f"Tier: {result['tier']}")
    if result.get("subject"):
        print(f"Subject: {result['subject']}")

    print(f"\nTotal Items: {result['total_items']:,}")
    print(f"Total Tokens: {format_tokens(result['total_tokens'])}")
    print(f"Total Cost: {format_cost(result['total_cost'])}")
    print(
        f"Average Cost per Item: {format_cost(result['total_cost'] / result['total_items'] if result['total_items'] > 0 else 0)}"
    )

    if detailed and result.get("files"):
        print("\n=== File Breakdown ===")
        headers = ["File", "Items", "Tokens", "Cost", "Cost/Item"]
        rows = []

        for file_info in result["files"]:
            filename = Path(file_info["file_path"]).name
            cost_per_item = file_info["total_cost"] / file_info["total_items"] if file_info["total_items"] > 0 else 0
            rows.append(
                [
                    filename,
                    f"{file_info['total_items']:,}",
                    format_tokens(file_info["total_tokens"]),
                    format_cost(file_info["total_cost"]),
                    format_cost(cost_per_item),
                ]
            )

        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))


def print_model_comparison(result: dict):
    """Print model comparison table"""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== Model Cost Comparison ===")
    print(f"Benchmark: {result['benchmark']}")
    print(f"Tier: {result['tier']}")
    if result.get("subject"):
        print(f"Subject: {result['subject']}")

    headers = ["Rank", "Model", "Total Cost", "Tokens", "Items", "Cost/Item"]
    rows = []

    for i, model_info in enumerate(result["models"], 1):
        rows.append(
            [
                i,
                model_info["model"],
                format_cost(model_info["total_cost"]),
                format_tokens(model_info["total_tokens"]),
                f"{model_info['total_items']:,}",
                format_cost(model_info["cost_per_item"]),
            ]
        )

    print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))


def list_available_models(estimator: CostEstimator, tier: Optional[str] = None):
    """List available models"""
    if tier:
        models = estimator.get_available_models(tier)
        print(f"\n=== Available Models ({tier} tier) ===")
    else:
        models = estimator.get_available_models()
        print("\n=== All Available Models ===")

    # Group by model family
    model_families = {}
    for model in models:
        if model.startswith("gpt-"):
            family = "OpenAI GPT"
        elif model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
            family = "OpenAI O-Series"
        elif model.startswith("claude-"):
            family = "Anthropic Claude"
        elif model.startswith("davinci-") or model.startswith("babbage-"):
            family = "OpenAI Legacy"
        else:
            family = "Other"

        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model)

    for family, family_models in sorted(model_families.items()):
        print(f"\n{family}:")
        for model in sorted(family_models):
            print(f"  - {model}")


def list_benchmarks():
    """List available benchmarks and subjects"""
    print("\n=== Available Benchmarks ===")

    benchmarks = {
        "financebench": {"description": "Financial document analysis", "subjects": ["financial_analysis"]},
        "mmlu": {
            "description": "Multiple-choice questions across 57 subjects",
            "subjects": ["marketing", "business_ethics", "management", "economics", "etc."],
        },
        "pulze": {
            "description": "Template-based evaluations across 56 subjects",
            "subjects": ["writing_marketing_materials", "creative_writing", "data_analysis", "etc."],
        },
        "marketing": {
            "description": "Marketing-specific evaluations",
            "subjects": ["writing_marketing_materials", "ulta_beauty", "marketing"],
        },
    }

    for benchmark, info in benchmarks.items():
        print(f"\n{benchmark}:")
        print(f"  Description: {info['description']}")
        print(f"  Subjects: {', '.join(info['subjects'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate costs for benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate cost for entire FinanceBench with GPT-4
  python cost_cli.py estimate --benchmark financebench --model gpt-4o

  # Estimate cost for MMLU marketing with multiple models
  python cost_cli.py estimate --benchmark mmlu --subject marketing --model gpt-4o-mini --model claude-3-haiku

  # Compare costs across models for a benchmark
  python cost_cli.py compare --benchmark marketing --models gpt-4o-mini,claude-3-haiku,gpt-3.5-turbo

  # Estimate with batch pricing
  python cost_cli.py estimate --benchmark pulze --subject writing_marketing_materials --model gpt-4o --tier batch

  # List available models
  python cost_cli.py list-models

  # List available benchmarks
  python cost_cli.py list-benchmarks
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate cost for a benchmark")
    estimate_parser.add_argument(
        "--benchmark",
        required=True,
        choices=["financebench", "mmlu", "pulze", "marketing"],
        help="Benchmark to estimate",
    )
    estimate_parser.add_argument(
        "--model", required=True, action="append", help="Model to estimate (can specify multiple)"
    )
    estimate_parser.add_argument("--subject", help="Specific subject (for multi-subject benchmarks)")
    estimate_parser.add_argument(
        "--tier",
        default="standard",
        choices=["standard", "batch", "flex", "priority"],
        help="Pricing tier (default: standard)",
    )
    estimate_parser.add_argument("--cached-input", action="store_true", help="Use cached input pricing where available")
    estimate_parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown by file")
    estimate_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare costs across models")
    compare_parser.add_argument(
        "--benchmark",
        required=True,
        choices=["financebench", "mmlu", "pulze", "marketing"],
        help="Benchmark to compare",
    )
    compare_parser.add_argument("--models", required=True, help="Comma-separated list of models to compare")
    compare_parser.add_argument("--subject", help="Specific subject (for multi-subject benchmarks)")
    compare_parser.add_argument(
        "--tier",
        default="standard",
        choices=["standard", "batch", "flex", "priority"],
        help="Pricing tier (default: standard)",
    )
    compare_parser.add_argument("--cached-input", action="store_true", help="Use cached input pricing where available")
    compare_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    list_models_parser.add_argument(
        "--tier", choices=["standard", "batch", "flex", "priority"], help="Filter by pricing tier"
    )

    # List benchmarks command
    subparsers.add_parser("list-benchmarks", help="List available benchmarks")

    # File command - estimate cost for specific file
    file_parser = subparsers.add_parser("file", help="Estimate cost for a specific file")
    file_parser.add_argument("--file", required=True, help="Path to JSONL file")
    file_parser.add_argument("--model", required=True, help="Model to estimate")
    file_parser.add_argument(
        "--tier",
        default="standard",
        choices=["standard", "batch", "flex", "priority"],
        help="Pricing tier (default: standard)",
    )
    file_parser.add_argument("--cached-input", action="store_true", help="Use cached input pricing where available")
    file_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    estimator = CostEstimator()

    try:
        if args.command == "estimate":
            if len(args.model) == 1:
                # Single model estimation
                result = estimator.estimate_cost_for_benchmark(
                    args.benchmark, args.model[0], args.tier, args.cached_input, args.subject
                )

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_cost_summary(result, args.detailed)
            else:
                # Multiple model comparison
                result = estimator.compare_models(args.benchmark, args.model, args.tier, args.subject)

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_model_comparison(result)

        elif args.command == "compare":
            models = [m.strip() for m in args.models.split(",")]
            result = estimator.compare_models(args.benchmark, models, args.tier, args.subject)

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_model_comparison(result)

        elif args.command == "list-models":
            list_available_models(estimator, args.tier)

        elif args.command == "list-benchmarks":
            list_benchmarks()

        elif args.command == "file":
            result = estimator.estimate_cost_for_file(args.file, args.model, args.tier, args.cached_input)

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("\n=== File Cost Estimation ===")
                    print(f"File: {result['file_path']}")
                    print(f"Model: {result['model']}")
                    print(f"Tier: {result['tier']}")
                    print(f"Items: {result['total_items']:,}")
                    print(f"Total Tokens: {format_tokens(result['total_tokens'])}")
                    print(f"Total Cost: {format_cost(result['total_cost'])}")
                    print(f"Average Cost per Item: {format_cost(result['average_cost_per_item'])}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

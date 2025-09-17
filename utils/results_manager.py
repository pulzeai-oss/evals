"""
Results manager for saving and loading evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResultsManager:
    """Manages saving and loading of evaluation results."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the results manager.

        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def save_results(
        self,
        results: List[Dict[str, Any]],
        benchmark: str,
        model: str,
        template: str = "default",
        rater_model: str = "default",
        subject: Optional[str] = None,
    ) -> str:
        """
        Save evaluation results to a JSONL file.

        Args:
            results: List of evaluation results
            benchmark: Benchmark name
            model: Model name
            template: Template used
            rater_model: Rater model used
            subject: Optional subject filter

        Returns:
            Path to the saved results file
        """
        # Create benchmark directory
        benchmark_dir = self.results_dir / benchmark
        benchmark_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_model = model.replace("/", "_").replace(":", "_")
        clean_template = template.replace("/", "_").replace(":", "_")
        clean_rater = rater_model.replace("/", "_").replace(":", "_")

        filename_parts = [clean_model, clean_template, clean_rater, timestamp]

        if subject:
            clean_subject = subject.replace("/", "_").replace(":", "_")
            filename_parts.insert(-1, clean_subject)

        filename = "_".join(filename_parts) + ".jsonl"
        filepath = benchmark_dir / filename

        # Save results
        with open(filepath, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Results saved to: {filepath}")
        return str(filepath)

    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load evaluation results from a JSONL file.
        Handles both legacy format (with detailed scoring fields) and new format (with simple score field).

        Args:
            filepath: Path to the results file

        Returns:
            List of evaluation results
        """
        results = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)

                    # Handle different score formats
                    if "score" not in result or result["score"] == 0.0:
                        if "overall_score" in result:
                            # Marketing format: use overall_score
                            result["score"] = result["overall_score"]
                        elif any(
                            field in result
                            for field in [
                                "relevance",
                                "correctness",
                                "clarity",
                                "completeness",
                                "conciseness",
                                "appropriateness",
                                "helpfulness",
                            ]
                        ):
                            # Legacy format: convert detailed scores to overall score
                            score_fields = [
                                "relevance",
                                "correctness",
                                "clarity",
                                "completeness",
                                "conciseness",
                                "appropriateness",
                                "helpfulness",
                            ]
                            scores = [result.get(field, 0) for field in score_fields if field in result]
                            if scores:
                                result["score"] = sum(scores) / len(scores) / 10.0  # Normalize to 0-1 scale
                            else:
                                result["score"] = 0.0
                        elif "label" in result:
                            # Binary scoring based on label (for FinanceBench files)
                            if result["label"] == "Correct Answer":
                                result["score"] = 1.0
                            elif result["label"] == "Incorrect Answer":
                                result["score"] = 0.0
                            else:
                                result["score"] = 0.0
                        else:
                            result["score"] = 0.0

                    # Ensure other required fields exist
                    if "model" not in result:
                        # Try different model field names
                        if "answering_model" in result:
                            result["model"] = result["answering_model"]
                        elif "model_name" in result:
                            result["model"] = result["model_name"]
                        else:
                            result["model"] = "unknown"
                    if "benchmark" not in result:
                        # Try to infer benchmark from filepath
                        path_parts = filepath.split("/")
                        if len(path_parts) >= 3 and path_parts[-2] == "results":
                            # Handle paths like "marketing/results/file.jsonl"
                            benchmark_name = path_parts[-3]
                        elif len(path_parts) >= 2:
                            # Handle paths like "results/benchmark/file.jsonl"
                            benchmark_name = path_parts[-2]
                        else:
                            benchmark_name = "unknown"
                        result["benchmark"] = benchmark_name
                    if "subject" not in result:
                        result["subject"] = "default"
                    if "template" not in result:
                        result["template"] = "default"
                    if "rater_model" not in result:
                        result["rater_model"] = "default"
                    if "timestamp" not in result:
                        result["timestamp"] = "unknown"

                    results.append(result)
        return results

    def _discover_benchmark_directories(self) -> List[str]:
        """
        Auto-discover benchmark directories that contain results.

        Returns:
            List of benchmark directory names
        """
        benchmark_dirs = []

        # Check current directory for benchmark folders with results
        current_dir = Path(".")
        for item in current_dir.iterdir():
            if item.is_dir() and item.name not in [".git", "__pycache__", "tests", "utils", "evaluators"]:
                # Check if it has a results directory with .jsonl files
                results_dir = item / "results"
                if results_dir.exists() and any(results_dir.glob("*.jsonl")):
                    benchmark_dirs.append(item.name)

        return benchmark_dirs

    def get_all_results(self, benchmark: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get all available result files, optionally filtered by benchmark.
        Auto-discovers benchmark directories with results.

        Args:
            benchmark: Optional benchmark filter

        Returns:
            Dictionary mapping benchmark names to lists of result files
        """
        all_results = {}

        if benchmark:
            # Get results for specific benchmark from results/ directory
            benchmark_dir = self.results_dir / benchmark
            if benchmark_dir.exists():
                result_files = [str(f) for f in benchmark_dir.glob("*.jsonl")]
                all_results[benchmark] = sorted(result_files)

            # Also check for benchmark-specific results directory (e.g., marketing/results)
            benchmark_results_dir = Path(benchmark) / "results"
            if benchmark_results_dir.exists():
                benchmark_files = [str(f) for f in benchmark_results_dir.glob("*.jsonl")]
                if benchmark_files:
                    all_results[benchmark] = sorted(all_results.get(benchmark, []) + benchmark_files)
        else:
            # Get results from standard results/ directory
            for benchmark_dir in self.results_dir.iterdir():
                if benchmark_dir.is_dir():
                    benchmark_name = benchmark_dir.name
                    result_files = [str(f) for f in benchmark_dir.glob("*.jsonl")]
                    if result_files:
                        all_results[benchmark_name] = sorted(result_files)

            # Auto-discover benchmark directories with their own results folders
            discovered_benchmarks = self._discover_benchmark_directories()
            for benchmark_name in discovered_benchmarks:
                benchmark_results_dir = Path(benchmark_name) / "results"
                if benchmark_results_dir.exists():
                    benchmark_files = [str(f) for f in benchmark_results_dir.glob("*.jsonl")]
                    if benchmark_files:
                        all_results[benchmark_name] = sorted(all_results.get(benchmark_name, []) + benchmark_files)

        return all_results

    def get_latest_results(self, benchmark: str, model: Optional[str] = None) -> Optional[str]:
        """
        Get the latest results file for a benchmark and optionally a specific model.

        Args:
            benchmark: Benchmark name
            model: Optional model filter

        Returns:
            Path to the latest results file, or None if not found
        """
        benchmark_dir = self.results_dir / benchmark
        if not benchmark_dir.exists():
            return None

        result_files = list(benchmark_dir.glob("*.jsonl"))
        if not result_files:
            return None

        if model:
            clean_model = model.replace("/", "_").replace(":", "_")
            result_files = [f for f in result_files if clean_model in f.name]

        if not result_files:
            return None

        # Sort by modification time and return the latest
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)

    def aggregate_results(self, benchmark: str) -> Dict[str, Any]:
        """
        Aggregate results for a benchmark across all models and runs.

        Args:
            benchmark: Benchmark name

        Returns:
            Aggregated results summary
        """
        all_results = []

        # Get all result files for this benchmark (from both locations)
        benchmark_results = self.get_all_results(benchmark)
        if benchmark not in benchmark_results:
            return {}

        result_files = benchmark_results[benchmark]

        for filepath in result_files:
            try:
                results = self.load_results(filepath)
                all_results.extend(results)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

        if not all_results:
            return {}

        # Aggregate by model and subject
        aggregated = {}
        for result in all_results:
            model = result.get("model", "unknown")
            subject = result.get("subject", "default")
            score = result.get("score", 0.0)

            # Skip entries with unknown/invalid model names
            if model.lower() in ["unknown", "unknown model", ""]:
                continue

            if model not in aggregated:
                aggregated[model] = {}

            if subject not in aggregated[model]:
                aggregated[model][subject] = {"scores": [], "count": 0, "total_score": 0.0}

            aggregated[model][subject]["scores"].append(score)
            aggregated[model][subject]["count"] += 1
            aggregated[model][subject]["total_score"] += score

        # Calculate averages
        for model in aggregated:
            for subject in aggregated[model]:
                data = aggregated[model][subject]
                data["average_score"] = data["total_score"] / data["count"] if data["count"] > 0 else 0.0
                data["min_score"] = min(data["scores"]) if data["scores"] else 0.0
                data["max_score"] = max(data["scores"]) if data["scores"] else 0.0

        return aggregated

    def export_results_csv(self, benchmark: str, output_file: str):
        """
        Export aggregated results to CSV format.

        Args:
            benchmark: Benchmark name
            output_file: Output CSV file path
        """
        aggregated = self.aggregate_results(benchmark)
        if not aggregated:
            print(f"No results found for benchmark: {benchmark}")
            return

        import csv

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(["Model", "Subject", "Average Score", "Count", "Min Score", "Max Score"])

            # Write data
            for model in sorted(aggregated.keys()):
                for subject in sorted(aggregated[model].keys()):
                    data = aggregated[model][subject]
                    writer.writerow(
                        [
                            model,
                            subject,
                            f"{data['average_score']:.4f}",
                            data["count"],
                            f"{data['min_score']:.4f}",
                            f"{data['max_score']:.4f}",
                        ]
                    )

        print(f"Results exported to: {output_file}")

    def cleanup_old_results(self, benchmark: str, keep_latest: int = 10):
        """
        Clean up old result files, keeping only the latest N files per benchmark.

        Args:
            benchmark: Benchmark name
            keep_latest: Number of latest files to keep
        """
        benchmark_dir = self.results_dir / benchmark
        if not benchmark_dir.exists():
            return

        result_files = list(benchmark_dir.glob("*.jsonl"))
        if len(result_files) <= keep_latest:
            return

        # Sort by modification time (oldest first)
        result_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest files
        files_to_remove = result_files[:-keep_latest]
        for filepath in files_to_remove:
            try:
                filepath.unlink()
                print(f"Removed old result file: {filepath}")
            except Exception as e:
                print(f"Error removing {filepath}: {e}")

    def get_result_summary(self, filepath: str) -> Dict[str, Any]:
        """
        Get a summary of a specific result file.

        Args:
            filepath: Path to the results file

        Returns:
            Summary information
        """
        try:
            results = self.load_results(filepath)
            if not results:
                return {}

            # Extract metadata from first result
            first_result = results[0]

            # Calculate statistics
            scores = [r.get("score", 0.0) for r in results if "score" in r]

            summary = {
                "filepath": filepath,
                "benchmark": first_result.get("benchmark", "unknown"),
                "model": first_result.get("model", "unknown"),
                "template": first_result.get("template", "unknown"),
                "rater_model": first_result.get("rater_model", "unknown"),
                "subject": first_result.get("subject", "unknown"),
                "timestamp": first_result.get("timestamp", "unknown"),
                "total_items": len(results),
                "items_with_scores": len(scores),
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "error_count": len([r for r in results if "error" in r]),
            }

            return summary

        except Exception as e:
            return {"error": str(e), "filepath": filepath}

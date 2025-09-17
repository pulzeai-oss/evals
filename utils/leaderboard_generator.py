"""
Leaderboard generator for visualizing evaluation results.
"""

from typing import Any, Dict, List, Optional

from .results_manager import ResultsManager


class LeaderboardGenerator:
    """Generates leaderboards from evaluation results."""

    def __init__(self, results_manager: ResultsManager):
        """
        Initialize the leaderboard generator.

        Args:
            results_manager: ResultsManager instance
        """
        self.results_manager = results_manager

    def generate_benchmark_leaderboard(self, benchmark: str, sort_by: str = "average") -> Dict[str, Any]:
        """
        Generate a leaderboard for a specific benchmark.

        Args:
            benchmark: Benchmark name
            sort_by: Sort criteria ("average", "total", "count")

        Returns:
            Leaderboard data
        """
        aggregated = self.results_manager.aggregate_results(benchmark)
        if not aggregated:
            return {"error": f"No results found for benchmark: {benchmark}"}

        # Calculate overall scores for each model
        model_scores = {}
        for model, subjects in aggregated.items():
            total_score = 0.0
            total_count = 0
            subject_scores = {}

            for subject, data in subjects.items():
                subject_scores[subject] = {
                    "average_score": data["average_score"],
                    "count": data["count"],
                    "min_score": data["min_score"],
                    "max_score": data["max_score"],
                }
                total_score += data["total_score"]
                total_count += data["count"]

            overall_average = total_score / total_count if total_count > 0 else 0.0

            model_scores[model] = {
                "overall_average": overall_average,
                "total_score": total_score,
                "total_count": total_count,
                "subjects": subject_scores,
            }

        # Sort models based on criteria
        if sort_by == "average":
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["overall_average"], reverse=True)
        elif sort_by == "total":
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["total_score"], reverse=True)
        elif sort_by == "count":
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["total_count"], reverse=True)
        else:
            sorted_models = list(model_scores.items())

        return {"benchmark": benchmark, "sort_by": sort_by, "models": sorted_models, "total_models": len(sorted_models)}

    def generate_cross_benchmark_leaderboard(self, benchmarks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a cross-benchmark leaderboard.

        Args:
            benchmarks: List of benchmarks to include (None for all)

        Returns:
            Cross-benchmark leaderboard data
        """
        if benchmarks is None:
            all_results = self.results_manager.get_all_results()
            benchmarks = list(all_results.keys())

        cross_benchmark_data = {}

        for benchmark in benchmarks:
            benchmark_data = self.generate_benchmark_leaderboard(benchmark)
            if "error" not in benchmark_data:
                cross_benchmark_data[benchmark] = benchmark_data

        # Calculate cross-benchmark model rankings
        model_cross_scores = {}

        for benchmark, data in cross_benchmark_data.items():
            for model, model_data in data["models"]:
                if model not in model_cross_scores:
                    model_cross_scores[model] = {"benchmarks": {}, "total_average": 0.0, "benchmark_count": 0}

                model_cross_scores[model]["benchmarks"][benchmark] = model_data["overall_average"]
                model_cross_scores[model]["total_average"] += model_data["overall_average"]
                model_cross_scores[model]["benchmark_count"] += 1

        # Calculate overall averages across benchmarks
        for model in model_cross_scores:
            count = model_cross_scores[model]["benchmark_count"]
            if count > 0:
                model_cross_scores[model]["overall_cross_average"] = model_cross_scores[model]["total_average"] / count
            else:
                model_cross_scores[model]["overall_cross_average"] = 0.0

        # Sort by cross-benchmark average
        sorted_cross_models = sorted(
            model_cross_scores.items(), key=lambda x: x[1]["overall_cross_average"], reverse=True
        )

        return {
            "benchmarks": benchmarks,
            "benchmark_data": cross_benchmark_data,
            "cross_benchmark_ranking": sorted_cross_models,
            "total_models": len(sorted_cross_models),
        }

    def print_benchmark_leaderboard(self, benchmark: str, sort_by: str = "average", top_n: Optional[int] = None):
        """
        Print a formatted leaderboard for a benchmark.

        Args:
            benchmark: Benchmark name
            sort_by: Sort criteria
            top_n: Number of top models to show (None for all)
        """
        leaderboard = self.generate_benchmark_leaderboard(benchmark, sort_by)

        if "error" in leaderboard:
            print(f"Error: {leaderboard['error']}")
            return

        print(f"\n{'='*80}")
        print(f"LEADERBOARD: {benchmark.upper()}")
        print(f"Sorted by: {sort_by}")
        print(f"{'='*80}")

        models = leaderboard["models"]
        if top_n:
            models = models[:top_n]

        print(f"{'Rank':<6} {'Model':<30} {'Overall Avg':<12} {'Total Count':<12} {'Subjects':<10}")
        print("-" * 80)

        for rank, (model, data) in enumerate(models, 1):
            subject_count = len(data["subjects"])
            print(
                f"{rank:<6} {model:<30} {data['overall_average']:<12.4f} "
                f"{data['total_count']:<12} {subject_count:<10}"
            )

        print("-" * 80)
        print(f"Total models: {len(leaderboard['models'])}")

        # Show subject breakdown for top model if available
        if models:
            top_model, top_data = models[0]
            print(f"\nSubject breakdown for top model ({top_model}):")
            print(f"{'Subject':<30} {'Avg Score':<12} {'Count':<8} {'Min':<8} {'Max':<8}")
            print("-" * 70)

            for subject, subject_data in sorted(top_data["subjects"].items()):
                print(
                    f"{subject:<30} {subject_data['average_score']:<12.4f} "
                    f"{subject_data['count']:<8} {subject_data['min_score']:<8.4f} "
                    f"{subject_data['max_score']:<8.4f}"
                )

    def print_cross_benchmark_leaderboard(self, benchmarks: Optional[List[str]] = None, top_n: Optional[int] = None):
        """
        Print a formatted cross-benchmark leaderboard.

        Args:
            benchmarks: List of benchmarks to include
            top_n: Number of top models to show
        """
        leaderboard = self.generate_cross_benchmark_leaderboard(benchmarks)

        print(f"\n{'='*100}")
        print("CROSS-BENCHMARK LEADERBOARD")
        print(f"Benchmarks: {', '.join(leaderboard['benchmarks'])}")
        print(f"{'='*100}")

        models = leaderboard["cross_benchmark_ranking"]
        if top_n:
            models = models[:top_n]

        # Print header
        header = f"{'Rank':<6} {'Model':<25} {'Cross Avg':<12}"
        for benchmark in leaderboard["benchmarks"]:
            header += f" {benchmark[:10]:<12}"
        print(header)
        print("-" * 100)

        # Print model data
        for rank, (model, data) in enumerate(models, 1):
            row = f"{rank:<6} {model:<25} {data['overall_cross_average']:<12.4f}"

            for benchmark in leaderboard["benchmarks"]:
                score = data["benchmarks"].get(benchmark, 0.0)
                row += f" {score:<12.4f}"

            print(row)

        print("-" * 100)
        print(f"Total models: {len(leaderboard['cross_benchmark_ranking'])}")

    def export_leaderboard_html(self, benchmark: str, output_file: str, sort_by: str = "average"):
        """
        Export leaderboard to HTML format.

        Args:
            benchmark: Benchmark name
            output_file: Output HTML file path
            sort_by: Sort criteria
        """
        leaderboard = self.generate_benchmark_leaderboard(benchmark, sort_by)

        if "error" in leaderboard:
            print(f"Error: {leaderboard['error']}")
            return

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{benchmark.title()} Leaderboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .rank {{ font-weight: bold; }}
        .model {{ font-weight: bold; color: #2c3e50; }}
        .score {{ text-align: right; }}
    </style>
</head>
<body>
    <h1>{benchmark.title()} Leaderboard</h1>
    <p>Sorted by: {sort_by}</p>
    <p>Generated on: {self._get_timestamp()}</p>

    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Overall Average</th>
                <th>Total Count</th>
                <th>Subjects</th>
            </tr>
        </thead>
        <tbody>
"""

        for rank, (model, data) in enumerate(leaderboard["models"], 1):
            subject_count = len(data["subjects"])
            html_content += f"""
            <tr>
                <td class="rank">{rank}</td>
                <td class="model">{model}</td>
                <td class="score">{data['overall_average']:.4f}</td>
                <td class="score">{data['total_count']}</td>
                <td class="score">{subject_count}</td>
            </tr>
"""

        html_content += """
        </tbody>
    </table>

    <h2>Subject Details</h2>
"""

        # Add subject details for each model
        for model, data in leaderboard["models"]:
            html_content += f"""
    <h3>{model}</h3>
    <table>
        <thead>
            <tr>
                <th>Subject</th>
                <th>Average Score</th>
                <th>Count</th>
                <th>Min Score</th>
                <th>Max Score</th>
            </tr>
        </thead>
        <tbody>
"""
            for subject, subject_data in sorted(data["subjects"].items()):
                html_content += f"""
            <tr>
                <td>{subject}</td>
                <td class="score">{subject_data['average_score']:.4f}</td>
                <td class="score">{subject_data['count']}</td>
                <td class="score">{subject_data['min_score']:.4f}</td>
                <td class="score">{subject_data['max_score']:.4f}</td>
            </tr>
"""
            html_content += """
        </tbody>
    </table>
"""

        html_content += """
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html_content)

        print(f"HTML leaderboard exported to: {output_file}")

    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

"""
Base evaluator class for all benchmark evaluations.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    def __init__(self, benchmark_name: str, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Args:
            benchmark_name: Name of the benchmark
            config: Configuration dictionary containing API keys and settings
        """
        self.benchmark_name = benchmark_name
        self.config = config
        self.results = []

    @abstractmethod
    def load_data(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load benchmark data, optionally filtered by subject.

        Args:
            subject: Optional subject filter

        Returns:
            List of evaluation items
        """
        pass

    @abstractmethod
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available subjects for this benchmark.

        Returns:
            List of subject names
        """
        pass

    @abstractmethod
    def evaluate_item(self, item: Dict[str, Any], model: str, template: str, rater_model: str) -> Dict[str, Any]:
        """
        Evaluate a single item.

        Args:
            item: The evaluation item
            model: Model to evaluate
            template: Template to use for evaluation
            rater_model: Model to use for rating/scoring

        Returns:
            Evaluation result
        """
        pass

    def evaluate(
        self, model: str, template: str, rater_model: str, subject: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on the benchmark.

        Args:
            model: Model to evaluate
            template: Template to use
            rater_model: Model to use for rating
            subject: Optional subject filter

        Returns:
            List of evaluation results
        """
        data = self.load_data(subject)
        results = []

        print(f"Evaluating {len(data)} items from {self.benchmark_name}")
        if subject:
            print(f"Subject: {subject}")

        for i, item in enumerate(data, 1):
            print(f"Processing item {i}/{len(data)}", end="\r")
            try:
                result = self.evaluate_item(item, model, template, rater_model)
                result.update(
                    {
                        "benchmark": self.benchmark_name,
                        "subject": subject or item.get("subject", "default"),
                        "model": model,
                        "template": template,
                        "rater_model": rater_model,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                results.append(result)
            except Exception as e:
                print(f"\nError processing item {i}: {e}")
                continue

        print(f"\nCompleted evaluation: {len(results)}/{len(data)} items processed")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save evaluation results to file.

        Args:
            results: List of evaluation results
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Results saved to {output_file}")

    def _get_api_client(self, model: str):
        """
        Get appropriate API client based on model name.

        Args:
            model: Model name

        Returns:
            API client instance
        """
        if model.startswith("pulze/"):
            return self._get_pulze_client()
        elif model.startswith("openai/") or model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
            return self._get_openai_client()
        elif model.startswith("anthropic/"):
            return self._get_anthropic_client()
        else:
            # Default to OpenAI-compatible endpoint
            return self._get_openai_client()

    def _get_pulze_client(self):
        """Get Pulze API client."""
        try:
            import openai

            return openai.OpenAI(
                api_key=self.config.get("PULZE_API_KEY"),
                base_url=self.config.get("PULZE_BASE_URL", "https://api.pulze.ai/v1"),
            )
        except ImportError:
            raise ImportError("openai package required for Pulze API")

    def _get_openai_client(self):
        """Get OpenAI API client."""
        try:
            import openai

            return openai.OpenAI(
                api_key=self.config.get("OPENAI_API_KEY"),
                base_url=self.config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        except ImportError:
            raise ImportError("openai package required for OpenAI API")

    def _get_anthropic_client(self):
        """Get Anthropic API client (via OpenAI-compatible endpoint)."""
        try:
            import openai

            return openai.OpenAI(
                api_key=self.config.get("OPENAI_API_KEY"),
                base_url=self.config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        except ImportError:
            raise ImportError("openai package required for Anthropic API")

    def _make_api_call(self, client, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Make API call to get model response.

        Args:
            client: API client
            model: Model name
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            Model response text
        """
        try:
            # Clean model name for API call
            clean_model = model.replace("pulze/", "").replace("openai/", "").replace("anthropic/", "")

            response = client.chat.completions.create(model=clean_model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"API call failed for model {model}: {e}")

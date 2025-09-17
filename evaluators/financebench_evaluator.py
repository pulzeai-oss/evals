"""
FinanceBench evaluator for financial document analysis.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class FinanceBenchEvaluator(BaseEvaluator):
    """Evaluator for FinanceBench benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("financebench", config)
        self.data_dir = "financebench/data"

    def load_data(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load FinanceBench data.

        Args:
            subject: Not used for FinanceBench (single subject)

        Returns:
            List of evaluation items
        """
        data_file = os.path.join(self.data_dir, "financebench_open_source.jsonl")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"FinanceBench data file not found: {data_file}")

        data = []
        with open(data_file, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)

        return data

    def get_available_subjects(self) -> List[str]:
        """
        Get available subjects for FinanceBench.

        Returns:
            List containing single subject
        """
        return ["financial_analysis"]

    def evaluate_item(self, item: Dict[str, Any], model: str, template: str, rater_model: str) -> Dict[str, Any]:
        """
        Evaluate a single FinanceBench item.

        Args:
            item: The evaluation item
            model: Model to evaluate
            template: Template to use (ignored for FinanceBench)
            rater_model: Model to use for rating

        Returns:
            Evaluation result
        """
        # Get API client for the model being evaluated
        client = self._get_api_client(model)

        # Construct the prompt for financial analysis
        question = item.get("question", "")
        evidence = item.get("evidence", "")

        # Create the evaluation prompt
        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst. Answer the following question based on the provided evidence from financial documents. Be precise and factual.",
            },
            {
                "role": "user",
                "content": f"Evidence: {evidence}\n\nQuestion: {question}\n\nProvide a clear, factual answer based on the evidence.",
            },
        ]

        # Get model response
        try:
            response = self._make_api_call(client, model, messages)
        except Exception as e:
            return {
                "question_id": item.get("id", "unknown"),
                "question": question,
                "expected_answer": item.get("answer", ""),
                "model_answer": "",
                "error": str(e),
                "score": 0.0,
            }

        # Rate the response using the rater model
        score = self._rate_response(item, response, rater_model)

        return {
            "question_id": item.get("id", "unknown"),
            "question": question,
            "expected_answer": item.get("answer", ""),
            "model_answer": response,
            "score": score,
            "evidence": evidence,
        }

    def _rate_response(self, item: Dict[str, Any], response: str, rater_model: str) -> float:
        """
        Rate the model response against the expected answer.

        Args:
            item: Original evaluation item
            response: Model response
            rater_model: Model to use for rating

        Returns:
            Score between 0.0 and 1.0
        """
        expected_answer = item.get("answer", "")
        question = item.get("question", "")

        # Get rater client
        rater_client = self._get_api_client(rater_model)

        rating_prompt = f"""
You are evaluating the quality of an answer to a financial analysis question.

Question: {question}
Expected Answer: {expected_answer}
Model Answer: {response}

Rate the model answer on a scale of 0.0 to 1.0 based on:
1. Factual accuracy compared to the expected answer
2. Completeness of the response
3. Clarity and precision

Provide only a numeric score between 0.0 and 1.0, with no additional text.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator for financial analysis tasks. Provide only numeric scores.",
            },
            {"role": "user", "content": rating_prompt},
        ]

        try:
            rating_response = self._make_api_call(rater_client, rater_model, messages)
            # Extract numeric score
            score_str = rating_response.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except (ValueError, Exception):
            # Fallback: simple string matching
            return self._simple_string_match(expected_answer, response)

    def _simple_string_match(self, expected: str, actual: str) -> float:
        """
        Simple fallback scoring based on string similarity.

        Args:
            expected: Expected answer
            actual: Actual answer

        Returns:
            Score between 0.0 and 1.0
        """
        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        if expected_lower == actual_lower:
            return 1.0
        elif expected_lower in actual_lower or actual_lower in expected_lower:
            return 0.7
        else:
            # Simple word overlap scoring
            expected_words = set(expected_lower.split())
            actual_words = set(actual_lower.split())

            if not expected_words:
                return 0.0

            overlap = len(expected_words.intersection(actual_words))
            return overlap / len(expected_words)

"""
MMLU evaluator for multiple-choice questions across various subjects.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class MMLUEvaluator(BaseEvaluator):
    """Evaluator for MMLU benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("mmlu", config)
        self.data_dir = "mmlu/data"

    def load_data(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load MMLU data, optionally filtered by subject.

        Args:
            subject: Optional subject filter

        Returns:
            List of evaluation items
        """
        data = []

        if subject:
            # Load specific subject
            data_file = os.path.join(self.data_dir, f"mmlu_{subject}.jsonl")
            if os.path.exists(data_file):
                data.extend(self._load_jsonl_file(data_file, subject))
            else:
                raise FileNotFoundError(f"MMLU subject file not found: {data_file}")
        else:
            # Load all subjects
            for filename in os.listdir(self.data_dir):
                if filename.startswith("mmlu_") and filename.endswith(".jsonl"):
                    subject_name = filename[5:-6]  # Remove "mmlu_" prefix and ".jsonl" suffix
                    filepath = os.path.join(self.data_dir, filename)
                    data.extend(self._load_jsonl_file(filepath, subject_name))

        return data

    def _load_jsonl_file(self, filepath: str, subject: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file and add subject information."""
        data = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["subject"] = subject
                    data.append(item)
        return data

    def get_available_subjects(self) -> List[str]:
        """
        Get available subjects for MMLU.

        Returns:
            List of subject names
        """
        subjects = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.startswith("mmlu_") and filename.endswith(".jsonl"):
                    subject = filename[5:-6]  # Remove "mmlu_" prefix and ".jsonl" suffix
                    subjects.append(subject)
        return sorted(subjects)

    def evaluate_item(self, item: Dict[str, Any], model: str, template: str, rater_model: str) -> Dict[str, Any]:
        """
        Evaluate a single MMLU item.

        Args:
            item: The evaluation item
            model: Model to evaluate
            template: Template to use (ignored for MMLU)
            rater_model: Model to use for rating (not used for multiple choice)

        Returns:
            Evaluation result
        """
        # Get API client for the model being evaluated
        client = self._get_api_client(model)

        # Extract question components
        question = item.get("question", "")
        choices = item.get("choices", [])
        correct_answer = item.get("answer", "")
        subject = item.get("subject", "unknown")

        # Format choices
        choice_text = ""
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices[:4]):  # MMLU typically has 4 choices
            if i < len(choice_labels):
                choice_text += f"{choice_labels[i]}) {choice}\n"

        # Create the evaluation prompt
        messages = [
            {
                "role": "system",
                "content": f"You are answering multiple choice questions in the subject of {subject.replace('_', ' ')}. Choose the best answer from the given options.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\n{choice_text}\nAnswer with only the letter (A, B, C, or D):",
            },
        ]

        # Get model response
        try:
            response = self._make_api_call(client, model, messages, max_tokens=10, temperature=0.0)
            predicted_answer = self._extract_answer_choice(response)
        except Exception as e:
            return {
                "question_id": item.get("id", "unknown"),
                "question": question,
                "choices": choices,
                "correct_answer": correct_answer,
                "predicted_answer": "",
                "model_response": "",
                "error": str(e),
                "score": 0.0,
                "subject": subject,
            }

        # Calculate score (1.0 for correct, 0.0 for incorrect)
        score = 1.0 if predicted_answer == correct_answer else 0.0

        return {
            "question_id": item.get("id", "unknown"),
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "model_response": response,
            "score": score,
            "subject": subject,
        }

    def _extract_answer_choice(self, response: str) -> str:
        """
        Extract the answer choice (A, B, C, D) from the model response.

        Args:
            response: Raw model response

        Returns:
            Extracted answer choice or empty string if not found
        """
        response = response.strip().upper()

        # Look for single letter answers
        for choice in ["A", "B", "C", "D"]:
            if choice in response:
                # Check if it's the first occurrence and likely the answer
                if response.startswith(choice) or f" {choice}" in response or f"({choice})" in response:
                    return choice

        # Fallback: return first letter if it's A, B, C, or D
        if response and response[0] in ["A", "B", "C", "D"]:
            return response[0]

        return ""

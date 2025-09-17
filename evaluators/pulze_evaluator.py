"""
Pulze evaluator for template-based evaluations.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class PulzeEvaluator(BaseEvaluator):
    """Evaluator for Pulze-v0.1 benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("pulze", config)
        self.data_dir = "pulze-v0.1/data"
        self.templates_dir = "pulze-v0.1/templates"

    def load_data(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load Pulze data, optionally filtered by subject.

        Args:
            subject: Optional subject filter

        Returns:
            List of evaluation items
        """
        data = []

        if subject:
            # Load specific subject
            data_file = os.path.join(self.data_dir, f"{subject}.jsonl")
            if os.path.exists(data_file):
                data.extend(self._load_jsonl_file(data_file, subject))
            else:
                raise FileNotFoundError(f"Pulze subject file not found: {data_file}")
        else:
            # Load all subjects
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith(".jsonl"):
                        subject_name = filename[:-6]  # Remove ".jsonl" suffix
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
        Get available subjects for Pulze.

        Returns:
            List of subject names
        """
        subjects = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".jsonl"):
                    subject = filename[:-6]  # Remove ".jsonl" suffix
                    subjects.append(subject)
        return sorted(subjects)

    def get_available_templates(self) -> List[str]:
        """
        Get available templates for Pulze evaluations.

        Returns:
            List of template names
        """
        templates = []
        if os.path.exists(self.templates_dir):
            for filename in os.listdir(self.templates_dir):
                if filename.endswith(".json"):
                    template = filename[:-5]  # Remove ".json" suffix
                    templates.append(template)
        return sorted(templates)

    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a template configuration.

        Args:
            template_name: Name of the template

        Returns:
            Template configuration
        """
        template_file = os.path.join(self.templates_dir, f"{template_name}.json")
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Template not found: {template_file}")

        with open(template_file, "r") as f:
            return json.load(f)

    def evaluate_item(self, item: Dict[str, Any], model: str, template: str, rater_model: str) -> Dict[str, Any]:
        """
        Evaluate a single Pulze item using the specified template.

        Args:
            item: The evaluation item
            model: Model to evaluate
            template: Template to use for evaluation
            rater_model: Model to use for rating

        Returns:
            Evaluation result
        """
        # Load template configuration
        try:
            template_config = self.load_template(template)
        except FileNotFoundError:
            return {
                "question_id": item.get("id", "unknown"),
                "error": f"Template not found: {template}",
                "score": 0.0,
                "subject": item.get("subject", "unknown"),
            }

        # Get API client for the model being evaluated
        client = self._get_api_client(model)

        # Extract item data
        question = item.get("question", "")
        context = item.get("context", "")
        subject = item.get("subject", "unknown")

        # Build prompt using template
        prompt = self._build_prompt_from_template(template_config, question, context)

        # Create messages
        messages = [
            {"role": "system", "content": template_config.get("system_prompt", "You are a helpful assistant.")},
            {"role": "user", "content": prompt},
        ]

        # Get model response
        try:
            response = self._make_api_call(client, model, messages)
        except Exception as e:
            return {
                "question_id": item.get("id", "unknown"),
                "question": question,
                "model_response": "",
                "error": str(e),
                "score": 0.0,
                "subject": subject,
            }

        # Rate the response using the template's evaluation criteria
        score = self._rate_response_with_template(item, response, template_config, rater_model)

        return {
            "question_id": item.get("id", "unknown"),
            "question": question,
            "context": context,
            "model_response": response,
            "score": score,
            "subject": subject,
            "template_used": template,
        }

    def _build_prompt_from_template(self, template_config: Dict[str, Any], question: str, context: str) -> str:
        """
        Build prompt using template configuration.

        Args:
            template_config: Template configuration
            question: Question text
            context: Context text

        Returns:
            Formatted prompt
        """
        prompt_template = template_config.get("prompt_template", "{question}")

        # Replace placeholders
        prompt = prompt_template.replace("{question}", question)
        prompt = prompt.replace("{context}", context)

        return prompt

    def _rate_response_with_template(
        self, item: Dict[str, Any], response: str, template_config: Dict[str, Any], rater_model: str
    ) -> float:
        """
        Rate the response using template-specific evaluation criteria.

        Args:
            item: Original evaluation item
            response: Model response
            template_config: Template configuration
            rater_model: Model to use for rating

        Returns:
            Score between 0.0 and 1.0
        """
        # Get evaluation criteria from template
        evaluation_criteria = template_config.get("evaluation_criteria", [])

        if not evaluation_criteria:
            # Fallback to simple scoring if no criteria specified
            return 0.5

        # Get rater client
        rater_client = self._get_api_client(rater_model)

        # Build evaluation prompt
        criteria_text = "\n".join([f"- {criterion}" for criterion in evaluation_criteria])

        rating_prompt = f"""
Evaluate the following response based on these criteria:
{criteria_text}

Question: {item.get('question', '')}
Context: {item.get('context', '')}
Response: {response}

Rate the response on a scale of 0.0 to 1.0 based on how well it meets the evaluation criteria.
Provide only a numeric score between 0.0 and 1.0, with no additional text.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide only numeric scores based on the given criteria.",
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
            # Fallback scoring
            return 0.5

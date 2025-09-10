"""
Marketing evaluator for marketing-specific evaluations.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class MarketingEvaluator(BaseEvaluator):
    """Evaluator for Marketing benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("marketing", config)
        self.data_dir = "marketing"

    def load_data(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load Marketing data, optionally filtered by subject.

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
                raise FileNotFoundError(f"Marketing subject file not found: {data_file}")
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
        Get available subjects for Marketing.

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

    def evaluate_item(self, item: Dict[str, Any], model: str, template: str, rater_model: str) -> Dict[str, Any]:
        """
        Evaluate a single Marketing item.

        Args:
            item: The evaluation item
            model: Model to evaluate
            template: Template to use for evaluation
            rater_model: Model to use for rating

        Returns:
            Evaluation result
        """
        # Get API client for the model being evaluated
        client = self._get_api_client(model)

        # Extract item data
        question = item.get("question", "")
        context = item.get("context", "")
        subject = item.get("subject", "unknown")

        # Create the evaluation prompt based on marketing context
        system_prompt = self._get_marketing_system_prompt(subject)
        user_prompt = self._build_marketing_prompt(question, context, template)

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

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

        # Rate the response using marketing-specific criteria
        score = self._rate_marketing_response(item, response, rater_model)

        return {
            "question_id": item.get("id", "unknown"),
            "question": question,
            "context": context,
            "model_response": response,
            "score": score,
            "subject": subject,
            "template_used": template,
        }

    def _get_marketing_system_prompt(self, subject: str) -> str:
        """
        Get system prompt based on marketing subject.

        Args:
            subject: Marketing subject

        Returns:
            System prompt
        """
        if "writing_marketing_materials" in subject:
            return "You are an expert marketing copywriter. Create compelling, persuasive marketing content that engages the target audience and drives action."
        elif "ulta" in subject.lower():
            return "You are a marketing expert specializing in beauty and retail. Provide strategic marketing insights for beauty brands and retail experiences."
        else:
            return "You are a marketing expert. Provide strategic, creative, and data-driven marketing solutions."

    def _build_marketing_prompt(self, question: str, context: str, template: str) -> str:
        """
        Build marketing-specific prompt.

        Args:
            question: Question text
            context: Context text
            template: Template name

        Returns:
            Formatted prompt
        """
        if template == "pulze_multi_dimensional_evaluation":
            # Use multi-dimensional evaluation approach
            prompt = f"""
Context: {context}

Task: {question}

Please provide a comprehensive response that addresses multiple dimensions:
1. Strategic approach
2. Creative execution
3. Target audience considerations
4. Measurable outcomes

Provide a detailed, actionable response.
"""
        else:
            # Default marketing prompt
            prompt = f"""
Context: {context}

Question: {question}

Provide a comprehensive marketing solution that is strategic, creative, and actionable.
"""

        return prompt

    def _rate_marketing_response(self, item: Dict[str, Any], response: str, rater_model: str) -> float:
        """
        Rate the marketing response using marketing-specific criteria.

        Args:
            item: Original evaluation item
            response: Model response
            rater_model: Model to use for rating

        Returns:
            Score between 0.0 and 1.0
        """
        question = item.get("question", "")
        context = item.get("context", "")

        # Get rater client
        rater_client = self._get_api_client(rater_model)

        rating_prompt = f"""
Evaluate the following marketing response based on these criteria:

1. Strategic Thinking (25%): Does the response demonstrate strategic marketing understanding?
2. Creativity (25%): Is the response creative and engaging?
3. Practicality (25%): Is the response actionable and realistic?
4. Audience Focus (25%): Does the response consider the target audience effectively?

Context: {context}
Question: {question}
Response: {response}

Rate the response on a scale of 0.0 to 1.0 based on how well it meets these marketing evaluation criteria.
Consider the overall quality, relevance, and effectiveness of the marketing solution provided.

Provide only a numeric score between 0.0 and 1.0, with no additional text.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert marketing evaluator. Assess responses based on strategic thinking, creativity, practicality, and audience focus. Provide only numeric scores.",
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
            # Fallback: simple keyword-based scoring
            return self._simple_marketing_score(response)

    def _simple_marketing_score(self, response: str) -> float:
        """
        Simple fallback scoring for marketing responses.

        Args:
            response: Model response

        Returns:
            Score between 0.0 and 1.0
        """
        response_lower = response.lower()

        # Marketing keywords that indicate quality
        marketing_keywords = [
            "target audience",
            "brand",
            "campaign",
            "strategy",
            "engagement",
            "conversion",
            "roi",
            "customer",
            "market",
            "positioning",
            "messaging",
            "creative",
            "channel",
            "digital",
            "social media",
            "analytics",
            "metrics",
            "kpi",
            "segmentation",
            "persona",
        ]

        # Count keyword matches
        keyword_matches = sum(1 for keyword in marketing_keywords if keyword in response_lower)

        # Base score on length and keyword density
        word_count = len(response.split())

        if word_count < 50:
            length_score = 0.3
        elif word_count < 150:
            length_score = 0.7
        else:
            length_score = 1.0

        keyword_score = min(1.0, keyword_matches / 10.0)  # Max score when 10+ keywords

        # Combine scores
        final_score = (length_score * 0.4) + (keyword_score * 0.6)
        return max(0.1, min(1.0, final_score))  # Ensure minimum 0.1 score

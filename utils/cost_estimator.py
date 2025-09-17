"""
Cost estimation utility for benchmark evaluations.
Calculates token usage and costs for different AI models based on current pricing.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken


@dataclass
class ModelPricing:
    """Model pricing information per 1M tokens"""

    input_price: float
    output_price: float
    cached_input_price: Optional[float] = None
    tier: str = "standard"  # standard, batch, flex, priority


class CostEstimator:
    """Estimates costs for benchmark evaluations based on model pricing"""

    def __init__(self):
        self.model_pricing = self._load_model_pricing()
        self.tokenizers = {}

    def _load_model_pricing(self) -> Dict[str, Dict[str, ModelPricing]]:
        """Load model pricing information organized by tier"""
        pricing = {
            "batch": {
                # OpenAI Models - Batch
                "gpt-5": ModelPricing(0.625, 5.00, 0.0625),
                "gpt-5-mini": ModelPricing(0.125, 1.00, 0.0125),
                "gpt-5-nano": ModelPricing(0.025, 0.20, 0.0025),
                "gpt-4.1": ModelPricing(1.00, 4.00),
                "gpt-4.1-mini": ModelPricing(0.20, 0.80),
                "gpt-4.1-nano": ModelPricing(0.05, 0.20),
                "gpt-4o": ModelPricing(1.25, 5.00),
                "gpt-4o-2024-05-13": ModelPricing(2.50, 7.50),
                "gpt-4o-mini": ModelPricing(0.075, 0.30),
                "o1": ModelPricing(7.50, 30.00),
                "o1-pro": ModelPricing(75.00, 300.00),
                "o3-pro": ModelPricing(10.00, 40.00),
                "o3": ModelPricing(1.00, 4.00),
                "o3-deep-research": ModelPricing(5.00, 20.00),
                "o4-mini": ModelPricing(0.55, 2.20),
                "o4-mini-deep-research": ModelPricing(1.00, 4.00),
                "o3-mini": ModelPricing(0.55, 2.20),
                "o1-mini": ModelPricing(0.55, 2.20),
                "computer-use-preview": ModelPricing(1.50, 6.00),
                # Legacy models - Batch
                "gpt-4-turbo-2024-04-09": ModelPricing(5.00, 15.00),
                "gpt-4-0125-preview": ModelPricing(5.00, 15.00),
                "gpt-4-1106-preview": ModelPricing(5.00, 15.00),
                "gpt-4-1106-vision-preview": ModelPricing(5.00, 15.00),
                "gpt-4-0613": ModelPricing(15.00, 30.00),
                "gpt-4-0314": ModelPricing(15.00, 30.00),
                "gpt-4-32k": ModelPricing(30.00, 60.00),
                "gpt-3.5-turbo-0125": ModelPricing(0.25, 0.75),
                "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
                "gpt-3.5-turbo-0613": ModelPricing(1.50, 2.00),
                "gpt-3.5-0301": ModelPricing(1.50, 2.00),
                "gpt-3.5-turbo-16k-0613": ModelPricing(1.50, 2.00),
                "davinci-002": ModelPricing(1.00, 1.00),
                "babbage-002": ModelPricing(0.20, 0.20),
            },
            "flex": {
                # OpenAI Models - Flex
                "gpt-5": ModelPricing(0.625, 5.00, 0.0625),
                "gpt-5-mini": ModelPricing(0.125, 1.00, 0.0125),
                "gpt-5-nano": ModelPricing(0.025, 0.20, 0.0025),
                "o3": ModelPricing(1.00, 4.00, 0.25),
                "o4-mini": ModelPricing(0.55, 2.20, 0.138),
            },
            "standard": {
                # OpenAI Models - Standard
                "gpt-5": ModelPricing(1.25, 10.00, 0.125),
                "gpt-5-mini": ModelPricing(0.25, 2.00, 0.025),
                "gpt-5-nano": ModelPricing(0.05, 0.40, 0.005),
                "gpt-5-chat-latest": ModelPricing(1.25, 10.00, 0.125),
                "gpt-4.1": ModelPricing(2.00, 8.00, 0.50),
                "gpt-4.1-mini": ModelPricing(0.40, 1.60, 0.10),
                "gpt-4.1-nano": ModelPricing(0.10, 0.40, 0.025),
                "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
                "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
                "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
                "gpt-realtime": ModelPricing(4.00, 16.00, 0.40),
                "gpt-4o-realtime-preview": ModelPricing(5.00, 20.00, 2.50),
                "gpt-4o-mini-realtime-preview": ModelPricing(0.60, 2.40, 0.30),
                "gpt-audio": ModelPricing(2.50, 10.00),
                "gpt-4o-audio-preview": ModelPricing(2.50, 10.00),
                "gpt-4o-mini-audio-preview": ModelPricing(0.15, 0.60),
                "o1": ModelPricing(15.00, 60.00, 7.50),
                "o1-pro": ModelPricing(150.00, 600.00),
                "o3-pro": ModelPricing(20.00, 80.00),
                "o3": ModelPricing(2.00, 8.00, 0.50),
                "o3-deep-research": ModelPricing(10.00, 40.00, 2.50),
                "o4-mini": ModelPricing(1.10, 4.40, 0.275),
                "o4-mini-deep-research": ModelPricing(2.00, 8.00, 0.50),
                "o3-mini": ModelPricing(1.10, 4.40, 0.55),
                "o1-mini": ModelPricing(1.10, 4.40, 0.55),
                "codex-mini-latest": ModelPricing(1.50, 6.00, 0.375),
                "gpt-4o-mini-search-preview": ModelPricing(0.15, 0.60),
                "gpt-4o-search-preview": ModelPricing(2.50, 10.00),
                "computer-use-preview": ModelPricing(3.00, 12.00),
                "gpt-image-1": ModelPricing(5.00, 0, 1.25),
                # Legacy models - Standard
                "chatgpt-4o-latest": ModelPricing(5.00, 15.00),
                "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
                "gpt-4-0125-preview": ModelPricing(10.00, 30.00),
                "gpt-4-1106-preview": ModelPricing(10.00, 30.00),
                "gpt-4-1106-vision-preview": ModelPricing(10.00, 30.00),
                "gpt-4-0613": ModelPricing(30.00, 60.00),
                "gpt-4-0314": ModelPricing(30.00, 60.00),
                "gpt-4-32k": ModelPricing(60.00, 120.00),
                "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
                "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
                "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
                "gpt-3.5-turbo-0613": ModelPricing(1.50, 2.00),
                "gpt-3.5-0301": ModelPricing(1.50, 2.00),
                "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
                "gpt-3.5-turbo-16k-0613": ModelPricing(3.00, 4.00),
                "davinci-002": ModelPricing(2.00, 2.00),
                "babbage-002": ModelPricing(0.40, 0.40),
            },
            "priority": {
                # OpenAI Models - Priority
                "gpt-5": ModelPricing(2.50, 20.00, 0.25),
                "gpt-5-mini": ModelPricing(0.45, 3.60, 0.045),
                "gpt-4.1": ModelPricing(3.50, 14.00, 0.875),
                "gpt-4.1-mini": ModelPricing(0.70, 2.80, 0.175),
                "gpt-4.1-nano": ModelPricing(0.20, 0.80, 0.05),
                "gpt-4o": ModelPricing(4.25, 17.00, 2.125),
                "gpt-4o-2024-05-13": ModelPricing(8.75, 26.25),
                "gpt-4o-mini": ModelPricing(0.25, 1.00, 0.125),
                "o3": ModelPricing(3.50, 14.00, 0.875),
                "o4-mini": ModelPricing(2.00, 8.00, 0.50),
            },
        }

        # Add Anthropic Claude models (using standard pricing as baseline)
        claude_models = {
            "claude-opus-4": ModelPricing(15.00, 75.00, 1.50),
            "claude-opus-3": ModelPricing(15.00, 75.00, 1.50),
            "claude-sonnet-3.7": ModelPricing(3.00, 15.00, 0.30),
            "claude-haiku-3": ModelPricing(0.25, 1.25, 0.03),
            "claude-opus-4.1": ModelPricing(15.00, 75.00, 1.50),
            "claude-sonnet-4": ModelPricing(3.00, 15.00, 0.30),  # ≤200K tokens
            "claude-haiku-3.5": ModelPricing(0.80, 4.00, 0.08),
            # Add common model name variations
            "claude-3-opus": ModelPricing(15.00, 75.00, 1.50),
            "claude-3-sonnet": ModelPricing(3.00, 15.00, 0.30),
            "claude-3-haiku": ModelPricing(0.25, 1.25, 0.03),
            "claude-3.5-sonnet": ModelPricing(3.00, 15.00, 0.30),
            "claude-3.5-haiku": ModelPricing(0.80, 4.00, 0.08),
        }

        # Add Claude models to standard tier
        pricing["standard"].update(claude_models)

        return pricing

    def get_tokenizer(self, model_name: str):
        """Get appropriate tokenizer for the model"""
        if model_name not in self.tokenizers:
            # Use cl100k_base for most OpenAI models, o200k_base for newer models
            if any(x in model_name.lower() for x in ["o1", "o3", "o4", "gpt-4o"]):
                encoding_name = "o200k_base"
            elif "gpt-3.5" in model_name.lower() or "gpt-4" in model_name.lower():
                encoding_name = "cl100k_base"
            elif "claude" in model_name.lower():
                # Claude uses a different tokenizer, approximate with cl100k_base
                encoding_name = "cl100k_base"
            else:
                # Default fallback
                encoding_name = "cl100k_base"

            try:
                self.tokenizers[model_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to cl100k_base if encoding not found
                self.tokenizers[model_name] = tiktoken.get_encoding("cl100k_base")

        return self.tokenizers[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for given model"""
        if not text:
            return 0

        tokenizer = self.get_tokenizer(model_name)
        return len(tokenizer.encode(str(text)))

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match pricing keys"""
        # Remove common prefixes
        normalized = model_name.lower()
        for prefix in ["openai/", "pulze/", "anthropic/"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]

        # Handle common aliases
        aliases = {
            "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
            "gpt-4": "gpt-4-0613",
            "claude-3-opus-20240229": "claude-3-opus",
            "claude-3-sonnet-20240229": "claude-3-sonnet",
            "claude-3-haiku-20240307": "claude-3-haiku",
            "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
            "claude-3-5-haiku-20241022": "claude-3.5-haiku",
        }

        return aliases.get(normalized, normalized)

    def get_model_pricing(self, model_name: str, tier: str = "standard") -> Optional[ModelPricing]:
        """Get pricing for a model in specified tier"""
        normalized_name = self.normalize_model_name(model_name)

        # Try the specified tier first
        if tier in self.model_pricing and normalized_name in self.model_pricing[tier]:
            return self.model_pricing[tier][normalized_name]

        # Fallback to other tiers
        for tier_name, models in self.model_pricing.items():
            if normalized_name in models:
                return models[normalized_name]

        return None

    def estimate_cost_for_item(
        self, item: Dict, model_name: str, tier: str = "standard", use_cached_input: bool = False
    ) -> Dict:
        """Estimate cost for a single benchmark item"""
        pricing = self.get_model_pricing(model_name, tier)
        if not pricing:
            return {
                "error": f"Pricing not found for model: {model_name}",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0,
            }

        # Extract text content based on benchmark type
        input_text = ""
        output_text = ""

        if "question" in item:
            input_text = str(item["question"])
        if "evidence" in item and isinstance(item["evidence"], list):
            # FinanceBench has evidence text
            for evidence in item["evidence"]:
                if "evidence_text" in evidence:
                    input_text += " " + str(evidence["evidence_text"])
        if "choices" in item:
            # MMLU has choices
            input_text += " " + " ".join(str(choice) for choice in item["choices"])

        if "answer" in item:
            output_text = str(item["answer"])
        if "model_answer" in item:
            output_text = str(item["model_answer"])

        # Count tokens
        input_tokens = self.count_tokens(input_text, model_name)
        output_tokens = self.count_tokens(output_text, model_name)

        # Calculate cost
        if use_cached_input and pricing.cached_input_price is not None:
            input_cost = (input_tokens / 1_000_000) * pricing.cached_input_price
        else:
            input_cost = (input_tokens / 1_000_000) * pricing.input_price

        output_cost = (output_tokens / 1_000_000) * pricing.output_price
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model_name,
            "tier": tier,
            "pricing": {
                "input_price_per_1m": pricing.cached_input_price
                if use_cached_input and pricing.cached_input_price
                else pricing.input_price,
                "output_price_per_1m": pricing.output_price,
            },
        }

    def estimate_cost_for_file(
        self, file_path: str, model_name: str, tier: str = "standard", use_cached_input: bool = False
    ) -> Dict:
        """Estimate cost for an entire benchmark file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                items = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            return {"error": f"Failed to read file {file_path}: {str(e)}"}

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        item_costs = []

        for i, item in enumerate(items):
            cost_info = self.estimate_cost_for_item(item, model_name, tier, use_cached_input)
            if "error" in cost_info:
                continue

            total_input_tokens += cost_info["input_tokens"]
            total_output_tokens += cost_info["output_tokens"]
            total_cost += cost_info["total_cost"]

            item_costs.append({"item_index": i, **cost_info})

        pricing = self.get_model_pricing(model_name, tier)

        return {
            "file_path": file_path,
            "model": model_name,
            "tier": tier,
            "total_items": len(items),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": total_cost,
            "average_cost_per_item": total_cost / len(items) if items else 0,
            "pricing_info": {
                "input_price_per_1m": pricing.cached_input_price
                if use_cached_input and pricing.cached_input_price
                else pricing.input_price,
                "output_price_per_1m": pricing.output_price,
            }
            if pricing
            else None,
            "item_costs": item_costs,
        }

    def estimate_cost_for_benchmark(
        self,
        benchmark_name: str,
        model_name: str,
        tier: str = "standard",
        use_cached_input: bool = False,
        subject: Optional[str] = None,
    ) -> Dict:
        """Estimate cost for an entire benchmark or specific subject"""
        benchmark_paths = {
            "financebench": "financebench/data/financebench_open_source.jsonl",
            "mmlu": "mmlu/data/",
            "pulze": "pulze-v0.1/data/",
            "marketing": "marketing/data/",
        }

        if benchmark_name not in benchmark_paths:
            return {"error": f"Unknown benchmark: {benchmark_name}"}

        base_path = Path(benchmark_paths[benchmark_name])
        results = {
            "benchmark": benchmark_name,
            "model": model_name,
            "tier": tier,
            "use_cached_input": use_cached_input,
            "subject": subject,
            "files": [],
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_items": 0,
        }

        if benchmark_name == "financebench":
            # Single file for financebench
            file_result = self.estimate_cost_for_file(str(base_path), model_name, tier, use_cached_input)
            if "error" not in file_result:
                results["files"].append(file_result)
                results["total_cost"] += file_result["total_cost"]
                results["total_tokens"] += file_result["total_tokens"]
                results["total_items"] += file_result["total_items"]
        else:
            # Multiple files for other benchmarks
            if not base_path.exists():
                return {"error": f"Benchmark data directory not found: {base_path}"}

            pattern = "*"
            if subject:
                if benchmark_name == "mmlu":
                    pattern = f"mmlu_{subject}.jsonl"
                elif benchmark_name == "pulze":
                    pattern = f"pulze-v0.1_{subject}.jsonl"
                elif benchmark_name == "marketing":
                    pattern = f"marketing-{subject}.jsonl"
            else:
                if benchmark_name == "mmlu":
                    pattern = "mmlu_*.jsonl"
                elif benchmark_name == "pulze":
                    pattern = "pulze-v0.1_*.jsonl"
                elif benchmark_name == "marketing":
                    pattern = "marketing-*.jsonl"

            files = list(base_path.glob(pattern))
            if not files:
                return {"error": f"No files found matching pattern: {pattern}"}

            for file_path in sorted(files):
                file_result = self.estimate_cost_for_file(str(file_path), model_name, tier, use_cached_input)
                if "error" not in file_result:
                    results["files"].append(file_result)
                    results["total_cost"] += file_result["total_cost"]
                    results["total_tokens"] += file_result["total_tokens"]
                    results["total_items"] += file_result["total_items"]

        return results

    def get_available_models(self, tier: str = None) -> List[str]:
        """Get list of available models, optionally filtered by tier"""
        if tier:
            return list(self.model_pricing.get(tier, {}).keys())

        all_models = set()
        for tier_models in self.model_pricing.values():
            all_models.update(tier_models.keys())
        return sorted(list(all_models))

    def get_available_tiers(self) -> List[str]:
        """Get list of available pricing tiers"""
        return list(self.model_pricing.keys())

    def compare_models(
        self, benchmark_name: str, models: List[str], tier: str = "standard", subject: Optional[str] = None
    ) -> Dict:
        """Compare costs across multiple models for a benchmark"""
        results = {"benchmark": benchmark_name, "subject": subject, "tier": tier, "models": []}

        for model in models:
            model_result = self.estimate_cost_for_benchmark(benchmark_name, model, tier, subject=subject)
            if "error" not in model_result:
                results["models"].append(
                    {
                        "model": model,
                        "total_cost": model_result["total_cost"],
                        "total_tokens": model_result["total_tokens"],
                        "total_items": model_result["total_items"],
                        "cost_per_item": model_result["total_cost"] / model_result["total_items"]
                        if model_result["total_items"] > 0
                        else 0,
                    }
                )

        # Sort by total cost
        results["models"].sort(key=lambda x: x["total_cost"])

        return results

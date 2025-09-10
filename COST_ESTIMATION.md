# Cost Estimation Tool for Benchmark Evaluations

This tool provides accurate cost estimates for running AI model evaluations across different benchmarks. It calculates token usage and costs based on current pricing from major AI providers.

## Features

- **Comprehensive Model Support**: Covers OpenAI GPT models, O-series models, and Anthropic Claude models
- **Multiple Pricing Tiers**: Standard, Batch, Flex, and Priority pricing tiers
- **Benchmark Coverage**: FinanceBench, MMLU, Pulze-v0.1, and Marketing benchmarks
- **Token-Accurate Estimates**: Uses tiktoken for precise token counting
- **Cached Input Support**: Accounts for cached input pricing where available
- **Comparison Tools**: Compare costs across multiple models
- **Export Options**: JSON output for integration with other tools

## Installation

The cost estimation tool is included with the evaluation framework. Ensure you have the required dependencies:

```bash
poetry install
```

## Usage

### Basic Commands

#### List Available Benchmarks

```bash
poetry run python cost_cli.py list-benchmarks
```

#### List Available Models

```bash
# All models
poetry run python cost_cli.py list-models

# Models for specific tier
poetry run python cost_cli.py list-models --tier batch
```

#### Estimate Cost for a Benchmark

```bash
# Single model estimation
poetry run python cost_cli.py estimate --benchmark mmlu --subject marketing --model gpt-4o-mini

# Multiple models comparison
poetry run python cost_cli.py estimate --benchmark financebench --model gpt-4o --model claude-3-sonnet
```

#### Compare Models

```bash
poetry run python cost_cli.py compare --benchmark marketing --models gpt-4o-mini,claude-3-haiku,gpt-3.5-turbo
```

#### Estimate for Specific File

```bash
poetry run python cost_cli.py file --file mmlu/data/mmlu_marketing.jsonl --model gpt-4o-mini
```

### Advanced Options

#### Pricing Tiers

Use different pricing tiers for cost optimization:

```bash
# Batch pricing (50% discount, higher latency)
poetry run python cost_cli.py estimate --benchmark financebench --model gpt-4o --tier batch

# Priority pricing (faster processing, higher cost)
poetry run python cost_cli.py estimate --benchmark mmlu --subject marketing --model gpt-4o --tier priority

# Flex pricing (variable pricing based on demand)
poetry run python cost_cli.py estimate --benchmark pulze --subject writing_marketing_materials --model o3 --tier flex
```

#### Cached Input Pricing

For models that support cached input pricing:

```bash
poetry run python cost_cli.py estimate --benchmark financebench --model gpt-4o --cached-input
```

#### Detailed Breakdown

Get file-by-file cost breakdown:

```bash
poetry run python cost_cli.py estimate --benchmark pulze --model gpt-4o-mini --detailed
```

#### JSON Output

Export results in JSON format for integration:

```bash
poetry run python cost_cli.py estimate --benchmark mmlu --subject marketing --model gpt-4o-mini --json
```

## Pricing Information

The tool includes current pricing for major AI models as of January 2025:

### OpenAI Models (Standard Tier, per 1M tokens)

| Model         | Input  | Output | Cached Input |
| ------------- | ------ | ------ | ------------ |
| gpt-4o        | $2.50  | $10.00 | $1.25        |
| gpt-4o-mini   | $0.15  | $0.60  | $0.075       |
| o1            | $15.00 | $60.00 | $7.50        |
| o1-mini       | $1.10  | $4.40  | $0.55        |
| gpt-3.5-turbo | $0.50  | $1.50  | -            |

### Anthropic Claude Models (Standard Tier, per 1M tokens)

| Model             | Input  | Output | Cached Input |
| ----------------- | ------ | ------ | ------------ |
| claude-3-opus     | $15.00 | $75.00 | $1.50        |
| claude-3-sonnet   | $3.00  | $15.00 | $0.30        |
| claude-3-haiku    | $0.25  | $1.25  | $0.03        |
| claude-3.5-sonnet | $3.00  | $15.00 | $0.30        |
| claude-3.5-haiku  | $0.80  | $4.00  | $0.08        |

### Pricing Tiers

- **Standard**: Regular API pricing
- **Batch**: 50% discount, higher latency (24-hour processing)
- **Flex**: Variable pricing based on demand
- **Priority**: Faster processing, higher cost

## Example Outputs

### Single Model Estimation

```
=== Cost Estimation Summary ===
Benchmark: mmlu
Model: gpt-4o-mini
Tier: standard
Subject: marketing

Total Items: 234
Total Tokens: 19.1K
Total Cost: $0.0030
Average Cost per Item: $0.0000
```

### Model Comparison

```
=== Model Cost Comparison ===
Benchmark: marketing
Tier: standard
+--------+----------------+--------------+----------+---------+-------------+
|   Rank | Model          | Total Cost   |   Tokens |   Items | Cost/Item   |
+========+================+==============+==========+=========+=============+
|      1 | gpt-4o-mini    | $0.0002      |      437 |       5 | $0.0000     |
+--------+----------------+--------------+----------+---------+-------------+
|      2 | claude-3-haiku | $0.0004      |      444 |       5 | $0.0001     |
+--------+----------------+--------------+----------+---------+-------------+
|      3 | gpt-3.5-turbo  | $0.0005      |      444 |       5 | $0.0001     |
+--------+----------------+--------------+----------+---------+-------------+
```

### High-Cost Model Example

```
=== Cost Estimation Summary ===
Benchmark: financebench
Model: o1
Tier: batch

Total Items: 150
Total Tokens: 83.6K
Total Cost: $0.693
Average Cost per Item: $0.0046
```

## Cost Optimization Tips

### 1. Choose the Right Model

- Use **gpt-4o-mini** or **claude-3-haiku** for cost-sensitive applications
- Reserve **o1** and **claude-3-opus** for complex reasoning tasks
- Consider **gpt-3.5-turbo** for simple tasks

### 2. Use Appropriate Pricing Tiers

- **Batch tier**: 50% savings for non-urgent evaluations
- **Flex tier**: Variable pricing, good for experimental work
- **Standard tier**: Balanced cost and speed
- **Priority tier**: Only when speed is critical

### 3. Leverage Cached Input Pricing

- Use `--cached-input` flag when available
- Significant savings for repeated evaluations with similar contexts

### 4. Benchmark Selection

- **MMLU**: Lowest cost per item (multiple choice)
- **Marketing**: Small dataset, good for testing
- **Pulze-v0.1**: Medium cost, comprehensive evaluation
- **FinanceBench**: Highest cost due to large context (financial documents)

## Cost Estimates by Benchmark

Based on standard tier pricing with gpt-4o-mini:

| Benchmark                   | Items | Avg Tokens/Item | Est. Cost | Cost/Item |
| --------------------------- | ----- | --------------- | --------- | --------- |
| MMLU Marketing              | 234   | 82              | $0.003    | $0.000013 |
| Marketing (all)             | 5     | 87              | $0.0002   | $0.00004  |
| Pulze-v0.1 (single subject) | ~20   | 200             | $0.001    | $0.00005  |
| FinanceBench                | 150   | 557             | $0.021    | $0.00014  |

## Integration with Evaluation Pipeline

The cost estimator can be integrated into your evaluation workflow:

```python
from utils.cost_estimator import CostEstimator

estimator = CostEstimator()

# Estimate before running evaluation
cost_info = estimator.estimate_cost_for_benchmark(
    "mmlu", "gpt-4o-mini", "standard", subject="marketing"
)

print(f"Estimated cost: ${cost_info['total_cost']:.4f}")

# Proceed with evaluation if cost is acceptable
if cost_info['total_cost'] < 0.10:  # $0.10 budget
    # Run evaluation
    pass
```

## Updating Pricing

To update model pricing, modify the `_load_model_pricing()` method in `utils/cost_estimator.py`. The pricing structure supports:

- Input token pricing
- Output token pricing
- Cached input pricing (optional)
- Multiple pricing tiers

## Limitations

1. **Token Counting**: Uses tiktoken approximation for non-OpenAI models
2. **Pricing Updates**: Manual updates required when providers change pricing
3. **Context Length**: Doesn't account for context length limits
4. **Real Usage**: Estimates may vary from actual usage due to:
   - Model-specific tokenization differences
   - API overhead
   - Retry logic
   - Rate limiting delays

## Support

For issues or feature requests related to cost estimation:

1. Check that your model name is correctly formatted
2. Verify the benchmark and subject names using `list-benchmarks`
3. Ensure pricing tier is supported for your model
4. Use `--json` output for debugging

The tool provides detailed error messages to help diagnose issues with model names, file paths, or pricing information.

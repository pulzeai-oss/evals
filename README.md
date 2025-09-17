# Unified Evaluation CLI

A comprehensive command-line interface for running evaluations across multiple benchmarks including FinanceBench, MMLU, Pulze-v0.1, and Marketing benchmarks.

## Features

- **Multi-benchmark support**: FinanceBench, MMLU, Pulze-v0.1, and Marketing
- **Flexible model support**: Works with Pulze API, OpenAI API, and other OpenAI-compatible endpoints
- **Template-based evaluation**: Support for custom evaluation templates
- **Comprehensive leaderboards**: View results across benchmarks and subjects
- **Export capabilities**: Export results to CSV, HTML formats
- **Modular architecture**: Easy to extend with new benchmarks and evaluators

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd evals
```

2. Install dependencies using Poetry:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# For development dependencies as well
poetry install --with dev
```

3. Set up environment variables:

```bash
# Copy the template and fill in your API keys
cp .env.template .env
# Edit .env with your API keys
```

## Configuration

The system uses environment variables for configuration:

### Required Environment Variables

At least one API key must be provided:

- `PULZE_API_KEY`: Your Pulze API key
- `OPENAI_API_KEY`: Your OpenAI API key (also used for other OpenAI-compatible endpoints)

### Optional Environment Variables

- `PULZE_BASE_URL`: Pulze API base URL (default: https://api.pulze.ai/v1)
- `OPENAI_BASE_URL`: OpenAI API base URL (default: https://api.openai.com/v1)
- `DEFAULT_TEMPLATE`: Default evaluation template (default: default)
- `DEFAULT_RATER_MODEL`: Default rater model (default: gpt-4)
- `RESULTS_DIR`: Results directory (default: results)
- `MAX_RETRIES`: Maximum API retries (default: 3)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 60)

### Environment Template

Create a `.env` file based on the template:

```bash
python eval_cli.py config  # This will show you current config
```

Or create an environment template:

```python
from utils import ConfigLoader
ConfigLoader.create_env_template()
```

## Usage

### Basic Commands

#### 1. List Available Benchmarks

```bash
python eval_cli.py list
```

This shows all available benchmarks and their subjects.

#### 2. Run Evaluations

**FinanceBench Evaluation:**

```bash
python eval_cli.py run --benchmark financebench --model pulze/llama-3.1-70b-instruct --rater gpt-4
```

**MMLU Marketing Evaluation:**

```bash
python eval_cli.py run --benchmark mmlu --subject marketing --model openai/gpt-4
```

**Pulze Evaluation with Template:**

```bash
python eval_cli.py run --benchmark pulze --subject writing_marketing_materials --model pulze/llama-3.1-70b-instruct --template pulze_multi_dimensional_evaluation
```

**Marketing Benchmark:**

```bash
python eval_cli.py run --benchmark marketing --subject writing_marketing_materials --model anthropic/claude-sonnet-4-0
```

#### 3. View Leaderboards

**Single Benchmark Leaderboard:**

```bash
python eval_cli.py leaderboard --benchmark marketing
```

**Cross-Benchmark Leaderboard:**

```bash
python eval_cli.py leaderboard --all
```

**Export Leaderboard:**

```bash
python eval_cli.py leaderboard --benchmark mmlu --export html
python eval_cli.py leaderboard --benchmark financebench --export csv
```

#### 4. Show Configuration

```bash
python eval_cli.py config
```

### Advanced Usage

#### Model Naming Conventions

The system automatically routes to the appropriate API based on model names:

- `pulze/model-name` → Pulze API
- `openai/model-name` → OpenAI API
- `anthropic/model-name` → OpenAI-compatible endpoint
- `gpt-3.5-turbo`, `gpt-4`, etc. → OpenAI API
- Other models → Default to OpenAI-compatible endpoint

#### Subject-Specific Evaluations

Each benchmark supports different subjects:

**MMLU Subjects (57 available):**

- `marketing`, `business_ethics`, `management`, `economics`, etc.

**Pulze-v0.1 Subjects (56 available):**

- `writing_marketing_materials`, `creative_writing`, `data_analysis`, etc.

**Marketing Subjects (3 available):**

- `writing_marketing_materials`, `Ulta Beauty`, `marketing`

**FinanceBench:**

- Single subject: `financial_analysis`

#### Template System

Templates define how evaluations are conducted:

- **Default**: Basic question-answer evaluation
- **pulze_multi_dimensional_evaluation**: Multi-dimensional scoring
- Custom templates can be added to the `pulze-v0.1/templates/` directory

## Benchmarks

### 1. FinanceBench

- **Purpose**: Financial document analysis
- **Format**: Question-answer with evidence from financial documents
- **Scoring**: AI-rated responses against expected answers
- **Data**: `financebench/data/financebench_open_source.jsonl`

### 2. MMLU (Massive Multitask Language Understanding)

- **Purpose**: Multiple-choice questions across 57 subjects
- **Format**: 4-choice multiple-choice questions
- **Scoring**: Exact match (1.0 for correct, 0.0 for incorrect)
- **Data**: `mmlu/data/mmlu_*.jsonl`

### 3. Pulze-v0.1

- **Purpose**: Template-based evaluations across 56 subjects
- **Format**: Configurable prompts with template-based scoring
- **Scoring**: AI-rated based on template criteria
- **Data**: `pulze-v0.1/data/*.jsonl`
- **Templates**: `pulze-v0.1/templates/*.json`

### 4. Marketing

- **Purpose**: Marketing-specific evaluations
- **Format**: Marketing scenarios and tasks
- **Scoring**: Multi-dimensional marketing criteria
- **Data**: `marketing/*.jsonl`

## Results and Leaderboards

### Results Storage

Results are stored in JSONL format in the `results/` directory:

```
results/
├── financebench/
│   └── model_template_rater_timestamp.jsonl
├── mmlu/
│   └── model_template_rater_subject_timestamp.jsonl
├── pulze/
│   └── model_template_rater_subject_timestamp.jsonl
└── marketing/
    └── model_template_rater_subject_timestamp.jsonl
```

### Leaderboard Features

- **Benchmark-specific leaderboards**: Compare models on individual benchmarks
- **Cross-benchmark leaderboards**: Compare models across multiple benchmarks
- **Subject breakdowns**: Detailed performance by subject
- **Export options**: HTML and CSV formats
- **Sorting options**: By average score, total score, or count

### Result Format

Each result entry contains:

```json
{
  "question_id": "unique_id",
  "question": "question_text",
  "model_answer": "model_response",
  "expected_answer": "expected_answer",
  "score": 0.85,
  "benchmark": "benchmark_name",
  "subject": "subject_name",
  "model": "model_name",
  "template": "template_name",
  "rater_model": "rater_model_name",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Architecture

### Modular Design

```
├── eval_cli.py              # Main CLI interface
├── evaluators/              # Benchmark evaluators
│   ├── __init__.py         # Evaluator factory
│   ├── base_evaluator.py   # Abstract base class
│   ├── financebench_evaluator.py
│   ├── mmlu_evaluator.py
│   ├── pulze_evaluator.py
│   └── marketing_evaluator.py
└── utils/                   # Utility modules
    ├── __init__.py
    ├── config_loader.py     # Configuration management
    ├── results_manager.py   # Results storage/loading
    └── leaderboard_generator.py  # Leaderboard generation
```

### Adding New Benchmarks

1. Create a new evaluator class inheriting from `BaseEvaluator`
2. Implement required methods: `load_data()`, `get_available_subjects()`, `evaluate_item()`
3. Add the evaluator to `evaluators/__init__.py`
4. Update CLI choices in `eval_cli.py`

### Adding New Templates

1. Create a JSON template file in `pulze-v0.1/templates/`
2. Define `system_prompt`, `prompt_template`, and `evaluation_criteria`
3. The template will be automatically available for Pulze evaluations

## Examples

### Complete Evaluation Workflow

```bash
# 1. Check configuration
python eval_cli.py config

# 2. List available benchmarks
python eval_cli.py list

# 3. Run evaluations
python eval_cli.py run --benchmark mmlu --subject marketing --model pulze/llama-3.1-70b-instruct
python eval_cli.py run --benchmark marketing --model openai/gpt-4

# 4. View results
python eval_cli.py leaderboard --benchmark mmlu
python eval_cli.py leaderboard --all

# 5. Export results
python eval_cli.py leaderboard --benchmark marketing --export html
```

### Batch Evaluation Script

```bash
#!/bin/bash
# Evaluate multiple models on marketing subjects

models=("pulze/llama-3.1-70b-instruct" "openai/gpt-4" "anthropic/claude-sonnet-4-0")

for model in "${models[@]}"; do
    echo "Evaluating $model on marketing..."
    python eval_cli.py run --benchmark marketing --model "$model"
    python eval_cli.py run --benchmark mmlu --subject marketing --model "$model"
done

# Generate comprehensive leaderboard
python eval_cli.py leaderboard --all --export html
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are set in environment variables
2. **Model Not Found**: Check model name format and API endpoint
3. **Template Not Found**: Verify template exists in `pulze-v0.1/templates/`
4. **Subject Not Found**: Use `python eval_cli.py list` to see available subjects

### Debug Mode

Set environment variable for detailed logging:

```bash
export PYTHONPATH=.
python -v eval_cli.py run --benchmark mmlu --subject marketing --model gpt-4
```

### Configuration Validation

```bash
python eval_cli.py config
```

This will show your current configuration and highlight any issues.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Activate the virtual environment
poetry shell

# Run tests
poetry run pytest tests/

# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run flake8 .
```

## License

[Add your license information here]

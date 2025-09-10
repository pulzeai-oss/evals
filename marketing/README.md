# Marketing

marketing benchmark v0.1

## Overview

This benchmark was exported from Pulze AI evaluation system and contains:

- **1 evaluation templates** with different assessment approaches
- **3 subjects** with dataset items grouped by topic/category
- **5 total dataset items** across all subjects
- **Results from 1 evaluation runs** showing model performance

## Structure

```
marketing/
├── data/                    # Dataset files (one per subject)
├── results/                 # Evaluation results (template_model.jsonl)
├── templates/              # Evaluation template configurations
├── benchmark.json          # Benchmark metadata
└── README.md              # This file
```

## Dataset Files (Subject-Based)

- `writing_marketing_materials.jsonl`: 1 items
- `Ulta Beauty.jsonl`: 1 items
- `marketing.jsonl`: 3 items

Each dataset file contains JSONL with format:

```json
{
  "question": "Question text",
  "answer": "Expected response",
  "subject": "category"
}
```

## Results Files

Results show how different models performed on each evaluation template:

**pulze_multi_dimensional_evaluation:**

- `pulze_multi_dimensional_evaluation_anthropicclaude_sonnet_4_0.jsonl`

Each results file contains JSONL with format:

```json
{
  "model_response": "Model's answer",
  "rater_response": "Evaluation feedback",
  "metrics_scores": { "accuracy": 0.9 },
  "overall_score": 0.85,
  "created_at": "2024-01-01T00:00:00"
}
```

## Templates

Template configurations show the evaluation methodology:

- Rater model used for assessment
- Evaluation prompts and criteria
- Metrics and scoring configuration

## Subject-Based Organization

This benchmark uses subject-based dataset organization, where items from multiple evaluation runs and templates are grouped by their subject/topic rather than by template. This allows for:

- **Cross-template analysis**: Compare how different evaluation approaches perform on the same subject matter
- **Subject-specific benchmarking**: Focus evaluation on specific domains or topics
- **Reduced redundancy**: Items with the same subject are merged, eliminating duplicates across different runs

## Usage

1. **Import to Pulze AI**: Drop this folder into your evals repository
2. **Custom Evaluation**: Use the templates and datasets for your own evaluation system
3. **Subject-specific Testing**: Use individual subject files to test models on specific domains
4. **Benchmarking**: Compare your models against the included baseline results

## Attribution

Exported from Pulze AI evaluation system.
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

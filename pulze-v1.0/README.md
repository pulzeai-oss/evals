# Pulze-V1 0

t

## Overview

This benchmark was exported from Pulze AI evaluation system and contains:
- **1 evaluation templates** with different assessment approaches
- **29 subjects** with dataset items grouped by topic/category
- **100 total dataset items** across all subjects
- **Results from 7 evaluation runs** showing model performance

## Structure

```
pulze-v1_0/
├── data/                    # Dataset files (one per subject)
├── results/                 # Evaluation results (template_model.jsonl)
├── templates/              # Evaluation template configurations
├── benchmark.json          # Benchmark metadata
└── README.md              # This file
```

## Dataset Files (Subject-Based)

- `abstract_algebra.jsonl`: 5 items
- `astronomy.jsonl`: 4 items
- `clinical_knowledge.jsonl`: 4 items
- `college_chemistry.jsonl`: 4 items
- `college_mathematics.jsonl`: 4 items
- `computer_security.jsonl`: 5 items
- `econometrics.jsonl`: 4 items
- `elementary_mathematics.jsonl`: 4 items
- `global_facts.jsonl`: 3 items
- `high_school_chemistry.jsonl`: 3 items
- `high_school_geography.jsonl`: 3 items
- `high_school_macroeconomics.jsonl`: 2 items
- `college_medicine.jsonl`: 4 items
- `high_school_computer_science.jsonl`: 2 items
- `high_school_microeconomics.jsonl`: 1 items
- `high_school_psychology.jsonl`: 1 items
- `college_computer_science.jsonl`: 4 items
- `high_school_biology.jsonl`: 4 items
- `college_biology.jsonl`: 4 items
- `formal_logic.jsonl`: 4 items
- `anatomy.jsonl`: 5 items
- `business_ethics.jsonl`: 4 items
- `electrical_engineering.jsonl`: 4 items
- `college_physics.jsonl`: 4 items
- `conceptual_physics.jsonl`: 4 items
- `high_school_european_history.jsonl`: 3 items
- `high_school_government_and_politics.jsonl`: 3 items
- `high_school_mathematics.jsonl`: 2 items
- `high_school_physics.jsonl`: 2 items

Each dataset file contains JSONL with format:
```json
{"question": "Question text", "answer": "Expected response", "subject": "category"}
```

## Results Files

Results show how different models performed on each evaluation template:

**pulze_multi_dimensional_evaluation_pulze_v01:**
- `pulze_multi_dimensional_evaluation_pulze_v01_openaigpt_5.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_anthropicclaude_sonnet_4_5.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_groqopenaigpt_oss_120b.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_googlegemini_25_pro.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_googlegemini_25_flash.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_groqmeta_llamallama_4_maverick_17b_128e_instruct.jsonl`
- `pulze_multi_dimensional_evaluation_pulze_v01_xaigrok_4_fast.jsonl`

Each results file contains JSONL with format:
```json
{"model_response": "Model's answer", "rater_response": "Evaluation feedback", "metrics_scores": {"accuracy": 0.9}, "overall_score": 0.85, "created_at": "2024-01-01T00:00:00"}
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

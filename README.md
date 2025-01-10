# Pulze AI Evals

Pulze AI Evals is an open-source evaluation framework designed to benchmark and assess the performance of AI models on the Pulze AI platform. Inspired by [OpenAI's Evals](https://github.com/openai/evals), our goal is to provide an easy-to-replicate and collaborative environment where developers and organizations can evaluate large language models (LLMs) or systems built using LLMs. In the words of [OpenAI's President Greg Brockman](https://twitter.com/gdb/status/1733553161884127435):

<img width="596" alt="https://x.com/gdb/status/1733553161884127435?s=20" src="https://github.com/openai/evals/assets/35577566/ce7840ff-43a8-4d88-bb2f-6b207410333b">

> **Empower your teams to build, automate, and collaborate on AI solutions with ease.**

## Overview

Pulze AI Evals allows you to:

- **Benchmark AI models** on standard datasets like **FinanceBench** and **MMLU**.
- **Collaborate** by sharing evaluations and results.
- **Customize** evaluations to suit your specific use cases.
- **Leverage Pulze AI's capabilities**, including dynamic LLM routing and advanced retrieval-augmented generation (RAG) pipelines.

By providing a centralized platform for evaluations, we aim to simplify the process of assessing AI models, ensuring you always have the right tools to make informed decisions.

## Repository Structure

- `financebench/`: Contains scripts and data for evaluating models on the FinanceBench dataset.
- `mmlu/`: Contains scripts and data for evaluating models on the MMLU benchmark.
- `results/`: Directory where evaluation results are stored.
- `README.md`: Documentation and instructions.

## Evaluations

### FinanceBench Evaluation

#### Introduction

[FinanceBench](https://github.com/patronus-ai/financebench) is a challenging benchmark designed to test AI systems using real-world financial documents. It evaluates a model's ability to extract and understand complex financial data from a large dataset.

#### Our Approach

- **Data Ingestion**: Ingested over **50,000+ pages** from the FinanceBench dataset, totaling approximately **525MB** of data, in just **1.5 hours**.
- **Space Search Requirement**: Employed **space search** to effectively find relevant information within a vast "haystack" of documents. This is crucial because the shared store evaluation simulates a scenario where the model has access to all documents, and must retrieve the pertinent information without confusion.
- **Model Utilization**: Used the **pulze-v0.1** model for dynamic LLM routing via our [knn-router](https://github.com/pulzeai-oss/knn-router), enhancing the model's ability to find the "needle in the haystack."

#### Code Explanation

- **Rater Model Selection**: We chose **gpt-4o** as the rater model due to its advanced reasoning capabilities, ensuring high-quality evaluations.
- **Correctness Threshold**: Set a threshold where a **correctness score over 6** is considered a correct answer. This calibration aligns with the scoring rubric and focuses on meaningful correctness.
- **API Usage**: Utilized Pulze AI's API with the `space-search` plugin to enable intelligent document retrieval.

#### Results

| Evaluation Mode | FinanceBench Score | Pulze AI Score |
|-----------------|--------------------|----------------|
| Shared Store    | 22%                | **74%**        |

We achieved a **236% improvement** over existing benchmarks in the Shared Store configuration, highlighting our model's exceptional performance in financial data processing.

### MMLU Evaluation

#### Introduction

The [Massive Multitask Language Understanding (MMLU)](https://crfm.stanford.edu/helm/mmlu/latest/) benchmark assesses AI performance across a broad range of subjects at different educational levels.

#### Our Approach

- **Subject Focus**: Concentrated on subjects imperative for our business customers, including economics, law, and business.
- **Agent Configuration**: Evaluated the model with and without the use of Pulze AI agents to measure their impact on performance.

#### Code Explanation

- **Dataset Preparation**: Extracted specific subjects from the MMLU dataset relevant to our customers.
- **Evaluation Metrics**: Used **gpt-4o** as the rater model, with correctness scores over 6 considered satisfactory.
- **Configurations**: Ran evaluations under two configurations:

  - **No Agent**: Direct model responses without agent assistance.
  - **With Agent**: Model responses enhanced with Pulze AI agents.

#### Results

Overall Evaluation Results:

**Configuration: no_agent**

| Metric                         | Value    |
|--------------------------------|----------|
| Percentage of Correct Answers  | 89.60%   |
| Total Questions                | 1,663    |
| Number of Correct Answers      | 1,490    |
| Number of Incorrect Answers    | 173      |

**Configuration: with_agent**

| Metric                         | Value    |
|--------------------------------|----------|
| Percentage of Correct Answers  | 86.59%   |
| Total Questions                | 1,663    |
| Number of Correct Answers      | 1,440    |
| Number of Incorrect Answers    | 223      |

Per-Subject Results:

| Subject                          | Configuration | Correct Answers | Total Questions | Percentage Correct |
|----------------------------------|---------------|-----------------|-----------------|--------------------|
| **Business Ethics**              | no_agent      | 80              | 100             | 80.00%             |
| Business Ethics                  | with_agent    | 72              | 100             | 72.00%             |
| **High School Microeconomics**   | no_agent      | 226             | 238             | 94.96%             |
| High School Microeconomics       | with_agent    | 226             | 238             | 94.96%             |
| **High School Macroeconomics**   | no_agent      | 357             | 386             | 92.49%             |
| High School Macroeconomics       | with_agent    | 336             | 386             | 87.05%             |
| **International Law**            | no_agent      | 110             | 121             | 90.91%             |
| International Law                | with_agent    | 105             | 121             | 86.78%             |
| **Management**                   | no_agent      | 90              | 103             | 87.38%             |
| Management                       | with_agent    | 90              | 103             | 87.38%             |
| **Marketing**                    | no_agent      | 221             | 234             | 94.44%             |
| Marketing                        | with_agent    | 217             | 234             | 92.74%             |
| **Professional Accounting**      | no_agent      | 252             | 281             | 89.68%             |
| Professional Accounting          | with_agent    | 251             | 281             | 89.32%             |
| **Professional Law**             | no_agent      | 154             | 200             | 77.00%             |
| Professional Law                 | with_agent    | 143             | 200             | 71.50%             |

These results demonstrate our model's proficiency across key business domains, reinforcing Pulze AI's capability to handle complex, multidisciplinary tasks.

## Pulze AI Capabilities

### Dynamic LLM Routing

We leveraged our **pulze-v0.1** model for dynamic LLM routing using our [KNN Router](https://github.com/pulzeai-oss/knn-router). This technology intelligently directs queries to the most appropriate models, enhancing performance and efficiency.

### Retrieval-Augmented Generation (RAG) Pipeline

Our advanced RAG pipeline combines both **Sparse Vector Search** and **Dense Vector Search** through a method called **Core Fusion**:

- **Sparse Vector Search**: Precise keyword matching.
- **Dense Vector Search**: Understanding related concepts.
- **Core Fusion**: Blending both methods for comprehensive results.

### Data Privacy and Security

- **SOC 2 Type 2 Compliance**: Ensuring rigorous data protection standards.
- **Selective Data Access**: Minimizing exposure of sensitive information.
- **Self-Hosted Solutions**: Offering maximum control and compliance within your environment.

## Using the Pulze AI API

Below is an example of how to interact with the Pulze AI API using a simple Python script:

```python
import json
import requests

# Set up your API key
api_key = "your-api-key"
url = "https://api.pulze.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "Pulze-Feature-Flags": '{ "auto_tools": "true" }'
}
data = {
    "plugins": ["web-search"],
    "model": "openai/gpt-4o",  # or use our pulze-v0.1 model
    "messages": [
        {"role": "user", "content": "Tell me a joke."}
    ]
}
response = requests.post(url, headers=headers, json=data)
response.raise_for_status()

# Parse the JSON response
json_response = response.json()

# Print the response
print(json.dumps(json_response, indent=4))

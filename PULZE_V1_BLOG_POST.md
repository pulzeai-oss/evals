# Building Enterprise-Grade LLM Routers: The Pulze V1.0 Technical Deep Dive

## From Benchmarks to Custom Mixture-of-Experts Routers

**TL;DR**: Pulze provides an end-to-end platform that transforms evaluation benchmarks into intelligent KNN-based routers, enabling enterprises to build custom Mixture-of-Experts (MoE) systems with complete data sovereignty and IP control.

---

## Pulze V1.0: Your Foundation, Not Your Ceiling

Much like Mistral AI's approach of releasing open-source foundation models (Mistral 7B, Mixtral) that customers then fine-tune with proprietary data to create superior domain-specific models, **Pulze v1.0 serves as a conversation starter and baseline** – not the end destination.

We've built Pulze v1.0 on publicly available MMLU benchmarks and released it openly to demonstrate the platform's capabilities. Internally, we've already built dozens of specialized routers extending v1.0's architecture, each optimized for specific use cases: code generation, medical queries, legal analysis, creative writing, and more.

### The Open Foundation Model

Pulze v1.0 is our "Mistral 7B moment" – a production-ready baseline that showcases:
- The technical feasibility of KNN-based routing
- Real performance data across 7 leading models
- A proven evaluation methodology using Router-as-a-Judge
- The complete end-to-end workflow from data to deployment

### Your Custom Models

Just as Mistral's customers don't stop at the base model, your journey doesn't end with v1.0. The platform enables you to:

**Build on the Foundation**: Start with v1.0's architecture and evaluation templates, then layer in your proprietary data – customer support tickets, internal documentation, domain-specific queries – to create routers that dramatically outperform generic solutions.

**Own Your IP**: Unlike fine-tuning a shared model, your custom routers are entirely yours. Export them, open-source them, or keep them internal. The routing intelligence you build from your data remains your competitive advantage.

**Continuous Improvement**: As you collect more interaction data, your routers improve automatically. Each query refines the routing decisions, creating a flywheel effect where your model gets smarter with use.

This is the future of enterprise AI: not choosing between vendor models, but building intelligent routing systems that leverage your unique data to orchestrate the best model for each task.

---

## The Architecture: Next-Generation KNN Routing

Pulze V1.0 represents a significant advancement in model routing technology, combining k-nearest neighbors (KNN) retrieval with modern embedding techniques to create intelligent, data-driven routing decisions.

### Technical Foundation

At its core, Pulze V1.0 uses a **hybrid retrieval system** built on three key components:

1. **Dense Embeddings** (768-dimensional): Semantic understanding via TEI (Text Embeddings Inference)
2. **Sparse Embeddings**: Token-level matching for precise query relevance
3. **Cross-Encoder Reranking**: Final scoring refinement for optimal model selection

This triple-layer approach ensures both semantic similarity and lexical precision when matching incoming prompts to historical performance data.

### The Training Data: Real-World Performance

Pulze V1.0 was trained on **7 leading language models** across **100 benchmark questions** spanning **29 MMLU subjects**:

- OpenAI GPT-5 (0.98 avg score)
- Anthropic Claude Sonnet 4.5 (0.9775 avg score)
- Google Gemini 2.5 Pro (0.9728 avg score)
- xAI Grok-4 Fast (0.9737 avg score)
- And 3 additional models

What makes this unique is the use of **Pulze v0.1 as "Router-as-a-Judge"** – our evaluation rater model that provides multi-dimensional scoring across accuracy, reasoning quality, and task completion.

---

## From Open Benchmarks to Custom Routers: The Complete Workflow

### 1. **Leverage Open-Source Benchmarks**

Our [pulzeai-oss/evals repository](https://github.com/pulzeai-oss/evals) provides production-ready evaluation benchmarks that you can use as starting points:

```bash
# Clone and explore MMLU-based benchmarks
git clone https://github.com/pulzeai-oss/evals
cd evals/pulze-v1.0/results/
```

Each evaluation result is stored as JSONL with detailed scoring:
```json
{
  "question": "What is the primary function of mitochondria?",
  "answering_model": "anthropic/claude-sonnet-4-5",
  "overall_score": 0.986,
  "accuracy": 1.0,
  "reasoning_quality": 0.95,
  "task_completion": 1.0
}
```

### 2. **Create Custom Datasets**

The platform supports three dataset creation methods:

**A. From Benchmarks**: Import existing MMLU, HELM, or custom benchmark data

**B. From Platform Interactions**: All your API interactions are logged (with zero training leakage guarantee) and can be curated into datasets

**C. Manual Creation**: Upload your proprietary prompt collections

Learn more in our [datasets documentation](https://pulze.ai/docs/pulze/data/datasets).

### 3. **Design Evaluation Templates**

Evaluation templates are the "rubrics" that define how models are scored. We've open-sourced our templates, including the multi-dimensional evaluation template used for Pulze v0.1:

```json
{
  "name": "pulze_multi_dimensional_evaluation",
  "rater_model": "pulze-v0.1",
  "dimensions": [
    {"name": "accuracy", "weight": 0.4},
    {"name": "reasoning_quality", "weight": 0.3},
    {"name": "task_completion", "weight": 0.3}
  ]
}
```

Create custom templates for your use case – whether that's code generation, creative writing, or domain-specific reasoning. Details in our [evaluations documentation](https://pulze.ai/docs/pulze/data/evaluations).

### 4. **Run Evaluations at Scale**

Execute evaluations across multiple models simultaneously:

```json
{
  "dataset_id": "your-proprietary-dataset",
  "template_id": "your-custom-template",
  "models_to_evaluate": [
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-5",
    "google/gemini-2.5-pro"
  ]
}
```

Results are automatically aggregated into prompt-level performance metrics.

### 5. **Generate Custom Routers**

Here's where the magic happens. The platform automatically:

1. **Extracts prompt-level performance** from evaluation results
2. **Resolves conflicts** using minimum scores (conservative approach)
3. **Generates hybrid embeddings** (dense + sparse) for each prompt
4. **Stores in Qdrant** with model performance metadata
5. **Creates a KNN router** that matches new queries to similar historical prompts

### 6. **Deploy as Synthetic Models**

Your custom router becomes a **synthetic model** – a Mixture-of-Experts system that:

- Routes each query to the optimal model based on KNN similarity
- Combines multiple models' strengths automatically
- Can be assigned to specific spaces or organizations
- Maintains complete data isolation and IP ownership

---

## Enterprise Features: Control and Customization

### Data Sovereignty
- **Zero Training Leakage**: Your proprietary prompts are never used for training
- **Complete Isolation**: Space and organization-level data separation
- **Full IP Ownership**: Export routers, open-source them, or keep them internal

### Visualization and QA
- **Router Performance Dashboards**: See top models across your evaluation set
- **Prompt-Level Analysis**: Drill down into individual routing decisions
- **Pre-Release Review**: Q&A interface to validate routing quality before deployment

---

## The Technical Stack

**Vector Store**: Qdrant with hybrid search
**Embeddings**: TEI (Text Embeddings Inference) for both dense and sparse vectors
**Reranking**: Cross-encoder fine-tuned for routing decisions
**Evaluation**: Pulze v0.1 as Router-as-a-Judge rater model
**Infrastructure**: Kubernetes-native with full observability

---

## Real-World Performance

Pulze V1.0 achieves remarkable routing accuracy by leveraging:
- **~100 evaluated prompts** across 29 MMLU subjects
- **7 models** with real performance scores (0.95-0.98 range)
- **Conservative conflict resolution** (minimum scores)
- **Hybrid retrieval** with 10x coarse hits multiplier + top-k reranking

The result: intelligent routing that adapts to your workload's unique characteristics.

---

## Getting Started

1. **Explore benchmarks**: Clone [pulzeai-oss/evals](https://github.com/pulzeai-oss/evals)
2. **Create datasets**: Start with our templates or build custom ones
3. **Define evaluation rubrics**: Use our open-source templates or design your own
4. **Run evaluations**: Test models on your proprietary prompts
5. **Generate routers**: One-click conversion from evaluation results to custom routers
6. **Deploy and iterate**: Monitor, refine, and optimize your routing strategy

---

## Conclusion

Pulze V1.0 demonstrates that the future of LLM deployment isn't about choosing a single model – it's about building intelligent routing systems that leverage the strengths of multiple models. By combining KNN-based retrieval with real evaluation data and enterprise-grade controls, we're enabling organizations to build truly custom Mixture-of-Experts systems tailored to their unique needs.

The platform is the complete end-to-end solution: from benchmarks to evaluations to custom routers, all with the data sovereignty and IP control that enterprises demand.

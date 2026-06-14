# Failure Taxonomy Builder

When you run a batch evaluation of 1,000 samples, you might get 150 failures. Manually reading 150 failures to figure out what went wrong takes hours.

`FailureTaxonomyBuilder` takes a list of `EvalResult` objects, extracts the `reasoning` and `evidence` from every failure, and automatically clusters them by root cause using an LLM.

## Tutorial: Clustering Failures

```python
from rageval import FailureTaxonomyBuilder, batch_evaluate
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()

# 1. Run your batch evaluation (assuming `samples` is a list of 1000 RAGSamples)
results = batch_evaluate(samples, metrics=[...])

# 2. Build the taxonomy from the results
taxonomy_builder = FailureTaxonomyBuilder(judge=judge)
taxonomy = taxonomy_builder.build(results)

# 3. Print the clusters
print(f"Found {len(taxonomy.clusters)} distinct failure modes.")

for cluster in taxonomy.clusters:
    print(f"\nCluster: {cluster.name} ({cluster.count} failures)")
    print(f"Root Cause: {cluster.trigger}")
    print(f"Suggested Fix: {cluster.fix}")
    print("Example Queries:")
    for query in cluster.example_evidence[:3]:
        print(f" - {query}")
```

## The Report Output

Instead of reading 150 unstructured failures, you get a clean taxonomy:

```text
Found 3 distinct failure modes.

Cluster: Outdated Pricing Information (85 failures)
Root Cause: The LLM is hallucinating 2023 pricing data because the retrieved documents do not explicitly state the 2024 pricing.
Suggested Fix: Inject a system prompt that strictly forbids answering pricing questions without a '2024 Pricing Guide' document in the context.
Example Queries:
 - How much is the Enterprise tier?
 - Cost of adding a new seat?
 - Volume discount pricing?

Cluster: Tabular Data Retrieval Failure (40 failures)
Root Cause: Context Precision is 0.0 for queries requiring comparative analysis across columns. The chunking strategy destroyed table structures.
Suggested Fix: Switch from recursive character chunking to an unstructured or markdown-aware chunking strategy for PDF tables.
Example Queries:
 - Compare the basic and pro plans.
 - Which plan has SLA guarantees?

Cluster: German Language Fallback (25 failures)
Root Cause: The pipeline fails to retrieve English documents when the user query is in German.
Suggested Fix: Add a query translation step (German -> English) before passing the query to the vector database.
Example Queries:
 - Was kostet der Basisplan?
 - Wie kann ich kündigen?
```

This turns "15% of our pipeline is broken" into three specific engineering tasks.

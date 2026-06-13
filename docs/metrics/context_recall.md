# Context Recall

## What it measures

Context Recall measures whether the retrieved context contains all information needed to produce the correct answer. A score of 1.0 means all claims in the ground truth answer can be found in or inferred from the retrieved context.

This metric requires `ground_truth` to be set in `RAGSample`.

## The math

```
Context Recall = |claims_found_in_context| / |total_claims_in_ground_truth|
```

## Algorithm

1. Break the ground truth answer into atomic factual claims (LLM call 1)
2. For each claim, check whether it can be found in or inferred from the retrieved context (LLM call 2)
3. Score = claims found / total claims

## Example output

```python
sample = RAGSample(
    query="Tell me about the Eiffel Tower.",
    retrieved_docs=[
        "The Eiffel Tower was constructed between 1887 and 1889.",
    ],
    answer="The Eiffel Tower was built by Gustave Eiffel.",
    ground_truth=(
        "The Eiffel Tower was constructed between 1887 and 1889. "
        "It was designed by Gustave Eiffel. "
        "The tower stands 330 meters tall."
    ),
)

result = ContextRecall(judge=judge, threshold=0.8).score(sample)
print(result.score)    # 0.67
print(result.evidence)
# ["MISSING: 'The tower stands 330 meters tall' — context does not mention the height"]
```

## What a low score tells you

Your retriever is missing important documents. Fixes:

- Increase top-k
- Improve embedding model
- Improve chunking (the relevant content may be split across chunk boundaries)

## Configuration

```python
ContextRecall(
    judge=judge,
    threshold=0.8,
)
```

## Common issues

**Requires ground_truth** — This metric will raise `ValueError` if `sample.ground_truth` is not set. Use `EvalDatasetGenerator` to generate ground truth answers from your documents automatically.

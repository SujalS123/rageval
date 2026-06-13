# Context Precision

## What it measures

Context Precision measures what fraction of the retrieved documents were actually useful for answering the query. A score of 1.0 means every retrieved document contributed. A score of 0.5 means half the retrieved documents were noise.

This metric does not require ground truth.

## The math

```
Context Precision = |useful_chunks| / |total_retrieved_chunks|
```

## Algorithm

1. For each retrieved document, ask the judge: is this document relevant to answering the query?
2. Score = relevant documents / total retrieved documents

## Example output

```python
result = ContextPrecision(judge=judge, threshold=0.7).score(sample)

print(result.score)     # 0.5
print(result.reasoning) # "1 of 2 retrieved documents were useful for answering the query."
print(result.evidence)
# ["Doc 2 NOT USEFUL: About publication dates, not authorship"]
```

With `RetrievedDoc` source tracking:

```
Doc 2 NOT USEFUL (source: bm25): About publication dates, not authorship
```

## What a low score tells you

Your retriever is returning noise. Common fixes:

- Reduce top-k (retrieve fewer documents)
- Improve your embedding model
- Improve chunking strategy (smaller, more focused chunks)
- Add a reranking step after retrieval

## Configuration

```python
ContextPrecision(
    judge=judge,
    threshold=0.7,
)
```

## Using RetrievedDoc for source tracking

```python
from rageval.core.retrieved_doc import RetrievedDoc

sample = RAGSample(
    query=query,
    retrieved_docs=[
        RetrievedDoc(content="...", source="vector", score=0.92),
        RetrievedDoc(content="...", source="bm25", score=0.61),
    ],
    answer=answer,
)
```

Evidence will include the source: `Doc 2 NOT USEFUL (source: bm25): reason here`.

## Common issues

**All documents flagged as useful** — The judge may be too lenient. Switch to a more capable model or add explicit instructions to be strict.

**All documents flagged as not useful** — Your retrieved documents may genuinely not address the query. Check your retriever.

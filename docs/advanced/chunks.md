# Chunk Quality Analysis

ChunkQualityAnalyzer analyzes chunk quality before indexing, without any LLM calls.

## Setup

```python
from rageval import ChunkQualityAnalyzer

analyzer = ChunkQualityAnalyzer()
```

## Analyzing chunks

```python
chunks = [
    "The Eiffel Tower was constructed between 1887 and 1889.",
    "It",  # too short, no context
    "The Eiffel Tower was constructed between 1887 and 1889. The Eiffel Tower was designed by Gustave Eiffel. The Eiffel Tower stands 330 meters tall. The Eiffel Tower was the tallest man-made structure for 41 years.",  # too long, should be split
]

report = analyzer.analyze(chunks)
print(report.summary())
# 1 chunk too short (< 50 tokens)
# 1 chunk too long (> 512 tokens) — consider splitting
# Average chunk length: 38 tokens
# Recommended chunk size for this corpus: 128–256 tokens
```

## Metrics computed

- Token length distribution
- Chunks below minimum useful length
- Chunks above recommended maximum
- Sentence boundary detection (does the chunk end mid-sentence?)
- Overlap detection between consecutive chunks

## Use case

Run before indexing to identify chunking problems that will cause ContextPrecision and ContextRecall to score low. Fix chunking issues here rather than debugging them through evaluation.

# Noise Sensitivity

## What it measures

Noise Sensitivity measures how robust a pipeline is when irrelevant documents are injected into the context. A score of 1.0 means the pipeline is fully robust — injecting noise had no effect on faithfulness. A low score means the pipeline degrades significantly when bad documents are retrieved.

This metric is unique to rageval. No other RAG evaluation library implements it.

## The math

```
NoiseSensitivity = 1.0 - max(0.0, clean_faithfulness - noisy_faithfulness)

Score of 1.0 = pipeline ignores noise completely
Score of 0.3 = faithfulness dropped 0.7 when noise was added
```

## Algorithm

1. Run Faithfulness on the original clean context
2. Inject `n_noise` random documents from a noise corpus and shuffle
3. Run Faithfulness again on the noisy sample
4. Score = 1.0 - degradation

## Example output

```python
noise_corpus = [
    "The French Revolution began in 1789.",
    "Photosynthesis converts sunlight to glucose.",
    "The Great Wall of China is over 13,000 miles long.",
]

result = NoiseSensitivity(
    judge=judge,
    noise_corpus=noise_corpus,
    n_noise=2,
    threshold=0.8,
).score(sample)

print(result.score)     # 0.85
print(result.reasoning) # "Clean faithfulness: 1.0. Noisy faithfulness: 0.85. Degradation: 0.15."
print(result.evidence)
# ["Noise injected: 'The French Revolution began in 1789.'",
#  "Noise injected: 'Photosynthesis converts sunlight to glucose.'"]
```

## What a low score tells you

Your pipeline is fragile. Injecting irrelevant documents causes the LLM to hallucinate. Common fixes:

- Add a reranking step to filter noise before generation
- Strengthen system prompt to ignore off-topic context
- Reduce top-k to retrieve fewer but more precise documents

## Configuration

```python
NoiseSensitivity(
    judge=judge,
    noise_corpus=noise_corpus,  # list[str] of irrelevant documents
    n_noise=2,                  # how many noise docs to inject
    threshold=0.8,
)
```

# Contradiction Detector

## What it measures

Contradiction Detector measures whether the answer directly contradicts what the retrieved context states. This is mathematically distinct from Faithfulness: Faithfulness catches claims not in context; ContradictionDetector catches claims that directly reverse what context states.

This metric does not require ground truth.

## The math

```
ContradictionDetector = 1.0 - (contradictions / total_answer_claims)
```

## Algorithm

1. Send the context and answer to the LLM
2. Ask it to find direct contradictions — places where the answer says the opposite of what context says
3. Score = 1.0 - (contradictions / total claims)

## Example output

```python
result = ContradictionDetector(judge=judge, threshold=0.8).score(sample)

print(result.score)     # 0.0
print(result.evidence)
# ["CONTRADICTION: 'The policy was approved.' | Context says: 'The policy was rejected.' | Severity: 1.0"]
```

## What a low score tells you

Your LLM is stating the opposite of what the context says. This is a more serious failure than an unsupported claim — the answer is actively misleading. Common causes:

- Negation confusion in the LLM
- Conflicting information in retrieved documents (the LLM picked the wrong one)

## Configuration

```python
ContradictionDetector(
    judge=judge,
    threshold=0.8,
)
```

## Difference from Faithfulness

Faithfulness flags: *claim not found in context* (could be an invention)

ContradictionDetector flags: *claim directly opposes context* (context explicitly says the opposite)

A claim can fail Faithfulness without being a contradiction. Use both metrics for full coverage.

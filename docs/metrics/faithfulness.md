# Faithfulness

## What it measures

Faithfulness measures whether the answer makes claims that are supported by the retrieved context. A score of 1.0 means every claim in the answer is grounded in the context. A score of 0.0 means every claim is hallucinated.

This metric does not require ground truth.

## The math

```
Faithfulness = |supported_claims| / |total_claims|
```

## Algorithm

1. Extract atomic claims from the answer (LLM call 1)
2. For each claim, verify whether the context supports it (LLM call 2)
3. Classify each unsupported claim by type and severity
4. Score = supported / total

Two separate LLM calls are used intentionally. One prompt asking for both extraction and verification produces lower quality results. Two focused prompts give the LLM one job at a time.

## Hallucination types

Each unsupported claim is classified into one of four types:

| Type | Meaning |
|---|---|
| FACTUAL_ERROR | States something verifiably false |
| UNSUPPORTED_CLAIM | States something not present in context |
| CONTRADICTION | Directly contradicts the context |
| FABRICATED_DETAIL | Invents specific numbers, names, or dates |

## Example output

```python
result = Faithfulness(judge=judge, threshold=0.8).score(sample)

print(result.score)     # 0.5
print(result.passed)    # False
print(result.reasoning) # "1 of 2 claims could not be verified from the context."
print(result.evidence)
# ["FACTUAL_ERROR: 'Romeo and Juliet was written by Charles Dickens'
#   (severity: 1.0) — Context states William Shakespeare"]
```

## What a low score tells you

Your LLM is generating claims that are not in the retrieved context. Common causes:

- System prompt does not restrict the LLM to context only
- Retrieved context is insufficient so the LLM fills gaps with parametric memory
- Context is too long and the LLM ignores parts of it

Fix by adding "Answer ONLY using the provided context" to your system prompt, or by improving retrieval quality.

## Configuration

```python
Faithfulness(
    judge=judge,
    threshold=0.8,  # fail if score < 0.8
)
```

## Accessing structured hallucinations

Faithfulness populates a `hallucinations` field with structured objects:

```python
result = Faithfulness(judge=judge).score(sample)

for h in result.hallucinations:
    print(h.claim)    # the hallucinated text
    print(h.type)     # HallucinationType enum
    print(h.severity) # float 0.0–1.0
    print(h.reason)   # why it was classified this way
```

## Common issues

**Score is always 1.0** — The claim extraction prompt is returning no claims. Check that your answer field contains actual factual statements.

**Score is always 0.0** — The context may be too short or the verification prompt is being overly strict. Try a more capable judge model.

**JSON parsing errors** — Some smaller models return malformed JSON. Use `complete_json()` which has a 3-attempt fallback. If errors persist, switch to a larger model.

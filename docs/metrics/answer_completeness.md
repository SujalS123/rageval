# Answer Completeness

## What it measures

Answer Completeness measures whether the answer covers all important information available in the context that is relevant to the query. It is the complement of Faithfulness: Faithfulness catches what the answer adds that is wrong; Completeness catches what the answer leaves out.

This metric does not require ground truth.

## The math

```
AnswerCompleteness = |mentioned_facts| / |total_relevant_facts_in_context|
```

## Algorithm

1. Extract all facts in the context relevant to the query (LLM call 1)
2. Check which of those facts the answer mentions (LLM call 2)
3. Score = mentioned / total

## Example output

```python
result = AnswerCompleteness(judge=judge, threshold=0.7).score(sample)

print(result.score)     # 0.67
print(result.evidence)
# ["MISSING FROM ANSWER: 'The policy includes a 30-day grace period' — not mentioned in answer",
#  "MISSING FROM ANSWER: 'Exceptions apply to non-profit organizations' — not mentioned in answer"]
```

## What a low score tells you

Your answer is leaving out important information that was available in the context. Common causes:

- Context is too long and the LLM summarizes instead of covering everything
- System prompt asks for concise answers, causing relevant details to be dropped

## Configuration

```python
AnswerCompleteness(
    judge=judge,
    threshold=0.7,
)
```

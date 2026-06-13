# Answer Relevancy

## What it measures

Answer Relevancy measures whether the answer addresses the original question. A score of 1.0 means the answer is directly on topic. A low score means the answer drifted off topic or ignored the question.

This metric does not require ground truth.

## The math

```
AnswerRelevancy = (1/N) × Σ cosine_similarity(original_query, generated_question_i)
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

## Algorithm

1. Given the answer, ask the LLM to generate 3 questions this answer would address
2. Compute cosine similarity between the original query and each generated question
3. Score = average cosine similarity

Reverse generation is used instead of directly asking "is this relevant?" because direct relevance scoring is inconsistent. Generating questions forces the LLM to commit to what the answer addresses, which is easier to score reliably.

## Example output

```python
result = AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=HeuristicJudge()).score(sample)

print(result.score)     # 0.88
print(result.reasoning) # "Average cosine similarity between original query and 3 reverse-generated questions: 0.88"
print(result.evidence)
# ["Generated Q1: 'Who wrote Romeo and Juliet?' — similarity: 0.97",
#  "Generated Q2: 'What author wrote Romeo and Juliet?' — similarity: 0.85",
#  "Generated Q3: 'Who is the author of Romeo and Juliet?' — similarity: 0.83"]
```

## What a low score tells you

Your answer is not addressing the question. Common causes:

- The LLM is answering what it thinks is an easier related question
- The retrieved context is misleading the LLM
- The system prompt is allowing too much freedom in response style

## Configuration

```python
from rageval.judges.heuristic import HeuristicJudge

AnswerRelevancy(
    judge=judge,               # LLM for generating questions
    threshold=0.7,
    embedding_judge=HeuristicJudge(),  # for computing similarity
)
```

The `embedding_judge` defaults to `HeuristicJudge` if not provided. It handles the cosine similarity computation using local embeddings — no API calls needed for this step.

## Common issues

**Score is high but answer is clearly irrelevant** — The LLM may be generating questions that match the query even for off-topic answers. Try a more capable judge model.

**Score varies between runs** — Ensure `temperature=0` in your judge. All built-in judges use temperature 0 by default.

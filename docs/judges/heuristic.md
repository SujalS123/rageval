# Heuristic Judge

The HeuristicJudge uses local sentence embeddings (all-MiniLM-L6-v2) for similarity computations. It makes no API calls and costs nothing to run. It is used as the `embedding_judge` in AnswerRelevancy for cosine similarity computation.

```python
from rageval.judges.heuristic import HeuristicJudge

judge = HeuristicJudge()

# Compute similarity between two texts
score = judge.similarity("What is Python?", "Python is a programming language.")
print(score)  # 0.73

# Batch similarity
scores = judge.batch_similarity(query, ["doc1", "doc2", "doc3"])
```

## Install

Built-in. No extra install needed.

The sentence-transformers model downloads automatically on first use (~90MB).

## Use as embedding judge

```python
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.judges.heuristic import HeuristicJudge

metric = AnswerRelevancy(
    judge=llm_judge,                       # LLM for generating questions
    embedding_judge=HeuristicJudge(),      # local embeddings for similarity
    threshold=0.7,
)
```

## Limitation

HeuristicJudge cannot evaluate faithfulness or context precision — those require language understanding, not just similarity. Use it only for the embedding step in AnswerRelevancy, or for fast development iteration.

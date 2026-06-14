# Consistency Analysis

A major problem with RAG systems is semantic fragility. A user asks "How do I reset my password?" and gets a perfect answer. Another user asks "Password reset instructions?" and the pipeline completely fails.

`rageval` includes a `ConsistencyAnalyzer` to measure how robust your pipeline is to phrasing variations.

## How it works

1. It takes your base query and automatically generates `N` semantic paraphrases.
2. It feeds each paraphrase into your RAG pipeline function.
3. It compares all generated answers against each other to check for contradictions or factual drift.

## Worked Example

Here is how you use the `ConsistencyAnalyzer` to test a custom RAG pipeline function.

```python
from rageval.consistency import ConsistencyAnalyzer
from rageval.judges.anthropic_judge import AnthropicJudge

# 1. Define your RAG pipeline as a standard Python function
def my_rag_pipeline(query: str) -> str:
    # Example logic using your preferred framework
    docs = my_vector_db.search(query)
    response = my_llm.generate(prompt=f"Answer {query} using {docs}")
    return response

# 2. Initialize the analyzer
judge = AnthropicJudge()
analyzer = ConsistencyAnalyzer(judge=judge)

# 3. Test a query for consistency
report = analyzer.analyze(
    base_query="What is the speed of light?",
    pipeline_fn=my_rag_pipeline,
    num_paraphrases=3  # It will generate 3 variations of the query
)

print(f"Consistency Score: {report.score}")
```

### Example Output

If your pipeline is fragile, `ConsistencyAnalyzer` will catch it:

```text
Consistency Score: 0.33
Failed Paraphrases:
1. "Can you tell me how fast light travels?"
   - Pipeline answered: "Light travels at 186,000 miles per second."
   - Contradicts base answer: "The speed of light is exactly 299,792,458 meters per second."

2. "What is light's speed limit?"
   - Pipeline answered: "I'm sorry, I don't have information about traffic limits."
   - Contradicts base answer: Answer drifted off-topic entirely.
```

By adding `ConsistencyAnalyzer` to your testing suite, you can ensure your prompt engineering and vector retrieval logic handle human unpredictability before you deploy.

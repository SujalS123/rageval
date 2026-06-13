# Your First Evaluation

This guide walks through evaluating a RAG pipeline you already have in production. It assumes your pipeline takes a query and returns retrieved documents and a generated answer.

## Step 1 — Extract your pipeline outputs

rageval operates on plain strings. It does not care if you use LangChain, LlamaIndex, or raw HTTP requests. You just need to extract the query, the retrieved documents, and the final answer.

Here is how you extract them depending on your framework:

### LangChain
```python
# Assuming you have a standard RetrievalQA chain
docs = retriever.get_relevant_documents(query)
retrieved_texts = [d.page_content for d in docs]
answer = chain.run(query)
```

### LlamaIndex
```python
response = query_engine.query(query)
retrieved_texts = [n.text for n in response.source_nodes]
answer = str(response)
```

### Custom / Raw API
```python
retrieved_texts = your_custom_retriever(query)  # list[str]
answer = generate_with_openai(query, retrieved_texts)  # str
```

## Step 2 — Construct a RAGSample

Every evaluation starts with a `RAGSample`. This is the universal contract for rageval.

```python
from rageval import RAGSample

sample = RAGSample(
    query=query,
    retrieved_docs=retrieved_texts,
    answer=answer,
    # ground_truth="The actual correct answer..."  # Only required for ContextRecall
)
```

## Step 3 — Initialize the Metrics and Judge

Next, choose your metrics and your judge. The judge is the LLM that evaluates the sample.

```python
from rageval import evaluate
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.judges.anthropic_judge import AnthropicJudge

# Reads ANTHROPIC_API_KEY from environment variables
judge = AnthropicJudge()

metrics = [
    Faithfulness(judge=judge, threshold=0.8),
    ContextPrecision(judge=judge, threshold=0.7),
    AnswerRelevancy(judge=judge, threshold=0.7),
]
```

## Step 4 — Run the Evaluation

Call `evaluate()`. This will run all metrics on your sample.

```python
result = evaluate(sample=sample, metrics=metrics)

# Print the beautiful, human-readable terminal summary
print(result.summary())
```

## Step 5 — Read the Evidence

When you run the code above, if your pipeline hallucinates, you won't just see a low score. You will see exactly *what* went wrong.

```text
-------------------------------------------------------
Query: What is the capital of France?
Overall: 0.66 | FAILED | 3210ms
-------------------------------------------------------
[FAIL] faithfulness : 0.00 (threshold=0.80)
Reasoning: 1 of 1 claims could not be verified from the context.
 Evidence:
 - FACTUAL_ERROR: 'The capital of France is London'
   (severity: 1.0) — Context states Paris
[PASS] context_precision : 1.00 (threshold=0.70)
[PASS] answer_relevancy : 1.00 (threshold=0.70)
-------------------------------------------------------
```

You now know:
1. The retriever did its job perfectly (`context_precision = 1.0`).
2. The generator ignored the context and hallucinated (`faithfulness = 0.0`).
3. The exact fabricated claim was "The capital of France is London".

## Next Steps

- Move from a single evaluation to [Batch Evaluation](../advanced/batch.md) to test hundreds of samples.
- Integrate this directly into your CI/CD pipeline using [rageval run](../cicd.md).

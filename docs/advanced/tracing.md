# RAG Pipeline Tracing

RAGTracer evaluates each step of a RAG pipeline independently and identifies the root cause step where quality first drops.

## Setup

```python
from rageval import RAGTracer
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()
tracer = RAGTracer(judge=judge)
```

## Wrapping your pipeline

```python
with tracer.trace(query="What is the boiling point of water?") as trace:
    # Step 1 — retrieval
    docs = retriever.search(query)
    trace.log_retrieval(docs=[d.page_content for d in docs])

    # Step 2 — reranking (optional)
    reranked = reranker.rerank(docs)
    trace.log_reranking(docs=[d.page_content for d in reranked])

    # Step 3 — generation
    answer = llm.generate(query, reranked)
    trace.log_generation(answer=answer)

# Evaluate the full trace
result = tracer.evaluate_trace(trace)
```

## Reading the output

```python
print(result.root_cause_step)  # "retrieval" or "reranking" or "generation"
print(result.root_cause_reason)
# "Faithfulness dropped from 1.0 (retrieval) to 0.5 (generation).
#  Root cause is in generation: LLM added claims not in context."

for step_name, step_result in result.step_results.items():
    print(f"{step_name}: faithfulness={step_result.score:.2f}")
```

## Use case

Use RAGTracer when you have a multi-step pipeline (retrieve → rerank → generate) and need to know which specific step is causing quality to drop. Without tracing you only see the final answer quality — you cannot tell if the retriever found bad documents or if the LLM hallucinated despite good context.

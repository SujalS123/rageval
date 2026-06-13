# Consistency Analysis

Inconsistent answers to semantically identical questions are one of the most common sources of user complaints in production RAG systems. Users ask "How do I reset my password?" and get a helpful guide, then later ask "Password reset instructions" and get "I don't know."

`ConsistencyAnalyzer` measures whether your pipeline returns consistent answers when the same question is asked in different ways, and helps identify *why* it failed.

## The Problem: Paraphrase Instability

Consider a user trying to find the refund policy:

1. **Query A:** "What is the refund policy?" -> Pipeline returns 30 days.
2. **Query B:** "Can I get my money back?" -> Pipeline says "I cannot answer that."

Did the retriever fail to find the document because the keywords changed? Or did the LLM just fail to generate a confident answer despite having the right documents?

## Tutorial: Real Worked Example

Here is how you use `ConsistencyAnalyzer` to catch this exact issue before users do.

```python
from rageval import ConsistencyAnalyzer, RAGSample
from rageval.judges.openai_judge import OpenAIJudge
from rageval.judges.heuristic import HeuristicJudge

# We need a judge for generation (LLM) and a judge for embeddings (Heuristic)
analyzer = ConsistencyAnalyzer(
    judge=OpenAIJudge(),
    embedding_judge=HeuristicJudge()
)

# You must wrap your existing pipeline in a function that takes a query
# and returns a RAGSample.
def my_production_pipeline(query: str) -> RAGSample:
    # 1. Your actual retriever
    docs = my_retriever.search(query)
    
    # 2. Your actual generator
    answer = my_llm.generate(query, docs)
    
    return RAGSample(
        query=query,
        retrieved_docs=docs,
        answer=answer
    )

# Run the analyzer
report = analyzer.analyze(
    query="What is the refund policy?",
    paraphrases=[
        "Can I get my money back?",
        "Tell me about refunds.",
    ],
    pipeline_fn=my_production_pipeline,
)

print(f"Consistency Score: {report.consistency_score}")
print(f"Root Cause Hypothesis:\n{report.root_cause_hypothesis}")
```

## The Report Output

If the pipeline is robust, `consistency_score` will be `1.0`.

But if we hit the paraphrase instability issue mentioned above, the output will look like this:

```text
Consistency Score: 0.33
Root Cause Hypothesis:
Low document similarity (0.41) between paraphrases suggests vocabulary mismatch in the embedding model. "Can I get my money back?" failed to retrieve the policy document that "What is the refund policy?" successfully found.

Suggested Fix:
Try query expansion (HyDE) before retrieval, or switch to an embedding model with better semantic understanding rather than keyword reliance.
```

### How it works

1. It runs `my_production_pipeline` on the original query and every paraphrase.
2. It extracts atomic claims from each answer using the LLM.
3. It cross-compares every pair of answers to find contradictions or omissions.
4. If it detects an inconsistency, it looks at the *retrieved documents*:
    - **Different documents retrieved?** The root cause is *vocabulary mismatch* (retriever issue).
    - **Same documents retrieved?** The root cause is *generation instability* (LLM/prompt issue).

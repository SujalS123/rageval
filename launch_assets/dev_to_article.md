---
title: Why RAG Evaluation Scores Are Useless (And How to Fix It)
published: false
tags: python, machinelearning, ai, opensource
---

You deployed a RAG chatbot. Users complained it was making things up. So, you ran an evaluation tool like RAGAS or DeepEval. 

You got a number back: `faithfulness: 0.43`.

Great. Now what?

A number tells you that something is wrong, but it doesn't tell you *what* is wrong or *how* to fix it. Did the LLM fabricate a specific claim? Did the retriever grab irrelevant noise? Did it contradict the provided context? 

To fix it, you still have to manually read through hundreds of LLM outputs to figure out why the score is 0.43. 

I spent two weeks doing exactly this manually. I decided there had to be a better way, so I built **[rageval](https://github.com/sujalsonawane/rageval)**.

## Not just a score. The specific evidence.

rageval is a Python library for RAG evaluation that returns the exact evidence that caused the score to drop. 

Here is what the terminal output of rageval looks like when it catches a hallucination:

```text
-------------------------------------------------------
Query: Who wrote Romeo and Juliet?
Overall: 0.00 | FAILED | 1240ms
-------------------------------------------------------
[FAIL] faithfulness : 0.00 (threshold=0.80)
Reasoning: 1 of 1 claims could not be verified from the context.
 Evidence:
 - FACTUAL_ERROR: 'Romeo and Juliet was written by Charles Dickens'
   (severity: 1.0) — Context states William Shakespeare
-------------------------------------------------------
```

Not just a number. A diagnosis. It gives you the exact hallucinated claim, the severity of the error, and what the context *actually* said.

## Framework Agnostic. Zero lock-in.

Most evaluation tools require you to use LangChain or LlamaIndex objects. rageval only accepts plain strings. It doesn't care what your tech stack is. 

```python
from rageval import evaluate, RAGSample
from rageval.metrics.faithfulness import Faithfulness
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()

result = evaluate(
    sample=RAGSample(
        query=my_query,
        retrieved_docs=["doc1", "doc2"], 
        answer=my_answer,
    ),
    metrics=[Faithfulness(judge=judge, threshold=0.8)],
)
```

## 7 Judges out of the box (including local ones)

You shouldn't be forced to use OpenAI to evaluate your pipeline. rageval supports Anthropic, Google Gemini, Cohere, Groq (Llama 3), Ollama (for running fully local, privacy-first evaluation), and a zero-cost local Heuristic judge.

You swap the judge by changing one line of code.

## Advanced Tools for Production

Evaluation isn't just about a one-off score. It's about maintaining quality over time. 

rageval comes with production-grade tools:
- **ConsistencyAnalyzer:** Checks if your pipeline gives contradictory answers when you paraphrase the same query.
- **RAGTracer:** Wraps your pipeline to pinpoint exactly which step (retrieval, reranking, or generation) caused the failure.
- **FailureTaxonomyBuilder:** Takes 1,000 failing batch evaluations and automatically clusters them by root cause, giving you a to-do list of what to fix.
- **SemanticDriftDetector:** Detects when production queries start drifting away from what your vector database actually contains.

## Try it out

rageval is open-source and MIT-licensed. 

```bash
pip install rageval
```

Check out the [GitHub repo](https://github.com/sujalsonawane/rageval) and drop a star if you find it useful. I built this to solve a pain point I experienced firsthand, and I'd love to hear your feedback on it. Let me know what you think in the comments!
